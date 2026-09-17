// Copyright 2024 ByteDance and/or its affiliates.
// Copyright 2020 The OneFlow Authors.
// Copyright 2021- HPC-AI Technology Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>

#include <THC/THCDeviceUtils.cuh>

#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "compat.h"
#include "type_shim.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

#define WarpNum 8
#define WarpSize 32
#define BlockSzie WarpNum*WarpSize

inline __device__ void WelfordOnline(float val, float* mean, float* m2, float* count) {
    *count += 1;
    float delta1 = val - *mean;
    *mean += delta1 / (*count);
    float delta2 = val - *mean;
    *m2 += delta1 * delta2;
}

inline __device__ void WelfordOnline(float b_mean, float b_m2, float b_count, float* mean,
                                     float* m2, float* count) {
    if (b_count == 0) {
        return;
    }
    float new_count = *count + b_count;
    float nb_n = b_count / new_count;
    float delta = b_mean - *mean;
    *mean += delta * nb_n;
    *m2 += b_m2 + delta * delta * (*count) * nb_n;
    *count = new_count;
}

__inline__ __device__ void WelfordWarpAllReduce(float thread_mean, float thread_m2,
                                                float thread_count, float* mean, float* m2,
                                                float* count) {
    *mean = thread_mean;
    *m2 = thread_m2;
    *count = thread_count;
    for (int mask = 1; mask < 32; mask *= 2) {
        float b_mean = __shfl_down_sync(0xffffffff, *mean, mask);
        float b_m2 = __shfl_down_sync(0xffffffff, *m2, mask);
        float b_count = __shfl_down_sync(0xffffffff, *count, mask);
        WelfordOnline(b_mean, b_m2, b_count, mean, m2, count);
    }

    *mean = __shfl_sync(0xffffffff, *mean, 0, 32);
    *m2 = __shfl_sync(0xffffffff, *m2, 0, 32);
    *count = __shfl_sync(0xffffffff, *count, 0, 32);
}

extern __shared__ float shared_data[];
template <typename T>
__global__ void LayerNormForward(T* input, T* output, T* gamma, T* beta, float* mean,
                                 float* invvar, int rows, int cols, double epsilon) {
    int warp_id = threadIdx.x / WarpSize;
    int lane_id = threadIdx.x % WarpSize;
    int row_offset = blockIdx.x * WarpNum + warp_id;

    T* shared_data_warp = (T*)shared_data + warp_id*cols;

    if (row_offset < rows) {
        T* row_input = input + (long long)(row_offset) * (long long)(cols); // Starting point for input data
        T* row_output = output + (long long)(row_offset) * (long long)(cols); // Starting point for output data

        float thread_mean = 0.f;
        float thread_m2 = 0.f;
        float thread_count = 0.f;

        float warp_mean;
        float warp_m2;
        float warp_count;
        // load data to shared memory
#pragma unroll
        for(int idx = lane_id; idx < cols; idx += WarpSize) {
            shared_data_warp[idx] = row_input[idx];
            WelfordOnline(static_cast<float>(shared_data_warp[idx]), &thread_mean, &thread_m2, &thread_count);
        }

        WelfordWarpAllReduce(thread_mean, thread_m2, thread_count, &warp_mean, &warp_m2,
                             &warp_count);

        float row_mean = warp_mean;
        float row_variance = max(warp_m2 / warp_count, 0.f);
        float row_inv_var = rsqrt(row_variance + epsilon);
        if (lane_id == 0) {
            mean[row_offset] = row_mean;
            invvar[row_offset] = row_inv_var;
        }
        int process_type = (gamma != NULL)*2 + (beta != NULL);
        if (process_type == 0) {
#pragma unroll
            for(int idx = lane_id; idx < cols; idx += WarpSize)
                row_output[idx] = static_cast<T>((static_cast<float>(shared_data_warp[idx]) - row_mean) * row_inv_var);
        } else if (process_type == 1) {
#pragma unroll
            for(int idx = lane_id; idx < cols; idx += WarpSize)
                row_output[idx] = static_cast<T>((static_cast<float>(shared_data_warp[idx]) - row_mean) * row_inv_var + beta[idx]);
        } else if(process_type == 2) {
#pragma unroll
            for(int idx = lane_id; idx < cols; idx += WarpSize)
                row_output[idx] = static_cast<T>((static_cast<float>(shared_data_warp[idx]) - row_mean) * row_inv_var * gamma[idx]);
        } else {
#pragma unroll
            for(int idx = lane_id; idx < cols; idx += WarpSize)
                row_output[idx] = static_cast<T>((static_cast<float>(shared_data_warp[idx]) - row_mean) * row_inv_var * gamma[idx] + beta[idx]);
        }
    }
}

void cuda_layer_norm(at::Tensor* output, at::Tensor* mean, at::Tensor* invvar, at::Tensor* input,
                     int rows, int cols, at::IntArrayRef normalized_shape, at::Tensor* gamma,
                     at::Tensor* beta, double epsilon) {
    int grid = (rows + WarpNum - 1) / WarpNum; // each warp process one line
    dim3 block(BlockSzie);
    // add shared memory size
    if (output->dtype() == torch::kFloat32) {
        int shared_meory_size = WarpNum*sizeof(float)*cols;
        LayerNormForward<float><<<grid, block, shared_meory_size>>>(
            (float*)input->data_ptr(), (float*)output->data_ptr(), 
            gamma != NULL ? (float*)gamma->data_ptr() : NULL, beta != NULL ? (float*)beta->data_ptr() : NULL, 
            (float*)mean->data_ptr(), (float*)invvar->data_ptr(), rows, cols, epsilon);
    } else if (output->dtype() == torch::kFloat16) {
        int shared_meory_size = WarpNum*sizeof(at::Half)*cols;
        LayerNormForward<at::Half><<<grid, block, shared_meory_size>>>(
            (at::Half*)input->data_ptr(), (at::Half*)output->data_ptr(),
            gamma != NULL ? (at::Half*)gamma->data_ptr() : NULL, beta != NULL ? (at::Half*)beta->data_ptr() : NULL, 
            (float*)mean->data_ptr(), (float*)invvar->data_ptr(), rows, cols, epsilon);
    } else if (output->dtype() == torch::kBFloat16) {
        int shared_meory_size = WarpNum*sizeof(at::BFloat16)*cols;
        LayerNormForward<at::BFloat16><<<grid, block, shared_meory_size>>>(
            (at::BFloat16*)input->data_ptr(), (at::BFloat16*)output->data_ptr(),
            gamma != NULL ? (at::BFloat16*)gamma->data_ptr() : NULL, beta != NULL ? (at::BFloat16*)beta->data_ptr() : NULL,
            (float*)mean->data_ptr(), (float*)invvar->data_ptr(), rows, cols, epsilon);
    }
}

template <typename T>
struct SharedMemory;

template <>
struct SharedMemory<float> {
    __device__ float* getPointer() {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template<typename T>
__inline__ __device__ T WarpReduce(T val) {
  for (int mask = 16; mask > 0; mask /= 2) { val += __shfl_xor_sync(0xffffffff, val, mask); }
  return val;
}

constexpr int tile_size = 32;
constexpr int num_per_block = 4;
constexpr int block_dim_x = 32;
constexpr int block_dim_y = 32 / num_per_block;

template <typename T, typename V>
__global__ void LayerNormParamGradStep1(int rows, int cols, const V* __restrict__ dy,
                                        const T* __restrict__ x, const float* __restrict__ mean,
                                        const float* __restrict__ inv_var,
                                        float* __restrict__ tmp_gamma_diff, float* __restrict__ tmp_beta_diff) {
  __shared__ float dgamma[32][33];
  __shared__ float dbeta[32][33];
  float dgamma_sum[num_per_block];
  float dbeta_sum[num_per_block];
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    dgamma_sum[index] = 0;
    dbeta_sum[index] = 0;
  }
  const int col_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (col_id < cols) {
    for (int i = blockIdx.y * tile_size + threadIdx.y; i < rows; i += tile_size * gridDim.y) {
#pragma unroll
      for (int index = 0; index < num_per_block; ++index) {
        int row_id = i + index * blockDim.y;
        if (row_id < rows) {
          int offset = row_id * cols + col_id;
          const float dy_val = static_cast<float>(dy[offset]);
          const float x_val = static_cast<float>(x[offset]);
          const float mean_val = mean[row_id];
          const float inv_var_val = inv_var[row_id];
          dgamma_sum[index] += dy_val * (x_val - mean_val) * inv_var_val;
          dbeta_sum[index] += dy_val;
        }
      }
    }
  }
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    dgamma[index * blockDim.y + threadIdx.y][threadIdx.x] = dgamma_sum[index];
    dbeta[index * blockDim.y + threadIdx.y][threadIdx.x] = dbeta_sum[index];
  }
  __syncthreads();
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    const int col_id = blockIdx.x * blockDim.x + threadIdx.y + index * blockDim.y;
    if (col_id < cols) {
      float gamma_sum = dgamma[threadIdx.x][threadIdx.y + index * blockDim.y];
      float beta_sum = dbeta[threadIdx.x][threadIdx.y + index * blockDim.y];
      float global_dgamma = WarpReduce<float>(gamma_sum);
      float global_dbeta = WarpReduce<float>(beta_sum);
      if (threadIdx.x == 0) {
        const int offset = blockIdx.y * cols + col_id;
        tmp_gamma_diff[offset] = global_dgamma;
        tmp_beta_diff[offset] = global_dbeta;
      }
    }
  }
}

template <typename T, typename V>
__global__ void LayerNormGammaGradStep1(int rows, int cols, const V* __restrict__ dy,
                                        const T* __restrict__ x, const float* __restrict__ mean,
                                        const float* __restrict__ inv_var, float* __restrict__ tmp_gamma_diff) {
  __shared__ float dgamma[32][33];
  float dgamma_sum[num_per_block];
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    dgamma_sum[index] = 0;
  }
  const int col_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (col_id < cols) {
    for (int i = blockIdx.y * tile_size + threadIdx.y; i < rows; i += tile_size * gridDim.y) {
#pragma unroll
      for (int index = 0; index < num_per_block; ++index) {
        int row_id = i + index * blockDim.y;
        if (row_id < rows) {
          int offset = row_id * cols + col_id;
          const float dy_val = static_cast<float>(dy[offset]);
          const float x_val = static_cast<float>(x[offset]);
          const float mean_val = mean[row_id];
          const float inv_var_val = inv_var[row_id];
          dgamma_sum[index] += dy_val * (x_val - mean_val) * inv_var_val;
        }
      }
    }
  }
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    dgamma[index * blockDim.y + threadIdx.y][threadIdx.x] = dgamma_sum[index];
  }
  __syncthreads();
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    const int col_id = blockIdx.x * blockDim.x + threadIdx.y + index * blockDim.y;
    if (col_id < cols) {
      float gamma_sum = dgamma[threadIdx.x][threadIdx.y + index * blockDim.y];
      float global_dgamma = WarpReduce<float>(gamma_sum);
      if (threadIdx.x == 0) {
        const int offset = blockIdx.y * cols + col_id;
        tmp_gamma_diff[offset] = global_dgamma;
      }
    }
  }
}

template <typename T, typename V>
__global__ void LayerNormBetaGradStep1(int rows, int cols, const V* __restrict__ dy,
                                        const T* __restrict__ x, const float* __restrict__ mean,
                                        const float* __restrict__ inv_var, float* __restrict__ tmp_beta_diff) {
  __shared__ float dbeta[32][33];
  float dbeta_sum[num_per_block];
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    dbeta_sum[index] = 0;
  }
  const int col_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (col_id < cols) {
    for (int i = blockIdx.y * tile_size + threadIdx.y; i < rows; i += tile_size * gridDim.y) {
#pragma unroll
      for (int index = 0; index < num_per_block; ++index) {
        int row_id = i + index * blockDim.y;
        if (row_id < rows) {
          int offset = row_id * cols + col_id;
          const float dy_val = static_cast<float>(dy[offset]);
          dbeta_sum[index] += dy_val;
        }
      }
    }
  }
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    dbeta[index * blockDim.y + threadIdx.y][threadIdx.x] = dbeta_sum[index];
  }
  __syncthreads();
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    const int col_id = blockIdx.x * blockDim.x + threadIdx.y + index * blockDim.y;
    if (col_id < cols) {
      float beta_sum = dbeta[threadIdx.x][threadIdx.y + index * blockDim.y];
      float global_dbeta = WarpReduce<float>(beta_sum);
      if (threadIdx.x == 0) {
        const int offset = blockIdx.y * cols + col_id;
        tmp_beta_diff[offset] = global_dbeta;
      }
    }
  }
}

template <typename V>
__global__ void LayerNormParamGradStep2(const float* part_grad_gamma, const float* part_grad_beta,
                                        const int part_size, const int n1, const int n2,
                                        V* grad_gamma, V* grad_beta) {
    // sum partial gradients for gamma and beta
    SharedMemory<float> shared;
    float* buf = shared.getPointer();
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
        // each warp does sequential reductions until reduced part_size is num_warps
        // int num_warp_reductions = part_size / blockDim.y;
        float sum_gamma = float(0);
        float sum_beta = float(0);
        const float* part_grad_gamma_ptr = part_grad_gamma + i2;
        const float* part_grad_beta_ptr = part_grad_beta + i2;
        for (int row_idx = threadIdx.y; row_idx < part_size; row_idx += blockDim.y) {
            sum_gamma += part_grad_gamma_ptr[row_idx * n2];
            sum_beta += part_grad_beta_ptr[row_idx * n2];
        }
        // inter-warp reductions
        const int nbsize3 = blockDim.x * blockDim.y / 2;
        for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
            // top half write to shared memory
            if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
                const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
                buf[write_idx] = sum_gamma;
                buf[write_idx + nbsize3] = sum_beta;
            }
            __syncthreads();
            // bottom half sums
            if (threadIdx.y < offset) {
                const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
                sum_gamma += buf[read_idx];
                sum_beta += buf[read_idx + nbsize3];
            }
            __syncthreads();
        }
        // write out fully summed gradients
        if (threadIdx.y == 0) {
            grad_gamma[i2] = sum_gamma;
            grad_beta[i2] = sum_beta;
        }
    }
}

template <typename V>
__global__ void LayerNormGammaGradStep2(const float* part_grad_gamma, const int part_size, const int n1, const int n2, V* grad_gamma) {
    // sum partial gradients for gamma and beta
    SharedMemory<float> shared;
    float* buf = shared.getPointer();
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
        // each warp does sequential reductions until reduced part_size is num_warps
        // int num_warp_reductions = part_size / blockDim.y;
        float sum_gamma = float(0);
        const float* part_grad_gamma_ptr = part_grad_gamma + i2;
        for (int row_idx = threadIdx.y; row_idx < part_size; row_idx += blockDim.y) {
            sum_gamma += part_grad_gamma_ptr[row_idx * n2];
        }
        // inter-warp reductions
        for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
            // top half write to shared memory
            if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
                const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
                buf[write_idx] = sum_gamma;
            }
            __syncthreads();
            // bottom half sums
            if (threadIdx.y < offset) {
                const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
                sum_gamma += buf[read_idx];
            }
            __syncthreads();
        }
        // write out fully summed gradients
        if (threadIdx.y == 0) {
            grad_gamma[i2] = sum_gamma;
        }
    }
}

template <typename V>
__global__ void LayerNormBetaGradStep2(const float* part_grad_beta, const int part_size, const int n1, const int n2, V* grad_beta) {
    // sum partial gradients for gamma and beta
    SharedMemory<float> shared;
    float* buf = shared.getPointer();
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
        // each warp does sequential reductions until reduced part_size is num_warps
        // int num_warp_reductions = part_size / blockDim.y;
        float sum_beta = float(0);
        const float* part_grad_beta_ptr = part_grad_beta + i2;
        for (int row_idx = threadIdx.y; row_idx < part_size; row_idx += blockDim.y) {
            sum_beta += part_grad_beta_ptr[row_idx * n2];
        }
        // inter-warp reductions
        for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
            // top half write to shared memory
            if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
                const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
                buf[write_idx] = sum_beta;
            }
            __syncthreads();
            // bottom half sums
            if (threadIdx.y < offset) {
                const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
                sum_beta += buf[read_idx];
            }
            __syncthreads();
        }
        // write out fully summed gradients
        if (threadIdx.y == 0) {
            grad_beta[i2] = sum_beta;
        }
    }
}

template <typename T, typename V>
__global__ void LayerNormInputGrad(const V* __restrict__ dout, const T* __restrict__ input,
                                   const int rows, const int cols, const float* __restrict__ mean,
                                   const float* __restrict__ invvar, float epsilon, const V* gamma,
                                   T* grad_input) {
    int WarpPerBlock = blockDim.x / WarpSize;
    int thread_idx = threadIdx.x;
    int warp_idx = thread_idx / WarpSize;
    int lane_idx = thread_idx % WarpSize;

    float* shared_dout = shared_data + warp_idx*cols;
    float* shared_input = shared_data + WarpPerBlock*cols + warp_idx*cols;
    float* shared_gamma = shared_data + 2*WarpPerBlock*cols;
    int row_stride = gridDim.x*WarpPerBlock;
    for(int row = blockIdx.x*WarpPerBlock+warp_idx; row < rows; row += row_stride) {
        float mean_r = mean[row];
        float invvar_r = invvar[row];
        // load dout, input and gamma
        long long data_offset = (long long)(row) * cols;
        const V* dout_r = dout + data_offset;
        const T* input_r = input + data_offset;
        T* grad_input_r = grad_input + data_offset;
#pragma unroll
        for(int col = lane_idx; col < cols; col += WarpSize) {
            shared_dout[col] = float(dout_r[col]);
            shared_input[col] = float(input_r[col]);
        }
        if(warp_idx == 0) {
#pragma unroll
            for(int col = lane_idx; col < cols; col += WarpSize) {
                shared_gamma[col] = gamma != NULL ? float(gamma[col]) : 1.0f;
            }
        }
        __syncthreads();

        float gamma_dout = 0.0;
        float gamma_dout_input_mean = 0.0;
        // reduction, gamma*dout and gamma*dout*(input-mean)
#pragma unroll
        for(int col = lane_idx; col < cols; col += WarpSize) {
            float temp = shared_gamma[col] * shared_dout[col];
            gamma_dout += temp;
            gamma_dout_input_mean += temp * (shared_input[col] - mean_r);
        }
        float global_gamma_dout = WarpReduce<float>(gamma_dout);
        float global_gamma_dout_input_mean = WarpReduce<float>(gamma_dout_input_mean);

        float part3_temp_value = global_gamma_dout_input_mean * invvar_r * invvar_r * invvar_r / cols;
        float part2 = global_gamma_dout * invvar_r / cols;
#pragma unroll
        for(int col = lane_idx; col < cols; col += WarpSize) {
            float part1 = shared_gamma[col] * shared_dout[col] * invvar_r;
            float part3 = (shared_input[col] - mean_r) * part3_temp_value;
            grad_input_r[col] = part1 - part2 - part3;
        }
    }
}

template <typename T, typename V>
int GetGirdDimY(const int64_t num_instances, const int64_t norm_size) {
    const int grid_dim_x = (norm_size + tile_size - 1) / tile_size;
    const int max_grid_dim_y = (num_instances + tile_size - 1) / tile_size;
    const int block_size = block_dim_x * block_dim_y;
    int max_active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, LayerNormParamGradStep1<T, V>, block_size, 0);
    int waves = 1;
    int dev;
    cudaGetDevice(&dev);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    int num_blocks = max_active_blocks * sm_count * waves;
    int grid_dim_y = std::min(max_grid_dim_y, static_cast<int>(num_blocks / grid_dim_x));
    return std::max(grid_dim_y, 1);
}

template <typename T, typename V>
void HostLayerNormGradient(const V* dout, const float* mean, const float* invvar, at::Tensor* input, int n1,
                           int n2, const V* gamma, const V* beta, double epsilon, T* grad_input,
                           V* grad_gamma, V* grad_beta) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL && beta != NULL) {
        // compute grad_gamma(j) and grad_beta(j)
        const int part_size = GetGirdDimY<T, V>(n1, n2);
        const int grid_dim_x = (n2 + tile_size - 1) / tile_size;
        const int grid_dim_y = part_size;

        at::Tensor part_grad_gamma = at::empty({part_size, n2}, input->options().dtype(at::ScalarType::Float));
        at::Tensor part_grad_beta = at::empty_like(part_grad_gamma);
        LayerNormParamGradStep1<T, V><<<dim3(grid_dim_x, grid_dim_y), dim3(32, 32 / num_per_block)>>>(
            n1, n2, dout, input->DATA_PTR<T>(), mean, invvar, part_grad_gamma.DATA_PTR<float>(), part_grad_beta.DATA_PTR<float>()
        );

        const dim3 threads3(32, 8, 1);
        const dim3 blocks3((n2 + 32 - 1) / 32, 1, 1);
        const int nshared3 = threads3.x * threads3.y * sizeof(float);
        LayerNormParamGradStep2<<<blocks3, threads3, nshared3, stream>>>(
            part_grad_gamma.DATA_PTR<float>(), part_grad_beta.DATA_PTR<float>(), part_size, n1, n2,
            grad_gamma, grad_beta);
    } else if (gamma != NULL && beta == NULL) {
        // compute grad_gamma(j) and grad_beta(j)
        const int part_size = GetGirdDimY<T, V>(n1, n2);
        const int grid_dim_x = (n2 + tile_size - 1) / tile_size;
        const int grid_dim_y = part_size;

        at::Tensor part_grad_gamma = at::empty({part_size, n2}, input->options().dtype(at::ScalarType::Float));
        LayerNormGammaGradStep1<T, V><<<dim3(grid_dim_x, grid_dim_y), dim3(32, 32 / num_per_block)>>>(
            n1, n2, dout, input->DATA_PTR<T>(), mean, invvar, part_grad_gamma.DATA_PTR<float>());

        const dim3 threads3(32, 8, 1);
        const dim3 blocks3((n2 + 32 - 1) / 32, 1, 1);
        const int nshared3 = threads3.x * threads3.y * sizeof(float);
        LayerNormGammaGradStep2<<<blocks3, threads3, nshared3, stream>>>(
            part_grad_gamma.DATA_PTR<float>(), part_size, n1, n2, grad_gamma);
    } else if (gamma == NULL && beta!= NULL) {
        // compute grad_gamma(j) and grad_beta(j)
        const int part_size = GetGirdDimY<T, V>(n1, n2);
        const int grid_dim_x = (n2 + tile_size - 1) / tile_size;
        const int grid_dim_y = part_size;

        at::Tensor part_grad_beta = at::empty({part_size, n2}, input->options().dtype(at::ScalarType::Float));
        LayerNormBetaGradStep1<T, V><<<dim3(grid_dim_x, grid_dim_y), dim3(32, 32 / num_per_block)>>>(
            n1, n2, dout, input->DATA_PTR<T>(), mean, invvar, part_grad_beta.DATA_PTR<float>()
        );

        const dim3 threads3(32, 8, 1);
        const dim3 blocks3((n2 + 32 - 1) / 32, 1, 1);
        const int nshared3 = threads3.x * threads3.y * sizeof(float);
        LayerNormBetaGradStep2<<<blocks3, threads3, nshared3, stream>>>(
            part_grad_beta.DATA_PTR<float>(), part_size, n1, n2, grad_beta);
    }

    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    #define BlockDim 128
    int WarpNumPerBlock = BlockDim / WarpSize;
    const dim3 threads1(BlockDim);
    int nshared = sizeof(float)*n2*(WarpNumPerBlock*2 + 1);

    int max_active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, LayerNormInputGrad<T, V>, BlockDim, nshared);
    int dev;
    cudaGetDevice(&dev);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);

    const dim3 blocks1(std::min((uint64_t)((n1 + WarpNumPerBlock - 1)/WarpNumPerBlock), (uint64_t)(max_active_blocks * sm_count)));
    LayerNormInputGrad<<<blocks1, threads1, nshared>>>(dout, input->DATA_PTR<T>(), n1, n2, mean, invvar, float(epsilon), gamma, grad_input);
}

void cuda_layer_norm_gradient(at::Tensor* dout, at::Tensor* mean, at::Tensor* invvar,
                              at::Tensor* input, int n1, int n2, at::IntArrayRef normalized_shape,
                              at::Tensor* gamma, at::Tensor* beta, double epsilon,
                              at::Tensor* grad_input, at::Tensor* grad_gamma,
                              at::Tensor* grad_beta) {
    using namespace at;
    DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input->scalar_type(), dout->scalar_type(), "cuda_layer_norm_gradient_kernel",
        HostLayerNormGradient(dout->DATA_PTR<scalar_t_out>(), mean->DATA_PTR<float>(),
                              invvar->DATA_PTR<float>(), input, n1, n2,
                              gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL,
                              beta != NULL ? beta->DATA_PTR<scalar_t_out>() : NULL, epsilon,
                              grad_input->DATA_PTR<scalar_t_in>(),
                              gamma != NULL ? grad_gamma->DATA_PTR<scalar_t_out>() : NULL,
                              beta != NULL ? grad_beta->DATA_PTR<scalar_t_out>() : NULL);)
}