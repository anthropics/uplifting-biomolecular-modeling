# Enformer (official TensorFlow implementation; the TF-Hub SavedModel deepmind/enformer/1) optimization kit — the container image of the tested
# stack: CPython 3.11 and every pinned distribution of environment/requirements.lock (TensorFlow 2.17.1 with its CUDA 12.3 / cuDNN 8.9 libraries,
# ptxas and libdevice from the lock's nvidia-* wheels; tensorflow-hub 0.16.1), and this kit installed under /kit by its own install step. One
# image serves H100 / H200 and A100. No CUDA toolkit is installed. The GPU driver is the host's: it must run CUDA 12.3 (driver 545 or newer).
#
# Build from the release tree's top directory (the one holding enformer_deepmind/; nothing outside it is copied):
#   docker build -f enformer_deepmind/environment/Dockerfile -t enformer-deepmind-kit:dev .
# Run with the TF-Hub cache mounted and named (TFHUB_CACHE_DIR, README.md 'Setup'; nothing is fetched at run time when the cache holds the model):
#   docker run --rm --gpus all -v /path/to/tfhub-cache:/weights -e TFHUB_CACHE_DIR=/weights enformer-deepmind-kit:dev bash run.sh check
FROM python:3.11.12-slim-bookworm

COPY enformer_deepmind/environment/requirements.lock /tmp/requirements.lock
RUN grep -E '^(pip|wheel|setuptools)==' /tmp/requirements.lock > /tmp/tools.txt \
 && python3.11 -m pip install --no-cache-dir --no-deps -r /tmp/tools.txt \
 && grep -v -E '^(#|(pip|wheel|setuptools)==)' /tmp/requirements.lock > /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && rm -f /tmp/stack.txt /tmp/tools.txt /tmp/requirements.lock

# The kit: enformer_deepmind_opt installed editable and the install checked against the pins — the one install step, `run.sh install` (README.md 'Setup').
COPY enformer_deepmind /kit/enformer_deepmind
WORKDIR /kit/enformer_deepmind
RUN bash run.sh install

ENV PYTHONHASHSEED=0 \
    TF_CPP_MIN_LOG_LEVEL=1
