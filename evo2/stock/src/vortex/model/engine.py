# Copyright (c) 2024, Michael Poli.

import gc
import warnings

import torch
import torch.nn.functional as F

from vortex.model.utils import column_split
from vortex.logging import activations_logger

# Optional fused Triton kernels (refs #16, #76). Guarded so `import vortex`
# still works on hosts without triton.
try:
    from vortex.ops.hcs_interface import hcs_conv
    from vortex.ops.hcm_interface import hcm_fft_conv
    from vortex.ops.hcl_interface import hcl_fft_conv
except ImportError:
    hcs_conv = None
    hcm_fft_conv = None
    hcl_fft_conv = None

IIR_PREFILL_MODES = [
    "recurrence",
    "modal-fft",
    "hybrid-modal-recurrence",
    "modal-scan",
    "canonical-fft",
    "iir-fir-caching",
]


def adjust_filter_shape_for_broadcast(u, h):
    h = h.squeeze()  # Standardize to [D, L] from [1, D, L] and [D, 1, L]

    # Case: u: [B, D, L], k_f: [D, L]
    if len(u.shape) > len(h.shape):
        h = h.unsqueeze(0)

    # Case: u: [B, D1, D2, L], k_f: [B, D, L]
    if len(u.shape) > 3:
        h = h.unsqueeze(1)
    return h


def fftconv_func(
    u,
    k,
    D,
    dropout_mask,
    gelu=True,
    k_rev=None,
    bidirectional=False,
    print_activations=False,
    layer_idx=None,
    **kwargs,
):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    k_f = adjust_filter_shape_for_broadcast(u, k_f)
    k = k.squeeze()

    if bidirectional:
        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
        k, k2 = k.split(k.shape[1] // 2, dim=1)
        k2_f = torch.fft.rfft(k2, n=fft_size) / fft_size
        y1 = u_f * k_f
        y2 = u_f.conj() * k2_f.conj()

        y = torch.fft.irfft(y1 + y2, n=fft_size, norm="forward")[..., :seqlen]

    else:
        if k_rev is not None:
            k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
            k_f = k_f + k_rev_f.conj()

        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

        y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    if print_activations:
        activations_logger.info(f"post fftconv pre bias {y} {y.min()} {y.max()}")

    out = y + u * D.unsqueeze(-1)

    if print_activations:
        activations_logger.info(f"post fftconv post bias {out} {out.min()} {out.max()}")

    return out.to(dtype=u.dtype)


def canonicalize_modal_system(poles, residues):
    """Canonicalize a modal system.

    Args:
        poles (Tensor): The poles of the system.
        residues (Tensor): The residues of the system.

    Returns:
        Tuple[Tensor, Tensor]: The canonicalized poles and residues.
    """
    raise NotImplementedError


def list_tensors(idx):
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and isinstance(obj, torch.Tensor):
                # dump to log
                print(type(obj), obj.size())
                el = obj[0]
                with open(f"tensors_{idx}.txt", "a") as f:
                    f.write(f"{type(obj)} {obj.size()} {el}\n")
        except Exception:
            pass


class HyenaInferenceEngine:
    def __init__(
        self,
        fir_fn=None,
        iir_prefill_style="modal-fft",
        layer_idx=None,
        ground_truth_activations_path=None,
        print_activations=False,
        hyena_flip_x1x2=False,
        use_hcs_kernel=False,
        use_hcm_kernel=False,
        use_hcl_kernel=False,
    ) -> None:
        self.fir_fn = fir_fn
        assert iir_prefill_style in IIR_PREFILL_MODES, f"iir_prefill_style must be one of {IIR_PREFILL_MODES}"
        self.iir_prefill_style = iir_prefill_style
        self.layer_idx = layer_idx
        self.low_mem_mode = False
        self.ground_truth_activations_path = ground_truth_activations_path
        self.print_activations = print_activations
        self.hyena_flip_x1x2 = hyena_flip_x1x2
        self.use_hcs_kernel = use_hcs_kernel
        self.use_hcm_kernel = use_hcm_kernel
        self.use_hcl_kernel = use_hcl_kernel

        # A flag set with its kernel symbol None means the optional triton
        # import failed; warn rather than silently fall back to the dense
        # path. warnings dedupes by message, so this fires once per flag
        # despite per-layer engine construction.
        for flag_name, requested, kernel in (
            ("use_hcs_kernel", use_hcs_kernel, hcs_conv),
            ("use_hcm_kernel", use_hcm_kernel, hcm_fft_conv),
            ("use_hcl_kernel", use_hcl_kernel, hcl_fft_conv),
        ):
            if requested and kernel is None:
                warnings.warn(
                    f"{flag_name}=True but its Triton kernel failed to import; "
                    "falling back to the dense reference path with no speedup.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def parallel_fir(
        self,
        fir_fn,
        u,
        weight,
        bias,
        L,
        dims,
        groups=None,
        gated_bias=False,
        column_split_hyena=False,
        dim_last=True,
        fir_length=3,
        gate=False,
        inference_params=None,
        prefill_mode=None,
        padding_mask=None,
    ):
        L = u.shape[1] if dim_last else u.shape[2]
        if gate:
            hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
            # Compatibility with training infra that column splits the projections
            if column_split_hyena:
                x2, x1, v = column_split(u, num_attention_heads, hidden_size_per_attention_head)
            else:
                x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
            if self.hyena_flip_x1x2:
                x1, x2 = x2, x1

            # Opt-in HCS kernel for the gated short-filter cascade (refs #16, #76).
            if self.use_hcs_kernel and hcs_conv is not None and fir_length < 128 and groups:
                z = hcs_conv(x1, x2, v, weight, bias, gated_bias=gated_bias, padding_mask=padding_mask)
                fir_state = (x1 * v)[..., -fir_length + 1 :] if inference_params is not None else None
                return z, fir_state

            u = x1 * v

            if self.print_activations:
                activations_logger.info(f"q: {x2}, {x2.min()}, {x2.max()}")
                activations_logger.info(f"k: {x1}, {x1.min()}, {x1.max()}")
                activations_logger.info(f"v: {v}, {v.min()}, {v.max()}")
                activations_logger.info(f"post pregate: {u}, {u.min()}, {u.max()}")

        # prepare input layout, dimensions and dispatch to fir kernel
        # Deprecated
        if fir_fn != torch.nn.functional.conv1d:
            if dim_last:
                u = u.permute(0, 2, 1)  # B, D, L
            z = fir_fn(u)[:, :L]  # B, L, D

        elif fir_length >= 128:
            # Opt-in HCM FFT-conv (refs #16, #76); flag off falls back to fftconv_func.
            fftconv = (
                hcm_fft_conv
                if self.use_hcm_kernel and hcm_fft_conv is not None
                else fftconv_func
            )
            with torch.autocast("cuda"):
                z = fftconv(
                    u.to(torch.float32),
                    weight[:, :, :L].to(torch.float32),
                    bias,
                    None,
                    gelu=False,
                    bidirectional=False,
                    print_activations=self.print_activations,
                    groups=groups,
                    layer_idx=self.layer_idx,
                )
                z = z.to(u.dtype)
        else:
            if dim_last:
                u = u.permute(0, 2, 1)  # B, D, L

            if groups is None:
                g = u.shape[1]
            else:
                g = groups

            z = fir_fn(
                u.to(torch.float32),
                weight.to(torch.float32),
                bias=None,
                stride=1,
                padding=fir_length - 1,
                groups=u.shape[1],  # always set to D, regardless of filter grouping
            )[..., :L]
            if self.print_activations:
                activations_logger.info(f"post filter: {z}, {z.min()}, {z.max()}")

            z = z.to(u.dtype)

            if gated_bias is False:
                if self.print_activations:
                    activations_logger.info(f"post dw conv {z} {z.min()} {z.max()}")
                    # if self.ground_truth_activations_path:
                    #     z_savanna = torch.load(f"{self.ground_truth_activations_path}/post_dw_conv_{self.layer_idx}.pt")
                    #     z_savanna = z_savanna.permute(1, 2, 0)
                    #     z_diff = (z.squeeze() - z_savanna.squeeze()).abs().max()
                    #     activations_logger.info(f"dw_conv_diff: {z_diff}")

            if bias is not None:
                if gated_bias:
                    z = z + bias[None, :, None] * u
                else:
                    z = z + bias[None, :, None]

        # handle padding post fir, the only place with biases
        if type(padding_mask) == torch.Tensor:
            z = z * padding_mask[:, None]

        if gate:
            # if self.layer_idx == 1:
            #    breakpoint()
            z = x2 * z

            if self.print_activations:
                activations_logger.info(f"hyena filter: {weight}, {weight.min()}, {weight.max()}")
                activations_logger.info(f"post postgate: {z}, {z.min()}, {z.max()}")
                # if self.ground_truth_activations_path:
                #     q_savanna = torch.load(f"{self.ground_truth_activations_path}/q_{self.layer_idx}.pt")
                #     k_savanna = torch.load(f"{self.ground_truth_activations_path}/k_{self.layer_idx}.pt")
                #     v_savanna = torch.load(f"{self.ground_truth_activations_path}/v_{self.layer_idx}.pt")

                #     q_diff = (x2 - q_savanna).abs()
                #     k_diff = (x1 - k_savanna).abs()
                #     v_diff = (v - v_savanna).abs()

                #     activations_logger.info(f"q_diff: {q_diff.max()}, {q_diff.mean()}")
                #     activations_logger.info(f"k_diff: {k_diff.max()}, {k_diff.mean()}")
                #     activations_logger.info(f"v_diff: {v_diff.max()}, {v_diff.mean()}")

                #     h_savanna = torch.load(f"/home/zymrael/checkpoints/evo2/activations/savanna/hyena_filter_{self.layer_idx}.pt")
                #     h_diff = (weight[..., :h_savanna.shape[-1]].squeeze() - h_savanna.squeeze()).abs()

                #     activations_logger.info(f"h_diff: {h_diff.max()}, {h_diff.mean()}")

        if inference_params is not None:
            fir_state = u[..., -fir_length + 1 :]
        else:
            fir_state = None

        return z, fir_state

    def parallel_iir(
        self,
        z_pre,
        h,
        D,
        L,
        poles,
        residues,
        t,
        dims,
        layer_idx,
        inference_params=None,
        prefill_style="fft",
        fftconv_fn=None,
        padding_mask=None,
        use_flashfft=False,
        column_split_hyena=False,
        long_fir_threshold=None,
    ):
        """Compute the output state of the short convolutional filter."""
        fft_size = 2 * L
        hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
        # Compatibility with training infra that column splits the projections
        if column_split_hyena:
            z = z_pre.reshape(
                z_pre.shape[0],
                num_attention_heads,
                3 * hidden_size_per_attention_head,
                z_pre.shape[2],
            )
            x2, x1, v = (
                z[:, :, :hidden_size_per_attention_head],
                z[
                    :,
                    :,
                    hidden_size_per_attention_head : 2 * hidden_size_per_attention_head,
                ],
                z[:, :, 2 * hidden_size_per_attention_head :],
            )
            x2, x1, v = (
                x2.reshape(x2.shape[0], -1, x2.shape[-1]),
                x1.reshape(x1.shape[0], -1, x1.shape[-1]),
                v.reshape(v.shape[0], -1, v.shape[-1]),
            )
        else:
            x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)

        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        x1v = x1 * v

        # Opt-in HCL FFT-conv (refs #16, #76). Skipped during prefill, when
        # flashfft owns even-L, or when long_fir_threshold pins the depthwise path.
        _use_hcl = (
            self.use_hcl_kernel
            and hcl_fft_conv is not None
            and inference_params is None
            and long_fir_threshold is None
            and not (use_flashfft and L % 2 == 0)
        )
        if _use_hcl:
            y = hcl_fft_conv(h, x1v, x2, D, L, fft_size)
        elif inference_params is not None and prefill_style == "recurrence":
            y = self.prefill_via_direct_recurrence(
                inference_params=inference_params,
                x1v=x1v,
                L=L,
                poles=poles,
                residues=residues,
            )

        else:
            if use_flashfft and (L % 2) == 0:  # only works with even L
                y = fftconv_fn(
                    x1v.to(dtype=torch.bfloat16).contiguous(),
                    h.to(dtype=torch.float32),
                )
                X_s = None

            elif long_fir_threshold is None:
                H = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
                X_s = torch.fft.fft(x1v.to(dtype=torch.float32), n=fft_size)
                X = X_s[..., : H.shape[-1]]
                if len(z_pre.shape) > 3:
                    H = H.unsqueeze(1)
                y = torch.fft.irfft(X * H, n=fft_size, norm="forward")[..., :L]

            else:
                assert h.shape[0] == 1, "batch size must be 1 for long_fir_threshold"
                h = h[0][:, None]  # rearrange to d, 1, l for depthwise conv1d
                h = h[..., :long_fir_threshold]
                y = F.conv1d(
                    x1v,
                    h.to(dtype=x1v.dtype),
                    stride=1,
                    groups=x1v.shape[1],
                    padding=h.shape[-1] - 1,
                )[..., :L]
        # if self.layer_idx == 2:
        #    breakpoint()
        y = y.to(dtype=x1v.dtype)
        # hcl_fft_conv already applied the post-conv (y + x1v*D[:, None]) * x2.
        if not _use_hcl:
            y = (y + x1v * D.unsqueeze(-1)) * x2

        if self.print_activations:
            activations_logger.info(f"hyena filter: {h}, {h.min()}, {h.max()}")
            activations_logger.info(f"post hyena iir gate: {y}, {y.min()}, {y.max()}")
            activations_logger.info(f"q: {x2}, {x2.min()}, {x2.max()}")
            activations_logger.info(f"k: {x1}, {x1.min()}, {x1.max()}")
            activations_logger.info(f"v: {v}, {v.min()}, {v.max()}")
            # if self.ground_truth_activations_path:
            #     q_savanna = torch.load(f"{self.ground_truth_activations_path}/q_{self.layer_idx}.pt")
            #     k_savanna = torch.load(f"{self.ground_truth_activations_path}/k_{self.layer_idx}.pt")
            #     v_savanna = torch.load(f"{self.ground_truth_activations_path}/v_{self.layer_idx}.pt")

            #     q_diff = (x2 - q_savanna).abs()
            #     k_diff = (x1 - k_savanna).abs()
            #     v_diff = (v - v_savanna).abs()

            #     activations_logger.info(f"q_diff: {q_diff.max()}, {q_diff.mean()}")
            #     activations_logger.info(f"k_diff: {k_diff.max()}, {k_diff.mean()}")
            #     activations_logger.info(f"v_diff: {v_diff.max()}, {v_diff.mean()}")

            #     h_savanna = torch.load(f"/home/zymrael/checkpoints/evo2/activations/savanna/hyena_filter_{self.layer_idx}.pt")

            #     h_diff = (h[..., :h_savanna.shape[-1]].squeeze() - h_savanna.squeeze()).abs()
            #     activations_logger.info(f"h_diff: {h_diff.max()}, {h_diff.mean()}")

        if inference_params is not None:
            if prefill_style == "fft":
                self.prefill_via_modal_fft(
                    inference_params=inference_params,
                    x1v=x1v,
                    X_s=X_s,
                    L=L,
                    t=t,
                    poles=poles,
                    dims=dims,
                    layer_idx=layer_idx,
                    use_flashfft=use_flashfft,
                    fftconv_fn=fftconv_fn,
                )

            elif prefill_style == "recurrence":
                # recurrent prefill is done before
                pass
            else:
                raise NotImplementedError
            if self.low_mem_mode:
                # TODO: smarter gc
                del z_pre, x2, x1, v, x1v, h, poles, residues
                torch.cuda.empty_cache()

        return y.permute(0, 2, 1)

    def step_fir(self, u, fir_state, weight, bias=None, gated_bias=False, flip_filter=False):
        """Steps forward FIR filters in the architecture.

        FIR filters generally include truncated convolutions in Hyena with an explicit or hybrid time-domain parametrization:
        * Short FIR filters in Hyena featurizers
        * Short and medium FIR filters in Hyena operators

        Note:
            `fir_state` contains the last FIR filter length - 1 elements of `u`: `u_(L-2), u_{L-1), ...`
            We assume dimensions of `short_filter_weight` to be `[d, 1, short_filter_len]`.
        """
        weight = weight.squeeze()

        cache_size = fir_state.shape[-1]
        filter_length = weight.shape[-1]
        if flip_filter:
            weight = weight.flip(-1)
            weight = weight[..., -cache_size - 1 :].unsqueeze(0)
        else:
            weight = weight[..., : cache_size + 1].unsqueeze(0)

        input_dtype = u.dtype
        weight = weight.to(torch.float32)
        u = u.to(torch.float32)
        fir_state = fir_state.to(torch.float32)
        bias = bias.to(torch.float32) if bias is not None else None

        h0, h = weight[..., -1], weight[..., :-1]
        y = h0 * u + torch.sum(fir_state * h, dim=-1)

        if bias is not None:
            if gated_bias:
                y = y + bias * u
            else:
                y = y + bias

        # Update the state
        if cache_size < filter_length - 1:
            fir_state = torch.cat([fir_state, u[..., None]], dim=-1)
        else:
            fir_state = torch.roll(fir_state, -1, dims=2)
            fir_state[..., -1] = u

        return y.to(input_dtype), fir_state

    def step_iir(self, x2, x1, v, D, residues, poles, iir_state, iir_groups=1):
        # TODO: kernelize
        x1v = x1 * v
        poles = torch.exp(poles)  # poles arg contains log_poles
        poles = poles[..., 0][None]  # squeeze dummy seqlen dim and add dummy batch dim
        residues = residues[None]  # add dummy batch dim
        iir_state = poles * iir_state + x1v[..., None]

        res_state = torch.sum(residues * iir_state, dim=-1)

        if iir_groups > 1:
            raise NotImplementedError
        # if self.layer_idx == 2:
        #    breakpoint()
        y = x2 * (res_state + D * x1v)

        return y, iir_state

    def prefill_via_fir_caching(self, u, inference_params, L, *args, **kwargs):
        """Turns the IIR filter into a FIR and uses a cache for decoding."""
        raise NotImplementedError(":)")

    def prefill_via_direct_recurrence(self, inference_params, x1v, L, residues, poles, *args, **kwargs) -> torch.Tensor:
        """
        Compute the IIR state via explicit recurrence (modal form)

        This is the most memory efficient prefilling method for Hyena filters.

        Note:
            dtypes: [state: float32, poles: float32, x1v: bfloat16, output: bfloat16]
        """
        state_dim = poles.shape[1]
        x1v_ = x1v[..., None, None]  # b, d, l, sdim, reim
        x1v_ = x1v_.repeat(1, 1, 1, state_dim, 2)  # b, d, l, sdim, reim
        x1v_[..., 1] = 0

        state = 0 * x1v_[:, :, 0]
        output = 0 * x1v_[:, :, :, 0, 0]  # b, d, l

        # suppress dummy seqlen dimension
        poles = poles[:, :, 0][None]
        residues = residues[:, :, 0][None].repeat(x1v_.shape[0], 1, 1, 1)  # b, d, sdim, reim

        # state: b, d, sdim, reim
        # poles: 1, d, sdim, reim
        # x1v_: b, d, l, sdim, reim
        for i in range(L):
            state[..., 0] = poles[..., 0] * state[..., 0] - poles[..., 1] * state[..., 1] + x1v_[:, :, i, :, 0]
            state[..., 1] = poles[..., 0] * state[..., 1] + poles[..., 1] * state[..., 0] + x1v_[:, :, i, :, 1]
            output[:, :, i] = torch.sum(residues * state, dim=-2)[..., 0]  # .real

        inference_params.state_dict[self.layer_idx] = state.to(dtype=torch.float32)

        return output

    def prefill_via_hybrid_recurrence(self, inference_params, u, log_poles, x1v_f_a, L, *args, **kwargs):
        """
        Compute the IIR state via hybrid recurrence-convolution over blocks
        """
        raise NotImplementedError(":)")

    def prefill_via_scan(self, u, inference_params=None, *args, **kwargs):
        raise NotImplementedError

    def prefill_via_canonical_fft(self, u, inference_params=None, *args, **kwargs):
        """
        Compute the IIR state via a single FFT

        This is the most memory efficient "parallelized" prefilling method for Hyena.

        From: https://arxiv.org/abs/2310.18780
        """
        raise NotImplementedError(":)")

    def prefill_via_modal_fft(
        self,
        inference_params,
        x1v,
        L,
        poles,
        t,
        dims,
        layer_idx,
        X_s=None,
        use_flashfft=False,
        fftconv_fn=None,
        state_dtype=torch.float32,
        *args,
        **kwargs,
    ):
        """
        Compute the IIR state via a single FFT
        """
        # When the model has a long convolution derived from a recurrence in modal form and prefill_style is "fft",
        # we split the filter into poles and residues and reuse FFT computation on the input.
        hidden_size, _, _, state_size, hyena_filter_groups = dims

        assert X_s is not None
        bs = x1v.shape[0]
        fft_size = 2 * L
        # poles = torch.view_as_complex(poles.to(torch.float32))
        state_s = (poles.to(torch.float32) * t).exp()

        # state_s = poles**t
        state_S = torch.fft.fft(state_s, n=fft_size).repeat(bs, 1, 1, 1)  # B, D, state_dim, 2 * L
        if hyena_filter_groups > 1:
            state_S = state_S.repeat_interleave(hidden_size // hyena_filter_groups, 1)
        state = torch.fft.ifft(X_s[..., None, :] * state_S, n=fft_size)
        inference_params.state_dict[layer_idx] = state[..., L - 1].to(dtype=state_dtype)

    def _compute_state(self, log_poles, u, t, L, *args, **kwargs):
        """
        Compute the IIR state given an input `u` and log_poles of the modal system.
        """
        bs = u.shape[0]
        fft_size = 2 * L
        U = torch.fft.rfft(u.to(torch.float32), n=fft_size)
        fft_size = 2 * L
        x = (log_poles * t).exp()
        # [batch, hidden_size, state_dim, 2 * seqlen]
        X = torch.fft.fft(x, n=fft_size).repeat(bs, 1, 1, 1)
        state = torch.fft.ifft(U[..., None, :] * X, n=fft_size)[..., :L]
        return state


class HyenaFilter:
    """Handles Hyena filter computations including FFT and direct convolution."""

    def __init__(self, use_flash_fft=False):
        self.use_flash_fft = use_flash_fft

    def fft_conv(self, u, k, D, **kwargs):
        """FFT-based convolution implementation."""
        seqlen = u.shape[-1]
        fft_size = 2 * seqlen

        k_f = self._prepare_filter(k, u, fft_size)
        y = self._compute_fft_conv(u, k_f, fft_size, seqlen, **kwargs)

        return y + u * D.unsqueeze(-1)

    def _prepare_filter(self, k, u, fft_size):
        """Prepare filter for FFT convolution."""
        k_f = torch.fft.rfft(k, n=fft_size) / fft_size
        return adjust_filter_shape_for_broadcast(u, k_f)
