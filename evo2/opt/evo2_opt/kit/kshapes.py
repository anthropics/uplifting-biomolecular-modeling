"""The vortex-kernels FFT chains' shape rules (no torch): transform sizes, the storage the spectrum and the inverse transform share, and which parallel_fir call is hcm_fft_conv's."""


def fft_sizes(L: int) -> tuple:
    """(n, F): hcl_fft_conv / hcm_fft_conv transform at n = 2L; a one-sided spectrum of a length-n real signal has F = n/2 + 1 = L + 1 bins."""
    n = 2 * int(L)
    return n, n // 2 + 1


def shared_floats(B: int, D: int, L: int) -> int:
    """fp32 elements of ONE storage that holds the (B, D, F) complex64 spectrum and then, once the product has consumed it, the (B, D, n) fp32
    inverse transform: the spectrum is the larger of the two by 8·B·D bytes."""
    n, F = fft_sizes(L)
    spec, out = int(B) * int(D) * F * 2, int(B) * int(D) * n
    assert spec >= out, (spec, out)
    return spec


def resident_shape_bytes(B: int, D: int, L: int) -> dict:
    """Bytes of the shape-bound buffers of the chain at (B, L) over D channels (shared by every HCL and HCM layer)."""
    n, F = fft_sizes(L)
    return {"pad_fp32": B * D * n * 4, "spectrum_then_output_fp32": shared_floats(B, D, L) * 4, "product_complex64": B * D * F * 8}


def spectrum_bytes(D: int, L: int) -> int:
    """Bytes of one cached filter spectrum (1, D, L+1) complex64."""
    return int(D) * (int(L) + 1) * 8


def is_hcm_fft_call(*, fir_length: int, gate: bool, dim_last: bool, has_inference_params: bool, has_padding_mask: bool, has_bias: bool,
                    fir_is_conv1d: bool, column_split_hyena: bool, use_hcm_kernel: bool, hcm_bound: bool) -> bool:
    """engine.py parallel_fir reaches hcm_fft_conv for the gated inner filter of fir_length >= 128 through F.conv1d's branch when the hcm kernel is
    enabled and bound; a full-sequence scoring call (no inference params, no padding mask) with the per-channel bias the kernel adds."""
    return (fir_length >= 128 and gate and not dim_last and not has_inference_params and not has_padding_mask and has_bias
            and fir_is_conv1d and not column_split_hyena and use_hcm_kernel and hcm_bound)


SPECTRA_BUDGET = (1, 7)     # resident filter spectra take at most this fraction of their device's total memory


def spectra_budget_bytes(total_memory: int) -> int:
    return int(total_memory) * SPECTRA_BUDGET[0] // SPECTRA_BUDGET[1]


def fits_budget(resident_bytes: int, new_bytes: int, total_memory: int) -> bool:
    """Cache one more spectrum iff the resident spectra plus it stay within the budget; otherwise that layer transforms its filter per call."""
    return int(resident_bytes) + int(new_bytes) <= spectra_budget_bytes(total_memory)


def group_rows(D: int, groups) -> int:
    """Rows per filter group of a (D, 1, W) FIR weight built by repeat_interleave(D // groups, 0): R identical consecutive rows (1 when ungrouped)."""
    g = int(groups) if groups else 0
    return int(D) // g if g > 0 and int(D) % g == 0 and g < int(D) else 1


def filter_tiles(D: int, tile: int) -> list:
    """[(start, stop), ...]: the channel ranges the torch long-filter build evaluates one after the other — consecutive, each at most `tile`
    rows, covering 0..D exactly once (one range when D <= tile)."""
    D, tile = int(D), int(tile)
    assert D > 0 and tile > 0, (D, tile)
    return [(d0, min(D, d0 + tile)) for d0 in range(0, D, tile)]


def out_view_floats(B: int, D: int, L: int) -> tuple:
    """(needed, available): the inverse transform's (B, D, n) fp32 output takes the first B·D·n floats of the full-row (B, D, n) complex64
    spectrum buffer's 2·B·D·n."""
    n, _ = fft_sizes(L)
    return int(B) * int(D) * n, int(B) * int(D) * n * 2
