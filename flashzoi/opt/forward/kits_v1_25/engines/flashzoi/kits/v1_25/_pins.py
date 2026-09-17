"""The kit's OWN constants and exceptions for the USER PATH (import, apply, install, forward): nothing outside the kit is imported."""
SEQ_LEN = 524288
OUTPUT_SHAPE = (7611, 6144)          # (N_TRACKS, N_BINS): the human head's 7611 tracks x the 6144 centre bins
AUTOCAST_DTYPE = "float16"           # THE PIN (the card's sampling_constants.autocast_dtype)
STEM_CHANNELS = 512                  # conv_dna's filters: the stage-1 kernel stores (n * SEQ_LEN//2 + q) * 512 + c at int32 offsets (2**27 elements per window)
TOWER1_CHANNELS = 608                # res_tower.0's filters: the first fused pool->BatchNorm->GELU site reads that conv's output, 608 x SEQ_LEN//2 elements per
                                     # window — indexed per window at int32 offsets from a 64-bit window base, so it does not bound the batch
STOCK_MAX_BATCH = 15                 # upstream's own forward at SEQ_LEN serves at most 15 windows in one call (torch.max_pool1d refuses a 16-window batch:
                                     # 'integer out of range'); the kit dispatches up to that many windows at once and larger batches in chunks (_wrap.kit_max_batch)


class PinDrift(RuntimeError):
    """A pin of the kit does not hold on this machine/model/call — refused by name."""


class KitPathRefused(PinDrift):
    """A documented surface the kit refuses by name."""


def canonical(out):
    """One member's (T, B) float32 C-contiguous host array: exactly the bytes the forward produced (no rounding), through the same pageable
    D2H copy the package's predict_tracks uses (.cpu().numpy()) ."""
    import numpy as np
    return np.ascontiguousarray(out.detach().to("cpu", copy=True).numpy(), dtype=np.float32)
