# rfd_se3fast/geometry.py — launch geometry of the two Triton kernels (kernels.py) per CUDA compute capability.
#
# BLOCK_E is the number of edges one program owns; num_warps is 4 for both kernels on every card. The best BLOCK_E is a property of the
# card: the "9.0" row (H100) is the geometry the kernels were written and measured with; the "8.0" row (A100 80GB) is measured on that
# card at the kernel shapes one denoising step issues at 100…500 residues (radial_trunk 0.55-0.65x, radial_conv 0.57-0.79x of the 9.0
# row's kernel time there). A capability without a row takes DEFAULT's row. Pure Python: kernels.py reads it at launch, tests read it
# without torch.
#   radial_trunk  TRUNK_BLOCK_E[cc]
#   radial_conv   the first (limit, BLOCK_E) of CONV_BLOCK_E[cc] whose limit holds the padded per-edge tile JP*KP (None: no limit) —
#                 the (BLOCK_E, JP, KP) fp32 register tile a program keeps stays bounded as the tile grows
DEFAULT = "9.0"
NUM_WARPS = 4
TRUNK_BLOCK_E = {"9.0": 64, "8.0": 128}
CONV_BLOCK_E = {"9.0": ((128, 64), (256, 32), (None, 16)),
                "8.0": ((128, 128), (None, 32))}


def capability_key(capability) -> str:
    """(major, minor) as torch.cuda.get_device_capability returns it -> the row key, e.g. (8, 0) -> "8.0"."""
    return f"{int(capability[0])}.{int(capability[1])}"


def trunk_block_e(cc: str) -> int:
    return TRUNK_BLOCK_E.get(cc, TRUNK_BLOCK_E[DEFAULT])


def conv_block_e(cc: str, jp_kp: int) -> int:
    for limit, block_e in CONV_BLOCK_E.get(cc, CONV_BLOCK_E[DEFAULT]):
        if limit is None or jp_kp <= limit:
            return block_e
    raise ValueError(f"CONV_BLOCK_E[{cc!r}] has no row holding JP*KP={jp_kp} (its last limit must be None)")
