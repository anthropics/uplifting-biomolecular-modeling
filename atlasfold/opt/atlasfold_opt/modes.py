"""Modes and the lever row each one applies. Every lever is an in-process patch of a named stock attribute (registry.LEVERS[<id>]["site"]),
applied by stack.activate in the row's order. One literal per mode row; MODE_NAMES is one
literal line (it is read as text as well as imported)."""
from typing import Dict, List, Tuple
MODE_NAMES: Tuple[str, ...] = ("off", "exact", "fast", "big")   # the modes this kit ships, the stock word first — ONE literal line (tools outside this tree parse it)
OFF, EXACT, FAST, BIG = MODE_NAMES

_FAST_ROW = ["trimul_v4", "flash_triattn", "triattn_exact", "triatt_block_exact", "triatt_block", "triattn_core", "lm_sdpa", "sampler_hostsync", "diffusion_bf16", "sampler_hoist", "atom_sdpa", "atom_tf32", "atom_kdedup", "atom_rows", "atom_bf16", "denoiser_graph", "graph_reuse", "ln_bf16",
             "pae_stream", "pae_stream_m", "conf_transition_chunk", "transition_exact", "pair_transition_chunk", "pair_transition", "pair_block_residual", "distogram_offload", "relpos_lazy", "output_overlap", "alloc_expandable", "lever_report", "exactln", "dit_apb"]
_BIG_ROW = [l for l in _FAST_ROW if l not in ("denoiser_graph", "graph_reuse")]     # big = fast minus the CUDA-graph lever (denoiser_graph: its private graph pool is extra reserved memory) and the lever that requires it (graph_reuse); every memory lever (pae_stream, pae_stream_m, conf_transition_chunk, pair_transition_chunk, distogram_offload, relpos_lazy, alloc_expandable) rides in all three rows
MODES: Dict[str, List[str]] = {
    EXACT: [                                      # stock arithmetic reorganized: byte-identical outputs vs off under --det 1
        "trimul_exact", "triattn_exact", "triatt_block_exact", "ln_bf16", "pae_stream", "pae_stream_m", "conf_transition_chunk", "transition_exact", "pair_transition_chunk", "distogram_offload", "relpos_lazy", "sampler_hostsync", "sampler_hoist", "atom_kdedup", "output_overlap", "alloc_expandable", "lever_report", "exactln", "dit_apb",
    ],
    FAST: list(_FAST_ROW),                        # the exact row with trimul_v4 in place of trimul_exact, plus the fast-class kernel and numerics levers: small documented numeric differences vs off
    BIG: list(_BIG_ROW),                      # the memory mode: fast minus denoiser_graph / graph_reuse (the CUDA-graph pool's reserved memory); fast-class numerics
    OFF: [],                                      # stock atlasfold: nothing set, nothing applied
}
DEFAULT_MODE = OFF
PLANNED: Dict[str, List[str]] = {                      # named, NOT applied (check prints them as state=planned); never printed as on
    EXACT: ["ztrunk_lazy_fp32"],
    FAST: [],
    BIG: [],
}


def levers_of(mode: str) -> List[str]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODE_NAMES}, not {mode!r}")
    return list(MODES[mode])
