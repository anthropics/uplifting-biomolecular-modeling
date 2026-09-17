"""The patch sites. Each module exposes ``install(mode, tag, ctx) -> Installed`` (patches applied, ledgers to report) and keeps the stock
callable as ``__wrapped_stock__`` on the wrapper (read by the kit's unit tests and by levers that need the callable beneath another
lever's wrapper). No upstream file is edited: attributes of the imported stock modules are rebound in this process only."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class LeverAborted(RuntimeError):
    """A lever ended the item BY NAME (today: denoiser_graph after a failed CUDA-graph capture, from which torch cannot continue). Carries the
    lever id and its reason word; `run.sh pred` prints `REFUSED: <lever> <reason> (...)` and finishes with the refusal exit code 3; an API caller
    (atlasfold_opt.enable + the stock main in its own process) catches `atlasfold_opt.LeverAborted` and maps it to the same class."""
    def __init__(self, lever: str, reason: str, message: str, hint: str = ""):
        super().__init__(message)
        self.lever, self.reason, self.hint = lever, reason, hint


@dataclass
class Installed:
    lever: str
    applied: bool
    reason: Optional[str] = None                  # why not applied (named; feeds the partial census)
    lines: List[Callable[[], str]] = field(default_factory=list)   # zero-arg callables rendering this lever's LEVER line(s) at exit
    gates: List[Callable[[], Any]] = field(default_factory=list)    # zero-arg callables returning opt_core.gates.Gate at exit
    facts: Dict[str, Any] = field(default_factory=dict)


def rebind(owner, attr: str, wrapper, stock) -> None:
    """Replace ``owner.attr`` by ``wrapper`` and remember the stock callable on it."""
    try:
        wrapper.__wrapped_stock__ = stock
    except Exception:  # noqa: BLE001
        pass
    setattr(owner, attr, wrapper)


def installers():
    """lever name -> install(mode, tag, ctx) for every lever this kit applies (modes.MODES minus lever_report). One table; the kit's
    unit tests assert it covers modes.MODES."""
    from . import trimul as H_trimul, triatt as H_triatt, triatt_block as H_tab, triattn_core as H_tcore, triattn_exact as H_tex, triatt_block_exact as H_tabx, pair_block_residual as H_pbr, lm as H_lm, diffusion as H_diff, atom_sdpa as H_asd, atom_tf32 as H_atf, denoiser_graph as H_dg, graph_reuse as H_gr, ln_bf16 as H_ln, pae as H_pae, distogram as H_dist, relpos as H_relpos, pae_multimer as H_paem, transition_chunk as H_trc, pair_transition_fused as H_ptf, alloc_expandable as H_alloc, transition_exact as H_trx
    from . import sampler_hoist as H_sh
    from . import sampler_hostsync as H_shs
    from . import output_overlap as H_oo
    from . import triatt_block_exact as H_tria
    from . import transition_exact as H_tran
    from . import dit_apb as H_dit_
    from . import exactln as H_xln
    from . import atom_kdedup as H_akd
    from . import atom_bf16 as H_abf
    from . import atom_rows as H_arw
    return {"alloc_expandable": H_alloc.install, "trimul_exact": H_trimul.install, "trimul_v4": H_trimul.install, "flash_triattn": H_triatt.install, "triatt_block": H_tab.install, "triattn_core": H_tcore.install, "triattn_exact": H_tex.install, "triatt_block_exact": H_tabx.install, "lm_sdpa": H_lm.install,
            "diffusion_bf16": H_diff.install_bf16, "atom_sdpa": H_asd.install, "atom_tf32": H_atf.install, "denoiser_graph": H_dg.install, "graph_reuse": H_gr.install, "ln_bf16": H_ln.install,
            "pae_stream": H_pae.install, "pae_stream_m": H_paem.install, "conf_transition_chunk": H_trc.install, "pair_transition_chunk": H_trc.install_pair, "transition_exact": H_trx.install, "pair_transition": H_ptf.install,"pair_block_residual": H_pbr.install, 
            "distogram_offload": H_dist.install, "relpos_lazy": H_relpos.install, "sampler_hostsync": H_shs.install, "sampler_hoist": H_sh.install, "output_overlap": H_oo.install,
            "exactln": H_xln.install, "atom_kdedup": H_akd.install, "dit_apb": H_dit_.install, "atom_bf16": H_abf.install, "atom_rows": H_arw.install, "transition_exact": H_tran.install, "triatt_block_exact": H_tria.install}


def graph_context(t) -> bool:
    """True while the current CUDA stream of `t`'s device is being captured or is not the default stream — i.e. inside denoiser_graph's warm-up /
    capture (both run on its side stream; the replays carry no host cost).  Provider-bound levers (dit_apb, exactln) pass it as the capture /
    timing-form key of the call class, so the row the warm-up compiles is the row the capture records."""
    import torch
    if not t.is_cuda:
        return False
    if torch.cuda.is_current_stream_capturing():
        return True
    dev = t.device
    return torch.cuda.current_stream(dev).cuda_stream != torch.cuda.default_stream(dev).cuda_stream


def size_gated(ledger):
    """Gate for a lever with a token-size floor: kernel errors and unexpected fallbacks refuse as always (the core Ledger rules), but a run in
    which EVERY call fell back for reasons the lever lists as expected (e.g. all inputs below min_tokens) is a legitimate stock run of that
    statement, reported ok — not `served 0`."""
    def gate():
        return ledger.gate(require_served=False)
    return gate
