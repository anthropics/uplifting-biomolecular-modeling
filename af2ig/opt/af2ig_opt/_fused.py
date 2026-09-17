"""Shared plumbing of the kit's two fused pair-block adapters (:mod:`af2ig_opt.triattn` L10, :mod:`af2ig_opt.fused_trimul` L11): the serve-layer
resolution, the haiku parameter reader that keeps the stock parameter names, the rebind through the tree's haiku recipe, and the Ledger →
census record / LEVER line formatting. Nothing here imports jax, haiku or opt_core at module import (the CLI imports the adapters on CPU)."""
from typing import Callable, Optional

TAG = "af2ig-opt"
_READER = {}


def serve(kind: str):
    """``opt_core.kernels.fpf_pallas_serve`` with a float32 ``kind`` block (``triattn`` | ``trimul``), or a named ``core_missing`` error."""
    try:
        from opt_core.kernels import fpf_pallas_serve as S
    except ImportError as e:
        raise RuntimeError(f"af2ig_opt: core_missing:opt_core.kernels.fpf_pallas_serve ({e})") from None
    block = {"triattn": "tri_attn_block", "trimul": "trimul_block"}[kind]
    if not hasattr(S, block) or "float32" not in tuple(dict(getattr(S, "SERVED_DTYPES", {})).get(kind, ())):
        raise RuntimeError(f"af2ig_opt: core_missing:fpf_pallas_serve float32 {block} (opt_core >= {'0.5.3' if kind == 'triattn' else '0.5.4'})")
    return S


F32_DETAIL = "no-f32-tiles"                                          # the serve layer's detail prefix when a part HAS a tile table but no float32 rows in it


def refusal_word(S, refusal, kind: str) -> Optional[str]:
    """The word that names why the block cannot run in this process, from the serve layer's refusal at setup:
    ``cannot_run:no_tiles_cc<NN>(<kind>)`` (no tile rows for this GPU's compute capability in the shared core) /
    ``cannot_run:no_f32_tiles_cc<NN>(<kind>)`` (a table without float32 rows: this kit runs the float32 blocks) /
    ``cannot_run:<refusal kind>(<kind>)`` for every other named refusal (jax below the Pallas floor, Pallas absent or its API removed, the
    kernel module not importable, a backend that is not gpu). None for an exception that names no kind (the caller re-raises it as is)."""
    k = getattr(refusal, "kind", None)
    if not k:
        return None
    if k != S.NO_TILES:
        return f"cannot_run:{k}({kind})"
    try:
        cc = (S.compute_capability() or "unknown").replace(".", "")
    except Exception:  # noqa: BLE001 — a serve layer that cannot probe the card has no cc to name
        cc = "unknown"
    which = "no_f32_tiles" if str(getattr(refusal, "detail", "")).startswith(F32_DETAIL) else "no_tiles"
    return f"cannot_run:{which}_cc{cc}({kind})"


def f32_rows(S, kind: str) -> int:
    """The float32 route's tile multiple for ``kind`` on this part (``required_multiple(kind, dtype=float32)``): raises the serve layer's
    ``no_tiles`` with detail ``no-f32-tiles:<cc>`` at SETUP when the part's table carries no float32 rows — the refusal a first trace would meet."""
    return int(S.required_multiple(kind, None, None, dtype=getattr(S, "F32_DTYPE", "float32")))


def refuse(lever: str, flag: str, word: str, refusal, line_text: str):
    """A lever of the mode cannot run in this process: the MODE refuses by name — a mode is all of its levers, never a run under the mode's
    name with a subset. Prints the lever's evidence line (``state=skipped reason=<word>``) and the kit's NOT ACTIVE line naming the lever,
    the reason and the way out, then ends the driver with the kit's not-active status (3) before any design is folded. (The package's own
    tiles gate makes these levers step aside on a card without rows before the driver starts; this is the refusal from inside the process for
    what only the driver can see, e.g. a jax without Pallas.) An "untested" environment — a card served by its generation's rows, a jax the kernel was not tested on — is NOT this: it engages, named."""
    from opt_core import report
    from . import TAG
    report.emit(line_text)
    report.emit(f"[{TAG}] NOT ACTIVE: lever {lever} ({flag}) cannot run in this process: {word} — {getattr(refusal, 'detail', '') or refusal}; "
                f"a mode is all of its levers: select a mode without {lever} (--mode exact)")
    raise SystemExit(report.EXIT_NOT_ACTIVE)


def param_reader():
    """A haiku module that only reads parameters by name: instantiated inside the rebound ``__call__`` with a stock submodule's name
    (``query_norm``, ``attention``, ``left_projection`` …), its parameters carry exactly the stock full names, so the checkpoint loads unchanged."""
    if "cls" not in _READER:
        import haiku as hk

        class ParamReader(hk.Module):
            def __call__(self, specs, dtype):
                return {n: hk.get_parameter(n, shape, dtype, init=init) for n, (shape, init) in specs.items()}
        _READER["cls"] = ParamReader
    return _READER["cls"]


def rebind(state: dict, cls, fn: Callable, lever: str):
    """Rebind ``cls.__call__`` to ``fn`` WITH haiku's method wrapper (the module keeps its name scope: ``opt_core.mem.rowpair_jax.haiku.rebind``);
    ``state['orig']`` = the unwrapped stock body for refused calls; idempotent."""
    from opt_core.mem.rowpair_jax import haiku as rp_hk
    if state.get("orig") is None:
        state["patches"] = rp_hk.PatchSet(lever)
        state["orig"] = rp_hk.rebind(state["patches"], cls, "__call__", fn, lever=lever)


def unbind(state: dict) -> None:
    if state.get("orig") is not None:
        state["patches"].restore()
        state["orig"] = None; state["patches"] = None


def ledger(state: dict, name: str, impl: str, expected=()):
    if state.get("ledger") is None:
        from opt_core.counters import Ledger
        state["ledger"] = Ledger(name, impl=impl, origin="core", expected=tuple(expected))
    return state["ledger"]


def census(led, probe: Optional[dict], **facts) -> dict:
    """The Ledger's fields for a timer record + the adapter's facts (precision …) + the probe's cc / tiles / jax."""
    out = dict(led.fields())
    out.pop("facts", None); out.pop("min_tokens", None)
    p = probe or {}
    out.update(facts); out.update(cc=p.get("cc"), tiles=p.get("tiles"), jax=p.get("jax"))
    return out


def emit(text: str) -> str:
    from opt_core import report
    return report.emit(text)


def provider():
    """``opt_core.kernels.pallas`` (P) and its serve faces (PS) — the ONE binding of every kernel family : tier word in, fastest measured row per cell out."""
    try:
        from opt_core.kernels import pallas as P            # noqa: WPS433
        from opt_core.kernels.pallas import serve as PS     # noqa: WPS433
    except ImportError as e:
        raise RuntimeError(f"af2ig_opt: core_missing:opt_core.kernels.pallas.serve ({e}); install the tree's core at the kit's pin") from None
    return P, PS


def served_arms(face: str) -> dict:
    """{arm: calls} the provider served through ``face`` in this process (serve.report()['counts'] keys 'served:<face>:<arm>')."""
    try:
        _, PS = provider()
        counts = dict(PS.report().get("counts") or {})
    except Exception:  # noqa: BLE001
        return {}
    pre = "served:%s:" % face
    return {k[len(pre):]: int(v) for k, v in counts.items() if k.startswith(pre)}


def refusals(face: str) -> dict:
    """{word: n} the provider refused by name through ``face`` (counts keys 'refused:<face>:…' / last_refusal) — for the census."""
    try:
        _, PS = provider()
        rep = PS.report(); counts = dict(rep.get("counts") or {})
    except Exception:  # noqa: BLE001
        return {}
    out = {k.split(":", 2)[-1]: int(v) for k, v in counts.items() if k.startswith("refused:%s:" % face) or (k.startswith("refused:") and face in k)}
    return out


def arms_word(d: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(d.items())) or "none"
