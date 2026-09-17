"""What the modes print and export on an H100 stack (cc 9.0, torch 2.13.0, triton 3.7.1), rendered without a GPU: per mode the switches
``modes.resolve`` hands env.sh and adds after it (the lever census env.sh acts on), the mode table, the multi-GPU line's constants, rank
environment, launcher command and ACTIVE line for ``--n_gpu 2``, and the ACTIVE / DRY-RUN / LEVER line formats over a fixed activation
record. ``render()`` is the one producer; ``tests/data/surface_lines.json`` holds its output and test_surface_lines.py compares the two.
Regenerate with ``python -m protenix_opt.tests._surface_render > opt/protenix_opt/tests/data/surface_lines.json`` from the kit directory."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

from protenix_opt import big, modes, registry, report, stack, tp
from protenix_opt.tests.test_modes_match_env_sh import _sandbox_bin

CC, TORCH, TRITON = "9.0", "2.13.0", "3.7.1"
PREFIX = report.PREFIX


def _norm(v, subs):
    """Strings with the kit root -> <KIT>, the interpreter -> <PY>, the sandbox bin -> <BIN> (the record is location-free)."""
    if isinstance(v, str):
        for old, new in subs:
            v = v.replace(old, new)
        return v
    if isinstance(v, dict):
        return {str(_norm(k, subs)): _norm(x, subs) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_norm(x, subs) for x in v]
    return v


def _big_record(bin_dir: str, tmp: str) -> dict:
    """The memory line applied on CPU (the protenix hook modules are absent here, so the hooked levers refuse by name): the markers
    stack._classify reads, the core record's composed line and per-lever lines, and the manifest block's line policy."""
    marks = big.activate(tp.BASE_MODE, environ={"PATH": bin_dir})
    rec = big._record()
    out = {"marks": marks, "install_error": big._STATE.get("install_error")}
    if rec is not None:
        block = rec.manifest_block()
        out.update({"core_active_line": rec.active_line(big.TAG), "core_lever_lines": rec.lever_lines(big.TAG),
                    "line_policy": block.get("line_policy"), "manifest_block_keys": sorted(block)})
    return out


def _fixed_rep(mode: str) -> dict:
    """A fixed activation record over the mode's own composition: every lever applied but the last, which fell back by name."""
    levers = list(modes.MODES[mode]) if mode in modes.MODES else []
    applied, fallback = levers[:-1], levers[-1:]
    return {"active": True, "mode": mode, "protenix_version": "2.0.0", "gpu": {"name": "NVIDIA H100 80GB HBM3", "cc": CC},
            "levers_applied": applied, "levers_fallback": fallback, "levers_not_in_arm": [],
            "fallback_reasons": {n: "rendered_reason" for n in fallback}, "n_gpu": 1, "n_gpu_source": "default"}


def render() -> dict:
    kit_root = os.path.dirname(stack.opt_dir()) if hasattr(stack, "opt_dir") else os.path.dirname(os.path.dirname(stack.kit_home()))
    fpf = stack.kit_home()
    tmp = tempfile.mkdtemp()
    keep = dict(os.environ)
    try:
        os.environ.pop("CUEQ_TRITON_CACHE_DIR", None)                  # the cueq_tuned_tiles LEVER pair reads the kit's own cueq_cache/, never a developer's cache dir
        bin_dir = _sandbox_bin(tmp, nvidia_smi_cc=CC, torch_version=TORCH)
        out: dict = {"stack": {"cc": CC, "torch": TORCH, "triton": TRITON}, "modes": {}, "resolve": {}, "lines": {}, "tp": {}}
        out["modes"] = {m: list(v) if isinstance(v, (list, tuple)) else v for m, v in modes.MODES.items()}
        out["registry"] = list(registry.LEVERS)
        for mode in ("exact", "fast", "big"):
            r = modes.resolve(mode, {"PATH": bin_dir}, fpf, compute_cap=CC, triton=TRITON, probe_gpu=False)
            out["resolve"][mode] = {"exports": dict(sorted(r.exports.items())), "unsets": sorted(r.unsets), "extras": dict(sorted(r.extras.items())),
                                    "pre_exports": dict(sorted(r.pre_exports.items())), "kernel_key": r.kernel_key, "row_key": r.row_key,
                                    "row": r.row, "kit_spec": r.kit_spec, "base": r.base, "pythonpath": list(r.pythonpath)}
            rep = _fixed_rep(mode)
            out["lines"][mode] = {"active": report.activation_line(rep), "dry_run": report.activation_line(dict(rep, dry_run=True)),
                                  "lever": report.lever_lines(rep)}
        os.environ.clear(); os.environ.update({"PATH": bin_dir, "HOME": tmp})
        base = _fixed_rep(tp.BASE_MODE)
        out["tp"] = {
            "constants": {"LINE_ARGS": list(tp.LINE_ARGS), "STOCK_ARG_DELTA": list(tp.STOCK_ARG_DELTA), "TP_PRE": dict(tp.TP_PRE), "TP_GATES": dict(tp.TP_GATES),
                          "DET_GUARDS": tp.DET_GUARDS, "NOT_GATES": list(tp.NOT_GATES), "BIND_ENV": dict(tp.BIND_ENV),
                          "TP_DROPPED": list(tp.TP_DROPPED), "CENSUS": {k: list(v) for k, v in tp.CENSUS.items()}, "BLOCKS": list(tp.BLOCKS),
                          "BLOCKS_SUPPORTED": list(tp.BLOCKS_SUPPORTED), "IMPL": tp.IMPL, "LINE": tp.LINE, "SCHEME": tp.SCHEME, "LAUNCHER": tp.LAUNCHER,
                          "LAUNCHER_MODULE": tp.LAUNCHER_MODULE, "PHASE_LOG_DIR": tp.PHASE_LOG_DIR, "STRATEGY": tp.STRATEGY},
            "tokens": {"census": tp.census_token(), "gates": tp.gates_token(), "p1_levers": tp.p1_levers_token(), "active_fields_2": tp.active_fields(2)},
            "command_2": tp.command(2, ["-i", "in.json", "-o", "OUT"], "OUT", "L"),
            "rank_env_2": {str(d): dict(sorted(tp.rank_env(2, "OUT", exports={"PTX_EXPORTED": "1"}, base={"PATH": bin_dir}, det_level=d).items())) for d in (0, 1)},
            "active_line_2": {str(d): tp.active_line(PREFIX, 2, base, det_level=d) for d in (0, 1)},
            "refusals": {f"{m}/{n}": tp.refusal(m, n) for m in ("exact", "fast", "big", "off") for n in (1, 2)},
        }
        out["big"] = _big_record(bin_dir, tmp)
        return _norm(out, ((kit_root, "<KIT>"), (sys.executable, "<PY>"), (bin_dir, "<BIN>"), (tmp, "<TMP>")))
    finally:
        os.environ.clear(); os.environ.update(keep)
        shutil.rmtree(tmp, ignore_errors=True)


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "surface_lines.json")

if __name__ == "__main__":
    json.dump(render(), sys.stdout, indent=1, sort_keys=True); sys.stdout.write("\n")
