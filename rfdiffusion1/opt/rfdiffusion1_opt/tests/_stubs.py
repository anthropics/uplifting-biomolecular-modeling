import json
"""A simulated box for the CPU tests: the GPU probe, the tools, the stock pins, the kit byte check, the checkout and the weights
directory, monkeypatched on `rfdiffusion1_opt.stack` so the gates can be exercised without a GPU, torch or the upstream checkout."""
import os
import sys

H100 = {"name": "NVIDIA H100 80GB HBM3", "mem_gib": 79.6, "cc": "9.0", "sm": "sm90", "source": "stub"}
H100NVL = {"name": "NVIDIA H100 NVL", "mem_gib": 93.6, "cc": "9.0", "sm": "sm90", "source": "stub"}
H200 = {"name": "NVIDIA H200", "mem_gib": 140.5, "cc": "9.0", "sm": "sm90", "source": "stub"}
A100 = {"name": "NVIDIA A100-SXM4-80GB", "mem_gib": 79.6, "cc": "8.0", "sm": "sm80", "source": "stub"}
L40S = {"name": "NVIDIA L40S", "mem_gib": 45.0, "cc": "8.9", "sm": "sm89", "source": "stub"}
B200 = {"name": "NVIDIA B200", "mem_gib": 179.0, "cc": "10.0", "sm": "sm100", "source": "stub"}
STACK_OF_RECORD = {"torch": "2.4.0", "dgl": "2.4.0+cu124", "triton": "3.0.0"}
TOOLS = {"python": sys.executable, "mps_control": None}


def good_box(monkeypatch, gpu=None, tools=None, pins_bad=None, rfd="/opt/rfd", weights="/models/rfdiffusion", versions=None):
    """Monkeypatch stack so that a mode resolves on an H100 box with the pinned stack, the pinned checkout and the weights."""
    from rfdiffusion1_opt import stack
    g = dict(H100 if gpu is None else gpu)
    vers = dict(STACK_OF_RECORD, **(versions or {}))
    monkeypatch.setattr(stack, "gpu_probe", lambda: dict(g))
    monkeypatch.setattr(stack, "tools", lambda: dict(TOOLS if tools is None else tools))
    monkeypatch.setattr(stack, "dist_version", lambda name: vers.get(name))
    monkeypatch.setattr(stack, "upstream_version", lambda: "1.1.0")
    detail = {"checkout": {"root": rfd, "pinned": not pins_bad, "git_head": "86507b6538f51fce57b5a72477165f03999ed7ae", "files_checked": 75, "files_differ": [], "files_missing": []},
              "stack": {"torch": {"installed": vers.get("torch"), "pinned": "2.4.0", "ok": True}, "dgl": {"installed": vers.get("dgl"), "pinned": "2.4.0+cu124", "ok": True}}}
    monkeypatch.setattr(stack, "check_pins", lambda stack=True, root=None: (list(pins_bad or []), detail))
    monkeypatch.setattr(stack, "rfd_root", lambda: rfd)
    monkeypatch.setenv("WEIGHTS", weights)
    monkeypatch.delenv("RFDIFFUSION1_OPT", raising=False)
    monkeypatch.delenv("RFDIFFUSION1_OPT_FORCE", raising=False)
    monkeypatch.delenv("MODEL_OPT_TARGET_GPU", raising=False)
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    return g


def write_case_outputs(out_dir: str, case: str, indices, traj: bool = False, seed: bytes = b"", prefix: str = None) -> str:
    """Synthetic <prefix>_<i>.pdb / .trb (+ traj) files of one case, deterministic in `seed` so two trees can be equal or differ by design.
    `prefix`: the case's own output prefix (the hydra form: upstream's inference.output_prefix); default `<out_dir>/<case>/des` (the cases form).
    Returns the directory written."""
    import pickle
    prefix = os.path.abspath(prefix) if prefix else os.path.join(out_dir, case, "des")
    d, base = os.path.dirname(prefix), os.path.basename(prefix)
    os.makedirs(os.path.join(d, "traj") if traj else d, exist_ok=True)
    for i in indices:
        open(f"{prefix}_{i}.pdb", "wb").write(b"ATOM " + seed + str(i).encode() + b"\n")
        trb = {"config": {"inference": {"output_prefix": prefix, "design_startnum": indices[0], "num_designs": len(indices)}, "diffuser": {"T": 50}},
               "plddt": [0.5 + 0.001 * i], "time": 1.0 + i, "device": "cuda:0", "con_hal_idx0": [1, 2, 3]}
        pickle.dump(trb, open(f"{prefix}_{i}.trb", "wb"))
        if traj:
            for kind in ("Xt-1", "pX0"):
                open(os.path.join(d, "traj", f"{base}_{i}_{kind}_traj.pdb"), "wb").write(b"MODEL " + seed + kind.encode() + b"\n")
    return d


EXACT_EVIDENCE = "print('fastpath levers:', ['chain_breaks']); print('prep lever: active = True'); print('einsum lever E: {}'); print('fullgraph mode: applied = True'); print('pdb writer lever IO1: armed (rfdiffusion.util.writepdb, writepdb_multi)'); print('[rfdiffusion1-opt.driver_run] PDBIO_FINAL ' + json.dumps(dict(armed=True, n_calls=3, n_verified=2, n_mismatch=0, verified_keys=['writepdb(...)', 'writepdb_multi(...)'], mismatches=[], seconds=0.1)))"   # the exact row's applied-lines as a fake driver prints them (registry evidence of C1, P, E_einsum, W1)
FULLGRAPH_OK = {"n_replay": 3, "n_capture": 2, "capture_errors": [], "eager_fallback_calls": 0}            # W1's final counters on a clean pass (design.COUNTER_RULES)
PRECISION_FP32 = {"param_dtype": "torch.float32", "allow_tf32_matmul": False, "cudnn_tf32": False, "autocast_enabled": False}   # the driver's per-case torch read-back under --tf32 0 (rfd_bench.py:139-141)
PRECISION_TF32 = dict(PRECISION_FP32, allow_tf32_matmul=True, cudnn_tf32=True)                                              # ... under --tf32 1 (lever TF32)
WRITE_TIMINGS = ("def write_timings(out, tag, cases, precision, **final):\n"
                 "    json.dump(dict(final, cases=[{'case': c['name'], 'precision': precision} for c in cases]), open(os.path.join(out, tag + '_timings.json'), 'w'))\n")   # source a fake driver embeds: the timings file with the per-case precision record and the given final counters


def fake_driver_source(evidence: str = EXACT_EVIDENCE, final: str = "fullgraph_final=FULLGRAPH_OK", precision: str = "PRECISION_FP32", extra: str = "") -> str:
    """The source of a fake resident driver: prints the evidence lines, writes the cases' outputs (traj when `--no-traj 0`), then the timings
    file (write_timings with `final` counters and `precision`), then `extra` statements."""
    return ("import sys, os, json\n"
            "a = sys.argv[1:]; out = a[a.index('--out')+1]; tag = a[a.index('--tag')+1]; cases = json.load(open(a[a.index('--cases')+1]))\n"
            + evidence + "\n"
            "sys.path.insert(0, os.environ['STUBS']); from _stubs import write_case_outputs, FULLGRAPH_OK, PRECISION_FP32, PRECISION_TF32\n"
            + WRITE_TIMINGS +
            "for c in cases: write_case_outputs(out, c['name'], list(range(c.get('startnum', 0), c.get('startnum', 0) + int(os.environ.get('N_OUT', c['num_designs'])))), traj=('--no-traj' in a and a[a.index('--no-traj')+1] == '0'), prefix=c.get('prefix'))\n"   # rfd_bench.py:113: the case's own prefix when it names one
            f"write_timings(out, tag, cases, {precision}, {final})\n" + extra)


def core_home() -> str:
    """The core's project directory (common/opt_core: the parent of the imported opt_core package) — a child interpreter of these tests puts it
    on PYTHONPATH beside the package's opt/ (an install carries both)."""
    import opt_core
    return os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))


def pkg_pythonpath(*extra: str) -> str:
    """PYTHONPATH for a child interpreter: rfdiffusion1/opt, the core's directory, then `extra`."""
    from rfdiffusion1_opt import stack
    return os.pathsep.join([os.path.join(stack.tree_root(), "opt"), core_home(), *[str(e) for e in extra]])
