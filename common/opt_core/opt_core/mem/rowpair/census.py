"""census.py — the per-stage CUDA memory census of a tensor-parallel (row-sharded pair) run, and the reader that turns its lines into the
scaling table (per-rank peak = a(N) + b(N)/P).

THE HOOK (in a kit process, any engine). Generic: imports no engine; the adapters place the stage marks. Off unless the environment says
``OPT_CORE_TP_CENSUS=1`` — an unset variable makes every call return ``None`` before touching torch, so a ``--n_gpu 1`` run's outputs and
transcript are unchanged by the presence of the calls.

    from opt_core.mem.rowpair import census        # adapters place the marks
    census.mark("trunk_entry")                      # -> one JSONL line on stderr (and in $OPT_CORE_TP_CENSUS_FILE.rank<r>.jsonl when set):
    TPCENSUS {"rank": 0, "world": 8, "stage": "trunk_entry", "alloc_gb": 3.12, "peak_gb": 3.4, "peak_delta_gb": 3.4, "reserved_gb": 3.6,
              "device_used_gb": 5.9, "device_total_gb": 79.6, "top": [[[1, 370, 2956, 128], "float32", 0.56], ...], "t": 12.3, ...}

Stage vocabulary (STAGES — the marks the adapters place; another name is accepted and flagged ``"known": false``, never refused: a census
counts, it does not gate): trunk_entry · after_template · after_msa · after_pairstack · after_gather (or no_gather when z stays sharded) ·
distogram · confidence · diffusion_start · diffusion_peak · done.  Replicated-by-design tensors (s, m, atom tensors) show up in ``top``
beside the z shard — that is the point of the storage census: a rank holding an [N, N, c] tensor whole is visible by shape.

Fields per line: ``alloc_gb`` = torch.cuda.memory_allocated; ``peak_gb`` = torch.cuda.max_memory_allocated since process start (or since
the engine's own last reset — this module never resets the allocator's peak unless ``OPT_CORE_TP_CENSUS_RESET=1``, because the kit reads the
whole-run peak for its EXIT lines); ``peak_delta_gb`` = peak_gb minus the previous mark's peak_gb on this rank (the stage's contribution
to the running peak); ``reserved_gb`` = the caching allocator's reservation; ``device_used_gb`` / ``device_total_gb`` = the DRIVER's view
(torch.cuda.mem_get_info: includes the CUDA context, NCCL / cuBLAS workspaces and every other process on the card — what an OOM is decided
against); ``top`` = the largest live CUDA storages found by a gc walk (>= TOP_MIN_BYTES, deduplicated by storage pointer, at most TOP_K),
each ``[shape, dtype, storage_gb]``; ``t`` = seconds since the first mark on this rank; ``rank_source`` names where the rank came from.
GB = 1e9 bytes throughout (the memory board's unit); torch's GiB numbers are x1.074.

THE READER (anywhere, stdlib only): ``parse(lines)`` collects TPCENSUS records from a transcript; ``peaks_by_rank(records)`` the per-rank
maximum of peak_gb and device_used_gb; ``fit_peak_vs_P(points)`` the least-squares fit per-rank peak = a + b/P over the P points of one
(engine, N); ``scaling_verdict(...)`` the scaling-fit acceptance rule.

    python -m opt_core.mem.rowpair.census parse <transcript or .jsonl> [...]      -> one summary JSON (per rank: peak_gb, device_used_gb, last stage; the stage table)
    python -m opt_core.mem.rowpair.census fit '{"1": 54.7, "2": 85.0, "4": 75.1, "8": 66.7}'   -> {"a": ..., "b": ..., "residuals": {...}, "r2": ...}
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = ["ENV_ON", "ENV_FILE", "ENV_RESET", "STAGES", "TOP_MIN_BYTES", "TOP_K", "enabled", "mark", "reset", "parse", "peaks_by_rank",
           "fit_peak_vs_P", "scaling_verdict", "MARKER", "A_GROWTH_MAX", "B_RATIO_TARGET", "B_RATIO_TOL", "REACH_RATIO_MIN"]

ENV_ON = "OPT_CORE_TP_CENSUS"          # "1" turns the hook on; anything else (or unset) = every call is a no-op returning None
ENV_FILE = "OPT_CORE_TP_CENSUS_FILE"   # when set: lines are ALSO appended to <value>.rank<r>.jsonl (the launcher points it under the run's output dir)
ENV_RESET = "OPT_CORE_TP_CENSUS_RESET"  # "1": reset the allocator's peak after each mark (per-stage peaks; changes the engine's own whole-run peak report)
MARKER = "TPCENSUS"
STAGES = ("trunk_entry", "after_template", "after_msa", "after_pairstack", "after_gather", "no_gather", "distogram", "confidence",
          "diffusion_start", "diffusion_peak", "done")
TOP_MIN_BYTES = 64 * 1000 * 1000       # live CUDA storages at least this large enter `top`
TOP_K = 8
GB = 1e9
RANK_ENVS = ("ROWPAIR_RANK", "RANK")                     # read only when opt_core's launcher is not importable (rank_source names which)
WORLD_ENVS = ("ROWPAIR_WORLD", "WORLD_SIZE")

_state: Dict[str, Any] = {"t0": None, "last_peak": {}, "n": 0}


def enabled() -> bool:
    return os.environ.get(ENV_ON, "") == "1"


def _rank_world() -> tuple:
    """(rank, world, source): opt_core.mem.rowpair.launch's own equality readers when importable (the core owns rank equality), else the
    namespaced environment variables, else (0, 1)."""
    try:
        from opt_core.mem.rowpair import launch as _launch   # the core's reader (ROWPAIR_RANK / torchrun names as it defines them)
        return int(_launch.rank()), int(_launch.world_size()), "opt_core.mem.rowpair.launch"
    except Exception:
        pass
    for rn, wn in zip(RANK_ENVS, WORLD_ENVS):
        if os.environ.get(rn) is not None:
            try:
                return int(os.environ[rn]), int(os.environ.get(wn) or 1), "env:" + rn
            except ValueError:
                continue
    return 0, 1, "default"


def _top_storages(torch, min_bytes: int = TOP_MIN_BYTES, k: int = TOP_K) -> List[list]:
    """The k largest live CUDA storages reachable by gc (deduplicated by storage data pointer): [[shape, dtype, storage_gb], ...].
    A tensor that is a view reports its own shape with the STORAGE's size (what the allocator holds)."""
    seen: Dict[int, list] = {}
    try:
        objs = gc.get_objects()
    except Exception:
        return []
    for o in objs:
        try:
            if not isinstance(o, torch.Tensor) or not o.is_cuda:
                continue
            st = o.untyped_storage()
            nb = int(st.nbytes())
            if nb < min_bytes:
                continue
            ptr = int(st.data_ptr())
            cur = seen.get(ptr)
            if cur is None or o.numel() > cur[3]:
                seen[ptr] = [list(o.shape), str(o.dtype).replace("torch.", ""), round(nb / GB, 3), int(o.numel())]
        except Exception:
            continue
    top = sorted(seen.values(), key=lambda r: -r[2])[:k]
    return [r[:3] for r in top]


def mark(stage: str, **extra: Any) -> Optional[dict]:
    """Emit one census line for `stage` on this rank; returns the record (None when the hook is off). With the hook on and CUDA
    absent one line with "cuda": false is still printed, so the transcript says the mark was reached."""
    if not enabled():
        return None
    rank, world, src = _rank_world()
    now = time.time()
    if _state["t0"] is None:
        _state["t0"] = now
    rec: Dict[str, Any] = {"rank": rank, "world": world, "stage": str(stage), "known": str(stage) in STAGES, "n": _state["n"],
                           "t": round(now - _state["t0"], 2), "rank_source": src}
    _state["n"] += 1
    try:
        import torch
    except Exception as e:   # a kit without torch cannot have CUDA tensors; the line names it
        rec.update({"cuda": False, "why": f"torch not importable ({type(e).__name__})"})
        _emit(rec, rank)
        return rec
    if not torch.cuda.is_available():
        rec.update({"cuda": False, "why": "torch.cuda.is_available() is False"})
        _emit(rec, rank)
        return rec
    try:
        dev = torch.cuda.current_device()
        torch.cuda.synchronize(dev)
        alloc = torch.cuda.memory_allocated(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        free_b, total_b = torch.cuda.mem_get_info(dev)
        last = _state["last_peak"].get(rank, 0)
        rec.update({"cuda": True, "device": int(dev), "alloc_gb": round(alloc / GB, 3), "peak_gb": round(peak / GB, 3),
                    "peak_delta_gb": round((peak - last) / GB, 3), "reserved_gb": round(reserved / GB, 3),
                    "device_used_gb": round((total_b - free_b) / GB, 3), "device_total_gb": round(total_b / GB, 3),
                    "top": _top_storages(torch)})
        _state["last_peak"][rank] = peak
        if os.environ.get(ENV_RESET, "") == "1":
            torch.cuda.reset_peak_memory_stats(dev)
            rec["peak_reset"] = True
            _state["last_peak"][rank] = torch.cuda.memory_allocated(dev)
    except Exception as e:   # the census never takes the run down: the failure is the line
        rec.update({"cuda": None, "why": f"{type(e).__name__}: {str(e)[:160]}"})
    if extra:
        rec["extra"] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in extra.items()}
    _emit(rec, rank)
    return rec


def reset() -> None:
    """Forget this process's mark history (tests)."""
    _state.update({"t0": None, "last_peak": {}, "n": 0})


def _emit(rec: Mapping[str, Any], rank: int) -> None:
    line = MARKER + " " + json.dumps(rec, sort_keys=True, default=str)
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass
    base = os.environ.get(ENV_FILE)
    if base:
        try:
            d = os.path.dirname(base)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(f"{base}.rank{rank}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as e:
            print(f"{MARKER}_FILE_ERROR {type(e).__name__}: {str(e)[:160]}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------------------------- the reader
def parse(lines: Iterable[str]) -> List[dict]:
    """Every TPCENSUS record in `lines` (a transcript, a rank log, a .jsonl), in order; a line whose JSON does not parse is kept as
    {"unparsed": <text>} (named, never dropped). Rank-log relays that prefix lines ('[r0] ', an epoch stamp) are tolerated."""
    out = []
    for ln in lines:
        i = ln.find(MARKER + " ")
        if i < 0 or ln[i + len(MARKER) + 1: i + len(MARKER) + 2] != "{":
            continue
        txt = ln[i + len(MARKER) + 1:].strip()
        try:
            out.append(json.loads(txt))
        except ValueError:
            out.append({"unparsed": txt[:400]})
    return out


def peaks_by_rank(records: Sequence[Mapping[str, Any]]) -> Dict[str, dict]:
    """{rank: {peak_gb, device_used_gb, reserved_gb (maxima over the rank's lines), last_stage, n_lines, stages: {stage: peak_gb}}}."""
    out: Dict[str, dict] = {}
    for r in records:
        if "rank" not in r:
            continue
        k = str(r["rank"])
        e = out.setdefault(k, {"peak_gb": None, "device_used_gb": None, "reserved_gb": None, "last_stage": None, "n_lines": 0, "stages": {}})
        e["n_lines"] += 1
        e["last_stage"] = r.get("stage")
        for f in ("peak_gb", "device_used_gb", "reserved_gb"):
            v = r.get(f)
            if isinstance(v, (int, float)) and (e[f] is None or v > e[f]):
                e[f] = v
        if isinstance(r.get("peak_gb"), (int, float)):
            e["stages"][str(r.get("stage"))] = r["peak_gb"]
    return out


def fit_peak_vs_P(points: Mapping[Any, float]) -> dict:
    """Least squares per-rank peak = a + b / P over {P: peak_gb} (>= 2 distinct P; refuses fewer by name). Returns
    {a, b, n, residuals: {P: observed - fitted}, r2, points}. Closed form (x = 1/P): b = cov(x, y) / var(x), a = mean(y) - b mean(x)."""
    pts = {int(p): float(v) for p, v in points.items() if v is not None}
    if len(pts) < 2:
        raise ValueError(f"fit_peak_vs_P: needs peaks at >= 2 distinct P, got {sorted(pts)} — refused by name")
    xs = [1.0 / p for p in sorted(pts)]
    ys = [pts[p] for p in sorted(pts)]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    if vx == 0:
        raise ValueError("fit_peak_vs_P: all P equal — refused by name")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx
    a = my - b * mx
    fitted = {p: a + b / p for p in sorted(pts)}
    res = {p: round(pts[p] - fitted[p], 3) for p in sorted(pts)}
    ss_res = sum(v ** 2 for v in res.values())
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"a": round(a, 3), "b": round(b, 3), "n": n, "residuals": res, "r2": round(r2, 4), "points": {p: pts[p] for p in sorted(pts)}}


# acceptance thresholds (README.md §Scaling-fit acceptance): the ONE home of the numbers
A_GROWTH_MAX = 1.5          # a(2N) / a(N) <= this (the replicated floor grows slowly, << N^2)
B_RATIO_TARGET = 4.0        # b(2N) / b(N) ~ 4 (b is the N^2 term)
B_RATIO_TOL = 0.15          # +-15 %
REACH_RATIO_MIN = 2.5       # N_max(P=8) / N_max(P=1) >= this (the sqrt(8) = 2.83 ideal, EXT bins)


def scaling_verdict(fit_n: Mapping[str, Any], fit_2n: Mapping[str, Any], n_tokens: int, reach_p1: Optional[int] = None,
                    reach_p8: Optional[int] = None, reference_b_per_n2: Optional[float] = None) -> dict:
    """The scaling-fit acceptance word for one engine from its two fits (at N and 2N tokens): PASS iff a(2N)/a(N) <= A_GROWTH_MAX and
    b(2N)/b(N) within B_RATIO_TARGET x (1 +- B_RATIO_TOL) and (when both reaches are given) reach_p8 / reach_p1 >= REACH_RATIO_MIN.
    b / N^2 and its ratio to a reference implementation's b / N^2 (``reference_b_per_n2``, bytes per token^2 at the same N convention) are reported beside,
    never deciding. The reach criterion with no EXT run yet is named unjudged: every other criterion passing then gives
    'PROVISIONAL (reach unjudged)', never PASS."""
    a1, a2, b1, b2 = float(fit_n["a"]), float(fit_2n["a"]), float(fit_n["b"]), float(fit_2n["b"])
    checks: Dict[str, Any] = {}
    checks["a_growth"] = {"value": round(a2 / a1, 3) if a1 > 0 else None, "max": A_GROWTH_MAX,
                          "pass": (a1 > 0 and a2 / a1 <= A_GROWTH_MAX) or (a1 <= 0 and a2 <= 0)}
    br = b2 / b1 if b1 != 0 else None
    checks["b_ratio"] = {"value": round(br, 3) if br is not None else None, "target": B_RATIO_TARGET, "tol": B_RATIO_TOL,
                         "pass": br is not None and abs(br - B_RATIO_TARGET) <= B_RATIO_TOL * B_RATIO_TARGET}
    if reach_p1 and reach_p8:
        rr = reach_p8 / float(reach_p1)
        checks["reach"] = {"value": round(rr, 3), "min": REACH_RATIO_MIN, "pass": rr >= REACH_RATIO_MIN}
    else:
        checks["reach"] = {"value": None, "min": REACH_RATIO_MIN, "pass": None, "why": "unjudged: N_max(P=1) and N_max(P=8) not both measured"}
    b_per_n2 = {"N": round(b1 / float(n_tokens) ** 2 * 1e9, 4), "2N": round(b2 / float(2 * n_tokens) ** 2 * 1e9, 4), "unit": "bytes per token^2 (b in GB, x P ranks)"}
    if reference_b_per_n2:
        b_per_n2["ratio_to_reference"] = round((b1 / float(n_tokens) ** 2 * 1e9) / reference_b_per_n2, 3)
    decided = [c["pass"] for c in checks.values() if c["pass"] is not None]
    if decided and all(decided) and checks["reach"]["pass"] is None:
        word = "PROVISIONAL (reach unjudged)"
    elif decided and all(decided):
        word = "PASS"
    else:
        word = "FAIL (" + ", ".join(k for k, c in checks.items() if c["pass"] is False) + ")"
    return {"word": word, "checks": checks, "b_per_n2": b_per_n2, "fits": {"N": dict(fit_n), "2N": dict(fit_2n)}, "n_tokens": int(n_tokens)}


def _main(argv: Sequence[str]) -> int:
    if len(argv) >= 2 and argv[0] == "parse":
        recs: List[dict] = []
        for p in argv[1:]:
            with open(p, encoding="utf-8", errors="replace") as fh:
                recs += parse(fh)
        print(json.dumps({"n_records": len(recs), "by_rank": peaks_by_rank(recs),
                          "unparsed": sum(1 for r in recs if "unparsed" in r)}, indent=1, sort_keys=True))
        return 0
    if len(argv) == 2 and argv[0] == "fit":
        print(json.dumps(fit_peak_vs_P(json.loads(argv[1])), indent=1, sort_keys=True))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
