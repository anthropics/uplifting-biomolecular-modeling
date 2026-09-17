"""The KERNELS line and its REQUIRE guard: establishes which accelerator implementation each pass actually ran, on every route, and refuses by name
when it isn't the one that route's mode table expects. The expectation is derived purely from the mode table (``modes.KIT_MODES``), the request
and the stack's fixed constants, and is compared, in-process, against what the model actually bound before the first timed item. ``account()``
folds the result into one ``KERNELS`` log line per pass; ``refusal()`` names the difference and ``cli`` exits 5 (``EXIT_KERNELS``) on any. No
escape hatch: a box lacking the implementation (no Triton, CPU-only) is refused on every route, no exception."""
from __future__ import annotations

import os
import re
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

from opt_core.mem import peak as _peak

from . import modes as _modes, settings as _settings, stock_pred
from .inprocess import kernels_probe as _probe
from .report import line as _line
from .stack import pins

EXIT_KERNELS = _probe.EXIT_REFUSED                                        # 5: the pass ran (or would run) a kernel implementation that is not the route's
ENV_EXPECT = _probe.ENV_EXPECT
PROBE_FILE = os.path.abspath(_probe.__file__)
PROBE_RX = re.compile(r"\[af3-jax-opt\] KERNELS-PROBE (?P<kv>.*)$")
CENSUS_RX = re.compile(r"\[af3-jax-opt\] KERNELS-CENSUS (?P<kv>.*)$")
REFUSED_RX = re.compile(r"\[af3-jax-opt\] KERNELS REFUSED in pid")
# upstream's own fallback prints, trapped from the transcript (file:line in the vendored tree / the tokamax wheel of the freeze):
TRAPS = {
    "cpu_auto_downgrade": re.compile(r"Setting --flash_attention_implementation=xla, required for"),      # stock/src/run_alphafold.py:1131-1134 (kit script :1411-1414)
    "glu_fallback": re.compile(r"Failed to run implementation"),                                           # tokamax/_src/ops/gated_linear_unit/api.py:116
    "autotune_failed": re.compile(r"Failed to autotune for op"),                                           # tokamax/_src/autotuning/api.py:318
}
KERNELS_RX = re.compile(r"\[af3-jax-opt\] KERNELS route=(?P<route>\S+) mode=(?P<mode>\S+) flash_impl=(?P<flash_impl>\S+) pair_attn=(?P<pair_attn>\S+) "
                        r"trimul=(?P<trimul>\S+) glu_impl=(?P<glu_impl>\S+) xla_flags=(?P<xla_flags>\S+) requested=(?P<requested>\S+) .*probe=(?P<probe>\S+) verdict=(?P<verdict>\S+)")
SITE_BY_LEVER = {"pair_attn": (("FPF_TRIATT", "fpf"), ("ATTNCFG", "attncfg")), "trimul": (("FPF_TRIMUL", "fpf"), ("GLUT", "glut"))}   # first lever present names the owner; none = the stock binding
STOCK_SITE = {"pair_attn": "tokamax", "trimul": "stock"}
FLASH_IMPLS = ("triton", "cudnn", "xla")                                  # upstream's --flash_attention_implementation enum (run_alphafold.py:391)
GLU_HEAD = "triton"                                                       # tokamax 0.0.12's GLU chain head on a Triton-capable GPU (gated_linear_unit/api.py:43-45): the implementation every route expects


def route_name(mode: str, n_gpu: int = 1) -> str:
    """The route word: ``stock`` (mode off), the kit mode's name, ``big_xP`` at ``--n_gpu P > 1``."""
    if mode == "off":
        return "stock"
    return f"{mode}_x{int(n_gpu)}" if int(n_gpu or 1) > 1 else mode


def site_owner(site: str, levers: List[str]) -> str:
    for lever, word in SITE_BY_LEVER[site]:
        if lever in levers:
            return word
    return STOCK_SITE[site]


def expected(mode: str, levers: List[str], requested_impl: str) -> Dict[str, str]:
    """The route's expected reading, from the mode table and the request only: {flag, dpa_site, triatt_site, trimul_site, glu_head, backend,
    attn_supported, attn_class}. ``mode`` is the EFFECTIVE mode (big in region fast = the fast line)."""
    levers = list(levers or [])
    if mode == "off" and levers:
        raise ValueError(f"mode off carries no levers (got {levers})")
    pair = site_owner("pair_attn", levers)
    return {"flag": requested_impl, "dpa_site": "attncfg" if pair == "attncfg" else "tokamax", "triatt_site": "fpf" if pair == "fpf" else "stock",
            "trimul_site": site_owner("trimul", levers), "glu_head": GLU_HEAD, "backend": "gpu", "attn_supported": "1", "attn_class": "present"}


def lever_impls(levers: List[str], lines: Sequence[str] = ()) -> Dict[str, Tuple[str, ...]]:
    """{lever: (implementation, ...)} for levers of the composition that bind their OWN attention call sites through a tokamax implementation other
    than the caller's --flash_attention_implementation (that flag is upstream's: it moves upstream's own sites). Today: DATTN — the DiT's and the
    pairformer's dense attention served by the shared core's provider (opt_core.kernels.pallas serve.attention) by tier word; WHICH rows served is
    read from the lever's own census line on the transcript (inprocess/dattn.py census_line: ``rows=<arm:n,...>``), never assumed: arm
    ``tokamax@triton`` calls under implementation=triton, ``tokamax@xla`` under xla, rows that are not tokamax make no counted call
    (dattn.census_impls). No census line (the process died before exit) = no declared implementation: a census call outside the route's request is
    then refused by name as before."""
    out: Dict[str, Tuple[str, ...]] = {}
    if "DATTN" in (levers or []):
        from .inprocess import dattn as _dattn
        rec = _dattn.scan(lines)
        out["DATTN"] = tuple(_dattn.census_impls(rec["rows"])) if rec else ()
    return out


def expect_env(exp: Dict[str, str]) -> Dict[str, str]:
    """{OPT_KERNELS_EXPECT: 'k=v;…'} — the expectation as the probe reads it (kernels_probe.parse_expect)."""
    return {ENV_EXPECT: ";".join(f"{k}={v}" for k, v in exp.items())}


def arm(env: dict, hook_dir: str, exp: Dict[str, str]) -> dict:
    """Put the probe beside the peak instrument in ``hook_dir`` (the directory peakmem.arm wrote and put LAST on the model process's PYTHONPATH):
    ``kernels_probe.py`` byte for byte, and ONE ``sitecustomize.py`` = the core's peak loader text followed by the probe's boot
    (the interpreter imports one sitecustomize: both instruments ride it); name the expectation in ``env``. Returns {dir, probe_py, sha256, expect}."""
    dst = os.path.join(hook_dir, "kernels_probe.py")
    shutil.copyfile(PROBE_FILE, dst)
    from .stack import sha256_file
    sha = sha256_file(dst)
    with open(os.path.join(hook_dir, "sitecustomize.py"), "w", encoding="utf-8") as fh:
        fh.write(_peak.SITECUSTOMIZE + SITECUSTOMIZE_TAIL)
    env.update(expect_env(exp))
    return {"dir": hook_dir, "probe_py": dst, "sha256": sha, "expect": dict(exp)}


SITECUSTOMIZE_TAIL = ("# the KERNELS probe's loader (af3_jax_opt.kernels.arm): the carried copy beside this file; inert unless OPT_KERNELS_EXPECT names a route's expectation.\n"
                      "try:\n import kernels_probe; kernels_probe.boot('sitecustomize')\n"
                      "except Exception as _e:\n import sys as _s; _s.stderr.write('[af3-jax-opt] KERNELS-PROBE REFUSED: loader: %r; no reading\\n' % (_e,))\n")


def parse_kv(text: str) -> Dict[str, str]:
    out = {}
    for tok in text.strip().split():
        if "=" in tok:
            k, v = tok.split("=", 1); out[k] = v
    return out


def is_kernels_line(ln: str) -> bool:
    """Whether a transcript line belongs to this reader (probe, census, in-process refusal, one of upstream's fallback prints, or the census line of
    a lever that binds its own attention sites — DATTN's `[af3-jax-opt] DATTN word= served= rows= …`, which lever_impls reads: without it the
    arms that lever served under another tokamax implementation than the route's request would be refused although they are named)."""
    from .inprocess import dattn as _dattn
    return bool(PROBE_RX.search(ln) or CENSUS_RX.search(ln) or REFUSED_RX.search(ln) or _dattn.LINE_RX.search(ln) or any(rx.search(ln) for rx in TRAPS.values()))


def scan(lines: List[str]) -> dict:
    """{probe: the FIRST probe line's fields | None, probes: n, census: {dpa: {impl: n}, glu: {impl: n}, lines: n}, traps: {name: [line, …]}, refused_inprocess: bool}."""
    probe, n_probe, refused = None, 0, False
    census = {"dpa": {}, "glu": {}, "lines": 0}
    traps = {k: [] for k in TRAPS}
    for ln in lines:
        m = PROBE_RX.search(ln)
        if m:
            n_probe += 1
            if probe is None:
                probe = parse_kv(m.group("kv"))
            continue
        m = CENSUS_RX.search(ln)
        if m:
            kv = parse_kv(m.group("kv")); census["lines"] += 1
            for key, field in (("dpa", "dpa_calls"), ("glu", "glu_calls")):
                for tok in (kv.get(field) or "none").split(","):
                    if ":" in tok and tok != "none":
                        impl, n = tok.rsplit(":", 1)
                        try:
                            census[key][impl] = census[key].get(impl, 0) + int(n)
                        except ValueError:
                            census[key][impl] = census[key].get(impl, 0)
            continue
        if REFUSED_RX.search(ln):
            refused = True; continue
        for name, rx in TRAPS.items():
            if rx.search(ln):
                traps[name].append(ln.strip())
    return {"probe": probe, "probes": n_probe, "census": census, "traps": traps, "refused_inprocess": refused}


def _flash_word(probe: Optional[dict], traps: dict) -> str:
    if traps.get("cpu_auto_downgrade"):
        return "fallback:cpu-auto-downgrade"
    if not probe:
        return "unread:no_probe_line"
    flag, cls, ver = probe.get("flag", "unread"), probe.get("attn_class", "unread"), probe.get("tokamax", "?")
    if cls == "absent":
        return f"absent:{flag}"
    if probe.get("attn_supported") != "1":
        return f"fallback:{flag}-unsupported"
    return f"engaged:{flag}@tokamax{ver}"


def _glu_word(probe: Optional[dict], traps: dict) -> str:
    if not probe:
        return "unread:no_probe_line"
    head, ver = probe.get("glu_head", "unread"), probe.get("tokamax", "?")
    n_fb = len(traps.get("glu_fallback") or [])
    if n_fb:
        chain = (probe.get("glu_chain") or "").split(",")
        nxt = chain[1] if len(chain) > 1 else "?"
        return f"fallback:{nxt}:{n_fb}"
    if head.startswith("unread") or head == "none":
        return f"unread:{head}"
    return f"engaged:{head}@tokamax{ver}"


def _xla_word(env: dict, caller_environ: Optional[dict] = None) -> str:
    """``upstream`` when the model process's XLA_FLAGS equals the image's pinned Dockerfile value (stock/PINS.json image.env, = stock/src/docker/Dockerfile:93);
    ``caller:<v>`` when the caller's own environment set another; ``other:<v>`` otherwise; ``unset``."""
    want = (pins().get("image", {}).get("env", {}) or {}).get("XLA_FLAGS")
    got = env.get("XLA_FLAGS")
    if got is None:
        return "unset"
    enc = got.replace(" ", "|")
    if got == want:
        return "upstream"
    if caller_environ is not None and caller_environ.get("XLA_FLAGS") == got:
        return f"caller:{enc}"
    return f"other:{enc}"


def _pair_word(probe: Optional[dict]) -> str:
    if not probe:
        return "unread"
    d, t = probe.get("dpa_site", "unread"), probe.get("triatt_site", "unread")
    if t == "fpf":
        return "fpf"
    if d == "attncfg":
        return "attncfg"
    if d == "tokamax" and t == "stock":
        return "tokamax"
    return f"other:{d}/{t}"


def refusal(exp: Dict[str, str], sc: dict, rc: Optional[int] = None, lever_impls: Optional[Dict[str, Sequence[str]]] = None) -> List[str]:
    """The pass's refusal reasons ([] = conforms): the probe's own verdict (kernels_probe.verdict on the first probe line — the same function the
    process ran), each fallback event on the transcript, a census call under an implementation the route does not request, no probe line."""
    probe, traps, census = sc["probe"], sc["traps"], sc["census"]
    reasons: List[str] = []
    if probe is None:
        reasons.append("no KERNELS-PROBE line: the model process never called tokamax through the probe (hook not loaded, or no kernel call) — the reading is missing")
    else:
        reasons += _probe.verdict(probe, exp)
    for name, hits in traps.items():
        if hits and name != "autotune_failed":
            reasons.append(f"{name}: {len(hits)} event(s): {hits[0][:160]}")
    allowed = {exp.get("flag")} | {w for ws in (lever_impls or {}).values() for w in ws}   # the route's requested implementation, plus the implementations a lever of the composition served its OWN call sites under, read from that lever's census line (DATTN: the provider rows on its `rows=` word) — a caller's --flash_attention_implementation moves upstream's sites, not the lever's
    for impl in census["dpa"]:
        if impl in allowed or (impl.startswith("obj:PallasTritonFlashAttention") and exp.get("dpa_site") == "attncfg" and exp.get("flag") == "triton"):
            continue
        reasons.append(f"dpa_calls: {census['dpa'][impl]} attention call(s) under implementation={impl}, the route requests {exp.get('flag')}")
    for impl in census["glu"]:
        if impl not in ("None", exp.get("glu_head")):
            reasons.append(f"glu_calls: {census['glu'][impl]} GLU call(s) under implementation={impl}")
    if sc.get("refused_inprocess") and not reasons:
        reasons.append(f"the probe refused in-process (rc {rc})")
    return reasons


def account(*, mode: str, n_gpu: int, levers: List[str], argv: List[str], env: dict, lines: List[str], rc: Optional[int] = None,
            requested_source: Optional[str] = None, caller_environ: Optional[dict] = None, tree: Optional[str] = None) -> dict:
    """Everything the KERNELS line and the exit rule read, for one pass: {route, mode, requested, expected, scan, words, reasons, ok, line}."""
    req = stock_pred.effective_flash_impl(argv, tree)
    exp = expected(mode, levers, req["impl"])
    sc = scan(lines)
    probe = sc["probe"]
    pinned = next((k for k in sc["census"]["dpa"] if k.startswith("obj:PallasTritonFlashAttention")), None)
    flash = _flash_word(probe, sc["traps"])
    if flash.startswith("engaged:") and exp["dpa_site"] == "attncfg" and probe and probe.get("dpa_site") == "attncfg":
        flash += ":pinned" + (pinned[len("obj:PallasTritonFlashAttention"):] if pinned else "")
    reasons = refusal(exp, sc, rc, lever_impls=lever_impls(levers, lines))
    fmt = lambda d: ",".join(f"{k}:{v}" for k, v in sorted(d.items())) or "none"
    words = {"route": route_name(mode, n_gpu), "mode": mode,
             "flash_impl": flash, "pair_attn": _pair_word(probe), "trimul": (probe or {}).get("trimul_site", "unread"), "glu_impl": _glu_word(probe, sc["traps"]),
             "xla_flags": _xla_word(env, caller_environ), "requested": f"{req['impl']}:{requested_source or req['source']}",
             "tokamax": (probe or {}).get("tokamax", "unread"), "device": (probe or {}).get("device", "unread"), "cc": (probe or {}).get("cc", "unread"),
             "autotune": f"failed:{len(sc['traps']['autotune_failed'])}" if sc["traps"]["autotune_failed"] else "ok",
             "dpa_calls": fmt(sc["census"]["dpa"]), "glu_calls": fmt(sc["census"]["glu"]),
             "probe": "missing" if probe is None else ("refused" if (probe.get("verdict", "ok") != "ok" or sc["refused_inprocess"]) else "ok"),
             "verdict": ("REFUSED:" + "|".join(r.split(":")[0].replace(" ", "_") for r in reasons)) if reasons else "ok"}
    return {"route": words["route"], "mode": mode, "requested": req, "expected": exp, "scan": sc, "words": words,
            "reasons": reasons, "ok": not reasons, "line": _line("KERNELS", **words)}


def kernels_line(acc: dict) -> str:
    return acc["line"]


def done_reason(acc: dict) -> Optional[str]:
    """``kernels_refused:<route>: <reasons>`` for the DONE line, or None."""
    return None if acc["ok"] else f"kernels_refused:{acc['route']}: " + " | ".join(acc["reasons"])
