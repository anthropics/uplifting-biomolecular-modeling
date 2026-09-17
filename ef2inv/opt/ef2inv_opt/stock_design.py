"""The design script — the ONE caller of the upstream cookbook loop, run in its own process by every arm (launch.py).

Every arm imports the ONE upstream file (stock/PINS.json ``upstream.file``, sha-checked against the tree's copy before import): ``off`` with
nothing from the kit on the path; ``exact`` / ``fast`` / ``big`` with the design kit's ``k/`` on ``sys.path``, ``EF2_FAST_KIT`` set by the
launcher, and the kit installed on the imported module from outside (fastkit.install: the featurisation cache, the kit-enable step, the
state guard checked at every design step) — the same bytes, the same calls. The recipe is the file's own
(``ESMFold2Design().load(use_scaling_critics)`` then ``.design(...)``, stock file l.1285-1355): every loop constant runs as shipped
(settings.check_shipped refuses a changed one); the two call arguments the verb sets and the fixed binder length are the named
deviations (settings.py). The per-call observer wraps ``fold_and_get_distogram`` (a module-global the loop looks up by name) to record a
CUDA-synchronised wall per fold, which model folded and whether the call was a design step or a critic; it calls the original and returns
its result unchanged.

usage (through launch.py; direct use is the same argv):
  python -I -m ef2inv_opt.stock_design --mode off|exact|fast|big --cookbook <file> --pins stock/PINS.json --target-name NAME [--target-sequence SEQ]
         [--binder-len L | --binder-name minibinder] --seed S --out DIR [--target-hotspot-ids 56 57] [--det 0|1]
         [--require-fast-env 0|1] [--kit-dir <k/>] [--chunk-size none|N] [--kernel-backend fused|cuequivariance|None] [--upstream-fix ID[,ID...]]
         [--batch-size 1] [--is-antibody 0|1] [--epitope-contact-distance 12.0] [--binder-sequence SEQ] [--use-scaling-critics 0|1]   (the cookbook's own knobs, settings.STOCK_KNOBS)
exit 0 = the output set written; 2 = usage; 3 = a refusal by name (proof, sha, shipped constants, kit activation, the pair-bias int32 bound); 4 = the loop failed
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback

from . import attention as AT, envproof, settings as S, det as D, evidence as EV, fastkit as FK, outputs as O, patches as PT, report as R, hardware as HW, upstream_fix as UF
from .modes import CRITIC_MODELS_ATTR, KIT_SWITCH, MODES, SDK_STANDIN, kit_paths


def sha256_file(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def resolve_target(target_name: str, target_sequence: Optional[str], built_in: Dict[str, str]) -> Tuple[str, Optional[str], str, str]:
    """The cookbook's own target rule (design_binder l.997-1005), applied before anything loads so a usage error exits by name: a
    ``target_name`` that IS a built-in ``TARGET_SEQUENCES`` key runs the built-in route and takes no ``target_sequence`` (the cookbook's
    "is a preset target; omit target_sequence"), any other name is the label of an explicit ``target_sequence``, which is then required
    ("is not a preset target; provide target_sequence"). Returns (target_name, the design() ``target_sequence`` keyword — None on the
    built-in route, route ``built-in`` | ``sequence``, the effective target sequence)."""
    if target_name in built_in:
        if target_sequence is not None:
            raise ValueError(f"{target_name!r} is a preset target; omit target_sequence.")          # the cookbook's words, l.1000
        return target_name, None, "built-in", built_in[target_name]
    if not target_sequence:
        raise ValueError(f"{target_name!r} is not a preset target; provide target_sequence.")      # l.1005
    return target_name, target_sequence, "sequence", target_sequence


def prepare_arm_process(mode, kit_dir: Optional[str] = None) -> None:
    """The arm process's import surface, set BEFORE the environment proof and any upstream import: the cookbook's `import modal` (l.28,
    used only by its cloud app class, never on the local route) resolves to the kit's no-op stand-in `ef2inv_opt._absent_sdk_stub`, registered here
    as ``sys.modules["modal"]`` — no directory holding a `modal` package is ever put on ``sys.path``; on a kit arm the design kit's ``k/``
    directory goes first on the path. The one function every arm process calls for this (this module's main)."""
    from . import _absent_sdk_stub
    have = sys.modules.get("modal")
    if have is not None and have is not _absent_sdk_stub:
        raise RuntimeError(f"modal is already imported from {getattr(have, '__file__', None)!r}: an arm process runs with the kit's stand-in only")
    sys.modules["modal"] = _absent_sdk_stub
    if mode.is_kit:
        if not kit_dir:
            raise ValueError("a kit arm needs the design kit's k/ directory (--kit-dir)")
        if kit_dir not in sys.path:
            sys.path.insert(0, kit_dir)


def import_cookbook(path: str, module_name: str):
    if getattr(sys.modules.get("modal"), "__file__", None) != SDK_STANDIN:          # fail loud: the stand-in is registered (prepare_arm_process) before the cookbook runs
        raise RuntimeError("import_cookbook before prepare_arm_process: sys.modules['modal'] is not the kit's stand-in")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


class ModelSwitchError(RuntimeError):
    """A model switch that did not take effect on a loaded model, read back from the fork's own attributes — the arm exits 3 by name."""


def model_handles(app, attrs: Optional[Tuple[str, ...]] = None) -> List[Tuple[str, object]]:
    """Every ESMFold2 model the loaded cookbook app holds, as (label, model): each dict attribute of the app whose values carry the stock API's
    ``set_kernel_backend`` / ``set_chunk_size`` (``inversion_models``, ``hf_critic_models``, the scaling critics when loaded), labelled
    ``<attribute>:<name>`` — a list, never a by-name merge (the inversion models and the hero critics share their names). One object held under two
    labels is listed once with both labels. ``attrs`` restricts the walk to the named attributes (the critic scope: ``(CRITIC_MODELS_ATTR,)``)."""
    out: List[Tuple[str, object]] = []; seen: Dict[int, int] = {}
    for attr in sorted(vars(app)):
        if attrs is not None and attr not in attrs:
            continue
        d = getattr(app, attr)
        if not isinstance(d, dict) or not d or not all(hasattr(m, "set_kernel_backend") and hasattr(m, "set_chunk_size") for m in d.values()):
            continue
        for name in sorted(d):
            m = d[name]; label = f"{attr}:{name}"
            if id(m) in seen:
                k = seen[id(m)]; out[k] = (out[k][0] + "=" + label, m)
            else:
                seen[id(m)] = len(out); out.append((label, m))
    return out


def _named_modules(root):
    nm = getattr(root, "named_modules", None)
    return list(nm()) if callable(nm) else [("", root)]


def read_back_switches(label: str, model, chunk_size, kernel_backend) -> Tuple[int, List[str]]:
    """Read the applied switches back from the attributes the fork stores them in (the transformers pin, modeling_esmfold2_common.py:
    ``_kernel_backend`` on AttentionPairBias l.1041-1048, Transition l.2507-2518, PairUpdateBlock l.2606-2619; the triangle-multiplication engine's
    ``_use_kernels`` = backend == "cuequivariance", l.2476-2478; ``_chunk_size`` on TriangleMultiplicativeBlock l.2377-2380 and Transition
    l.2505-2510), over the scope the model's own setters reach (modeling_esmfold2_experimental.py:577-588: ``folding_trunk`` and ``confidence_head``
    for both switches, ``structure_head`` for the backend; a model without those attributes is read whole). Returns (modules readback, mismatches
    worded ``<label> <module path> <attribute>=<got> expected=<want>``); ``shipped`` reads back nothing for that switch."""
    n = 0; bad: List[str] = []
    named = [(a, getattr(model, a)) for a in ("folding_trunk", "confidence_head", "structure_head") if getattr(model, a, None) is not None]
    chunk_roots = [(a, r) for a, r in named if a != "structure_head"] or [("", model)]
    backend_roots = named or [("", model)]
    if kernel_backend != S.KERNEL_BACKEND_SHIPPED:
        want_cueq = kernel_backend == "cuequivariance"
        for prefix, root in backend_roots:
            for path, m in _named_modules(root):
                where = ".".join(x for x in (prefix, path) if x) or "<model>"
                if "_kernel_backend" in vars(m):
                    n += 1
                    if vars(m)["_kernel_backend"] != kernel_backend:
                        bad.append(f"{label} {where} _kernel_backend={vars(m)['_kernel_backend']!r} expected={kernel_backend!r}")
                if "_use_kernels" in vars(m):
                    n += 1
                    if bool(vars(m)["_use_kernels"]) != want_cueq:
                        bad.append(f"{label} {where} _use_kernels={vars(m)['_use_kernels']!r} expected={want_cueq!r} (backend {kernel_backend!r})")
    if chunk_size != S.CHUNK_SHIPPED:
        for prefix, root in chunk_roots:
            for path, m in _named_modules(root):
                where = ".".join(x for x in (prefix, path) if x) or "<model>"
                if "_chunk_size" in vars(m):
                    n += 1
                    if vars(m)["_chunk_size"] != chunk_size:
                        bad.append(f"{label} {where} _chunk_size={vars(m)['_chunk_size']!r} expected={chunk_size!r}")
    return n, bad


def _set_and_read_back(handles: List[Tuple[str, object]], chunk_size, kernel_backend) -> int:
    """Call the two setters (the non-shipped ones) on every handle and read them back (read_back_switches); ModelSwitchError by name on any
    module that does not read the applied value or a model exposing none of the fork's attributes. Returns the modules read back."""
    readback = 0; bad: List[str] = []
    for label, m in handles:
        if chunk_size != S.CHUNK_SHIPPED:
            m.set_chunk_size(chunk_size)
        if kernel_backend != S.KERNEL_BACKEND_SHIPPED:
            m.set_kernel_backend(kernel_backend)
        n, b = read_back_switches(label, m, chunk_size, kernel_backend)
        readback += n; bad += b
        if n == 0:
            bad.append(f"{label} no module carries the fork's switch attributes (_kernel_backend / _use_kernels / _chunk_size): nothing to read back")
    if bad:
        raise ModelSwitchError("model switch not applied: " + "; ".join(bad))
    return readback


def apply_model_switches(app, chunk_size, kernel_backend, arm: str, *, critic_chunk_size=S.CHUNK_SHIPPED, critic_kernel_backend=S.KERNEL_BACKEND_SHIPPED) -> Dict[str, object]:
    """The mode's (or the user's) two model switches (settings.parse_chunk_size / parse_kernel_backend), applied right after ``app.load`` through
    the stock API's own one-liners — ``model.set_chunk_size(v)`` (None = the pair stack unchunked, N = chunks of N) and ``model.set_kernel_backend(v)``
    (the fork's literal ``"fused"`` | ``"cuequivariance"`` | ``None``) — in two scopes: ``chunk_size`` / ``kernel_backend`` on EVERY loaded ESMFold2
    model (model_handles: the inversion models AND the critics; the stock arm's flags, exact's pins, a user flag), then ``critic_chunk_size`` /
    ``critic_kernel_backend`` on the HERO CRITICS only (model_handles restricted to ``hf_critic_models``; fast / big: the stock arm's none +
    cuequivariance, modes.Mode.critic_switches) for each key the every-model scope left shipped — an every-model value covers the critics and the
    critic value yields to it. Every applied value is READ BACK per model from the fork's attributes (read_back_switches): a module that does not read
    it raises ModelSwitchError by name (the arm exits 3, ``NOT ACTIVE: model switch not applied: …``), never a silent partial. ``shipped`` touches and
    reads back nothing for that switch and scope. One ``MODEL-SWITCH … scope=all|critics`` line per applied switch and scope is printed only after the
    read-back succeeds, naming the models it covered; returns the applied switches for run.json ``model_switches`` (``critics``: what the hero critics
    fold with, per key, when anything was called on them). What each backend value reaches under grad is the fork's (settings.KERNEL_BACKEND_NOTES);
    the mode's own composition runs after this at the kit's enable, on the inversion models."""
    applied: Dict[str, object] = {}
    crit_chunk = critic_chunk_size if chunk_size == S.CHUNK_SHIPPED else S.CHUNK_SHIPPED                       # the critic scope acts only where the every-model scope made no call
    crit_backend = critic_kernel_backend if kernel_backend == S.KERNEL_BACKEND_SHIPPED else S.KERNEL_BACKEND_SHIPPED
    every = chunk_size != S.CHUNK_SHIPPED or kernel_backend != S.KERNEL_BACKEND_SHIPPED
    critics_only = crit_chunk != S.CHUNK_SHIPPED or crit_backend != S.KERNEL_BACKEND_SHIPPED
    if not every and not critics_only:
        return applied
    if every:
        handles = model_handles(app)
        if not handles:
            raise ModelSwitchError(f"model switch not applied: the loaded app holds no ESMFold2 model (chunk_size={chunk_size!r} kernel_backend={S.kernel_backend_word(kernel_backend)})")
        readback = _set_and_read_back(handles, chunk_size, kernel_backend)
        names = ",".join(label for label, _ in handles)
        if chunk_size != S.CHUNK_SHIPPED:
            applied["chunk_size"] = chunk_size
            print(f"[ef2inv-opt {arm}] MODEL-SWITCH chunk_size={chunk_size} on the loaded ESMFold2 models (shipped: {S.SHIPPED_CALLS['chunk_size']}) models={len(handles)} names={names} readback={readback} scope=all", file=sys.stderr)
        if kernel_backend != S.KERNEL_BACKEND_SHIPPED:
            applied["kernel_backend"] = kernel_backend
            print(f"[ef2inv-opt {arm}] MODEL-SWITCH kernel_backend={S.kernel_backend_word(kernel_backend)} on the loaded ESMFold2 models (shipped: {S.SHIPPED_CALLS['kernel_backend']}) models={len(handles)} names={names} readback={readback} scope=all", file=sys.stderr)
    if critics_only:
        critics = model_handles(app, attrs=(CRITIC_MODELS_ATTR,))
        if not critics:
            raise ModelSwitchError(f"model switch not applied: the loaded app holds no hero critic model under {CRITIC_MODELS_ATTR} (critic_chunk_size={crit_chunk!r} critic_kernel_backend={S.kernel_backend_word(crit_backend)})")
        readback = _set_and_read_back(critics, crit_chunk, crit_backend)
        names = ",".join(label for label, _ in critics)
        if crit_chunk != S.CHUNK_SHIPPED:
            print(f"[ef2inv-opt {arm}] MODEL-SWITCH chunk_size={crit_chunk} on the hero critic models (the stock arm's setting; shipped: {S.SHIPPED_CALLS['chunk_size']}) models={len(critics)} names={names} readback={readback} scope=critics", file=sys.stderr)
        if crit_backend != S.KERNEL_BACKEND_SHIPPED:
            print(f"[ef2inv-opt {arm}] MODEL-SWITCH kernel_backend={S.kernel_backend_word(crit_backend)} on the hero critic models (the stock arm's setting; shipped: {S.SHIPPED_CALLS['kernel_backend']}) models={len(critics)} names={names} readback={readback} scope=critics", file=sys.stderr)
    if critics_only:                                                                                        # the hero critics' effective pair + who set each key, recorded when the critic scope itself made a call
        applied["critics"] = {"chunk_size": chunk_size if chunk_size != S.CHUNK_SHIPPED else crit_chunk,
                              "kernel_backend": kernel_backend if kernel_backend != S.KERNEL_BACKEND_SHIPPED else crit_backend,
                              "scope": {"chunk_size": "all" if chunk_size != S.CHUNK_SHIPPED else ("critics" if crit_chunk != S.CHUNK_SHIPPED else "shipped"),
                                        "kernel_backend": "all" if kernel_backend != S.KERNEL_BACKEND_SHIPPED else ("critics" if crit_backend != S.KERNEL_BACKEND_SHIPPED else "shipped")}}
    return applied


def is_cuda_oom(err_text: str) -> bool:
    """A CUDA out-of-memory failure, by the exception's name or torch's message (torch.OutOfMemoryError / 'CUDA out of memory')."""
    return "OutOfMemoryError" in err_text or "CUDA out of memory" in err_text


def oom_line(arm: str, n_tokens, n_design: int) -> str:
    """The one line an out-of-memory arm exits with: named, with what the mode keeps resident at that size; no fallback is applied."""
    kit = MODES[arm].kit_switch if arm in MODES else None
    pooled = kit is not None and isinstance(n_tokens, int) and FK.trunk_pool_wanted(kit, n_tokens)   # a composition that carries the trunk graph pool at this size (exact, fast at <= POOL_MAX_TOKENS; big carries none)
    hint = (f" — at or below {FK.POOL_MAX_TOKENS} tokens the mode keeps the trunk graph pool's resident activations (README §Modes; LEVER ef2_stepgraph); "
            f"{FK.LEVERS_OFF_VAR}=ef2_stepgraph runs the mode without them (the memory plan keeps what fits)") if pooled else ""
    return f"[ef2inv-opt {arm}] OUT OF MEMORY (CUDA) mode={arm} tokens={n_tokens} after {n_design} design fold(s){hint}. No fallback applied; exit 4."


def absent_setters(a):
    """No ``--chunk-size`` / ``--kernel-backend`` token on the arm argv (plain off, fast, big) = no setter call: the namespace gets the internal
    word (``settings.CHUNK_SHIPPED`` / ``KERNEL_BACKEND_SHIPPED``) AFTER the parse — the flags carry no argparse default because argparse runs a
    flag's ``type`` on a string default and the parsers accept only upstream's values."""
    for key, absent in (("chunk_size", S.CHUNK_SHIPPED), ("kernel_backend", S.KERNEL_BACKEND_SHIPPED)):
        if not hasattr(a, key):
            setattr(a, key, absent)
    return a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", dest="arm", required=True, choices=sorted(MODES))
    ap.add_argument("--cookbook", required=True)
    ap.add_argument("--pins", required=True)
    ap.add_argument("--target-name", required=True, help="the cookbook's main(target_name): one of its built-in TARGET_SEQUENCES names (its own sequence; no --target-sequence), else the name of the explicit --target-sequence")
    ap.add_argument("--target-sequence", default=None, help="the cookbook's main(target_sequence): the target's amino-acid sequence when --target-name is not a built-in")
    ap.add_argument("--binder-len", type=int, default=None, help="a fixed binder length (the kit's PromptFactory with length_ranges (L, L)); absent = --binder-name")
    ap.add_argument("--binder-name", default=None, help=f"the cookbook's main(binder_name): a registered BINDER_PROMPT_FACTORIES entry (default {S.STOCK_BINDER_NAME}); its length is sampled by the cookbook per seed")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-hotspot-ids", nargs="+", default=None, metavar="ID", help="the cookbook's design(target_hotspot_ids), l.1334: the target's hotspot residue ids (absent = None)")
    ap.add_argument("--det", type=int, default=0)
    ap.add_argument("--require-fast-env", type=int, default=0, choices=(0, 1), help="1 = compare every attention / MLP / rotary word with the pinned stack's (attention.FAST_ENV; configs/*.env EF2INV_REQUIRE_FAST_ENV, passed by the launcher) and word any difference on one info line, the arm runs with its levers unchanged (esmc_rope=torch exempt when EF2INV-0003 is applied); 0 = the words report what is bound and the arm runs")
    ap.add_argument("--batch-size", type=int, default=1, help="the cookbook's design(batch_size), l.1331: B trajectories in one process (design.fasta / design.pdb = trajectory 0; every trajectory in critics.json / trajectory.jsonl)")
    ap.add_argument("--is-antibody", type=int, default=None, choices=(0, 1), help="the cookbook's design(is_antibody), l.1331; unset = stock's None (the factory's value / auto-detected)")
    ap.add_argument("--epitope-contact-distance", type=float, default=12.0, help="the cookbook's design(epitope_contact_distance), l.1335")
    ap.add_argument("--binder-sequence", default=None, help="the cookbook's design(binder_sequence): the starting binder (its sequence route, l.1030-1048); its length must be --binder-len")
    ap.add_argument("--use-scaling-critics", type=int, default=0, choices=(0, 1), help="the cookbook's load(use_scaling_critics), l.1285; base 0 (the 15 scaling-critic snapshots are not among the pinned weights); 1 loads them from HF_HOME as the cookbook does")
    ap.add_argument("--kit-dir", default=None)
    ap.add_argument("--warm-only", action="store_true", help="load the models and fold the case once per inversion model (the one-time costs); no design")
    ap.add_argument("--chunk-size", default=argparse.SUPPRESS, type=S.as_argparse_type(S.parse_chunk_size), help="set_chunk_size(v) on every loaded ESMFold2 model after load — absent: no call (the fork's default, 64, untouched) | none (unchunked) | N; the launcher passes the mode's value or the user's")
    ap.add_argument("--kernel-backend", default=argparse.SUPPRESS, type=S.as_argparse_type(S.parse_kernel_backend), help="set_kernel_backend(fused | cuequivariance | None) on every loaded ESMFold2 model after load — absent: no call (shipped = untouched: the cookbook's own choice); the launcher passes the mode's value or the user's")
    ap.add_argument(UF.FLAG, dest="upstream_fix", default=None, metavar="ID[,ID...]", help="apply the named upstream-issue fixes (upstream_issues/<ID>_<slug>.py) in this arm process before any model is built; absent = the upstream library as installed (upstream_fix.py)")
    a = absent_setters(ap.parse_args(argv))
    problem = S.batch_size_problem(MODES[a.arm], a.batch_size)          # a kit arm runs one trajectory per process (its memory plan and graph pools are sized for one): a batch above one is the launcher's usage error, restated here for direct use — before any import of the stack
    if problem:
        print(problem, file=sys.stderr); return 2
    mode = MODES[a.arm]
    pins = json.load(open(a.pins))
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()

    # ---- the import surface (the `modal` stand-in; the kit's k/ on a kit arm), then the proof, before any upstream import ----
    try:
        prepare_arm_process(mode, a.kit_dir)
    except (ValueError, RuntimeError) as e:
        print(f"[ef2inv-opt] {e}", file=sys.stderr); return 2
    prefixes = pins["stock_environment"]["must_be_absent_prefixes"]
    allowed = {KIT_SWITCH: mode.kit_switch} if mode.kit_switch else None
    if allowed is not None and (os.environ.get(FK.LEVERS_OFF_VAR) or "").strip():
        allowed[FK.LEVERS_OFF_VAR] = os.environ[FK.LEVERS_OFF_VAR]         # the ablation word: a kit arm's second allowed variable (resolved by name at install below; recorded in the proof and on the ACTIVE line)
    proof = envproof.prove(prefixes, allowed, arm=a.arm)
    R.log(envproof.line(proof))
    if a.arm == "off":
        json.dump(proof, open(os.path.join(a.out, "stock_env_proof.json"), "w"), indent=1)
    if not proof["clean"]:
        return 3

    # ---- the cookbook file: its bytes match the tree's own vendored copy (both hashed here; no stored digest) ----
    model_opt = os.path.dirname(os.path.dirname(os.path.abspath(a.pins)))
    ref = os.path.join(model_opt, "stock", "src", "cookbook", "tutorials", "binder_design.py")
    got = sha256_file(a.cookbook)
    want = sha256_file(ref)
    if got != want:
        print(f"[ef2inv-opt {a.arm}] REFUSED cookbook file {a.cookbook} sha256 {got} != tree copy {ref} sha256 {want}", file=sys.stderr); return 3

    cap = EV.LogCapture()
    import logging
    logging.getLogger().addHandler(cap)
    t_import = time.time()
    BD = import_cookbook(a.cookbook, "binder_design")
    logging.getLogger().addHandler(cap)      # the cookbook's basicConfig may have reset the root handlers
    import torch
    t_import = time.time() - t_import
    bad = S.check_shipped(BD)
    if bad:
        print(f"[ef2inv-opt {a.arm}] REFUSED the cookbook module's constants differ from the shipped ones: {bad}", file=sys.stderr); return 3
    overrides = {}                                                      # the loop constants run as shipped (checked above); nothing overrides them
    if mode.is_kit and os.environ.get(KIT_SWITCH) != mode.kit_switch:  # the launcher sets the design kit's switch for the mode; its k/ modules read it
        print(f"[ef2inv-opt {a.arm}] REFUSED {KIT_SWITCH}={os.environ.get(KIT_SWITCH)!r} in this process, the mode is {mode.kit_switch!r}", file=sys.stderr); return 3
    user_switches = S.user_switches(mode, a.chunk_size, a.kernel_backend)   # the pair-stack switches THE USER set over the mode's value (applied to every model at load like any switch; the kit's enable reads them so a lever whose contract needs another value steps aside by name)
    try:
        kit_rec = FK.install(BD, mode, user_switches=user_switches) if mode.is_kit else FK.record_off()   # the design kit on the imported stock module (fastkit.py: featurisation cache, kit-enable step, guard check; the ablation word resolved once); off: nothing
    except FK.LeversOffError as e:                                      # the launcher refused it already; a direct launch of this script meets the same words
        print(f"[ef2inv-opt {a.arm}] REFUSED {e}", file=sys.stderr); return 3
    print(f"[ef2inv-opt {a.arm}] PATCH {kit_rec['name']} applied={kit_rec['applied']}" + (f" kit={kit_rec['kit']} sha={kit_rec['function_sha256'][:12]} hooks={','.join(kit_rec['hooks'])}" if kit_rec["applied"] else "")
          + (f" levers_off={','.join(kit_rec['levers_off'])}" if kit_rec.get("levers_off") else ""), file=sys.stderr)
    if not torch.cuda.is_available():
        print(f"[ef2inv-opt {a.arm}] REFUSED no CUDA device (the recipe folds on cuda, stock file l.215/701)", file=sys.stderr); return 3
    fix_ids = UF.parse(a.upstream_fix)                                  # --upstream-fix: the named upstream-issue fixes, applied here before any model is built; [] = the library as installed
    try:
        fix_recs = UF.apply(fix_ids, model_opt, log=lambda m: print(f"[ef2inv-opt {a.arm}] {m}", file=sys.stderr, flush=True))   # one `UPSTREAM-FIX <ID> applied file=… targets=…` line per fix
    except Exception as e:  # noqa: BLE001 — a requested fix that cannot be installed refuses the arm by name; the run never proceeds without it
        print(f"[ef2inv-opt {a.arm}] NOT ACTIVE: {UF.NOT_APPLIED} ({UF.FLAG} {','.join(fix_ids)}): {type(e).__name__}: {e}", file=sys.stderr); return 3
    rope_rec = next((r for r in fix_recs if r.get("name") == PT.ESMC_ROPE_NAME), None)   # upstream fix EF2INV-0003's record (patches.pin_esmc_rope) when requested, else None: ESM-C's RoPE as imported
    rope_pin = rope_rec["impl"] if rope_rec else None
    stack = pins.get("pinned_stack") or {}
    def fast_env_problems(rec):                                         # EF2INV_REQUIRE_FAST_ENV=1: every word compared with the pinned image's; a difference is worded, never refused (attention.py)
        return AT.require_problems(rec, stack.get("image"), (stack.get("image_recipe") or {}).get("base"), rope_pin=rope_pin) if a.require_fast_env else []
    device_name = torch.cuda.get_device_name(0)
    device_mib = int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))          # torch's accounting, recorded beside nvidia-smi's (never gated on)
    smi_name, smi_mib, _ = HW.card()

    # ---- the case ----
    stock_factory = None                                                # the cookbook's own binder route (--binder-len absent, no --binder-sequence): main(binder_name) names a registered factory
    binder_len = a.binder_len
    if binder_len is None and a.binder_sequence is None:
        stock_factory = a.binder_name or S.STOCK_BINDER_NAME
        if stock_factory not in BD.BINDER_PROMPT_FACTORIES:
            print(f"[ef2inv-opt {a.arm}] REFUSED --binder-name {stock_factory!r} is not a registered BINDER_PROMPT_FACTORIES entry of the cookbook ({sorted(BD.BINDER_PROMPT_FACTORIES)})", file=sys.stderr); return 2
        binder_len = len(BD.BINDER_PROMPT_FACTORIES[stock_factory].sample(seed=a.seed).replace("|", ""))   # the length the cookbook samples for this seed (PromptFactory.sample is seed-determined; design() draws the same prompt)
    elif binder_len is None:
        binder_len = len(a.binder_sequence)                             # the cookbook's sequence route: the starting binder sets the length
    try:                                                                # the cookbook's pass-through knobs (settings.STOCK_KNOBS) + the length; usage errors by name
        st = S.for_run(binder_len, binder_name=stock_factory, batch_size=a.batch_size, is_antibody=a.is_antibody, epitope_contact_distance=a.epitope_contact_distance,
                       use_scaling_critics=a.use_scaling_critics, binder_sequence=a.binder_sequence)
    except ValueError as e:
        print(f"[ef2inv-opt {a.arm}] REFUSED {e}", file=sys.stderr); return 3
    try:                                                                # the cookbook's own target rule (a built-in name, or a name + its sequence), before anything loads
        target_name, target_sequence, target_route, target_seq = resolve_target(a.target_name, a.target_sequence, BD.TARGET_SEQUENCES)
    except ValueError as e:
        print(f"[ef2inv-opt {a.arm}] REFUSED target: {e}", file=sys.stderr); return 3
    if stock_factory is not None:                                       # the cookbook's own factory, untouched (main(binder_name=<factory>): its template, lengths and is_antibody as registered)
        binder_name = stock_factory
    elif a.binder_sequence is None:                                     # the fixed-length factory route (upstream l.1030-1040: binder_name names a factory, binder_sequence None)
        binder_name = f"mb{st.binder_len}"
        BD.BINDER_PROMPT_FACTORIES[binder_name] = BD.PromptFactory(name=binder_name, template="{seq}", length_ranges={"seq": (st.binder_len, st.binder_len)}, is_antibody=bool(st.is_antibody))
    else:                                                               # the cookbook's sequence route (l.1041-1048: binder_name NOT a factory, binder_sequence the starting binder)
        binder_name = f"seq{st.binder_len}"
        BD.BINDER_PROMPT_FACTORIES.pop(binder_name, None)
    hotspots = [str(h) for h in a.target_hotspot_ids] if a.target_hotspot_ids else None   # design(target_hotspot_ids): the cookbook's own list, None when absent
    n_tokens = len(target_seq) + st.binder_len
    problem = S.pair_bias_int32_problem(a.arm, a.kernel_backend, st.batch_size, n_tokens)   # fast / big: a batch x length plane past the stock fused pair-bias kernel's int32 offsets is refused by name before anything loads (settings.py)
    if problem:
        print(f"[ef2inv-opt {a.arm}] REFUSED {problem}", file=sys.stderr)
        json.dump({"before": {"refusals": [problem]}}, open(os.path.join(a.out, "activation.json"), "w"), indent=1); return 3   # opt_manifest.json `refusals` carries the words (manifest.py)
    case = {"target_name": target_name, "target_route": target_route, "target_len": len(target_seq), "target_sequence_sha256": hashlib.sha256(target_seq.encode()).hexdigest(),
            "binder_len": st.binder_len, "binder_factory": binder_name,
            "n_tokens": n_tokens, "seed": a.seed, "hotspots": hotspots}
    if a.binder_sequence is not None:                                   # the given starting binder (the sequence route): recorded with the case
        case.update({"binder_route": "sequence", "binder_sequence": a.binder_sequence, "binder_sequence_sha256": hashlib.sha256(a.binder_sequence.encode()).hexdigest()})

    # ---- load, det, observers ----
    t_load = time.time()
    app = BD.ESMFold2Design()
    app.load(use_scaling_critics=st.use_scaling_critics)
    cmode = dataclasses.replace(mode, critic_chunk_size=S.CHUNK_SHIPPED, critic_kernel_backend=S.KERNEL_BACKEND_SHIPPED) if FK.ablated(BD, FK.CRITIC_LEVER) else mode   # `critic_switches` ablated: the hero critics keep the loader's choices (LEVER critic_switches state=ablated)
    critic_switches = cmode.critic_switches(a.chunk_size, a.kernel_backend)   # what the hero critics fold with, per key: an every-model value in force, else the mode's critic-scope value (fast / big: stock's), else shipped
    try:                                                                # the mode's (or the user's) two model switches on EVERY loaded model, then the mode's critic-scope pair on the hero critics; read back, before det / patches / the kit's enable
        model_switches = apply_model_switches(app, a.chunk_size, a.kernel_backend, a.arm, critic_chunk_size=cmode.critic_chunk_size, critic_kernel_backend=cmode.critic_kernel_backend)
    except ModelSwitchError as e:
        print(f"[ef2inv-opt {a.arm}] NOT ACTIVE: {e}", file=sys.stderr); return 3
    torch.cuda.synchronize(); t_load = time.time() - t_load
    R.log(R.ready_line(time.time() - t_start, None, t_load))
    det_rec = D.apply(D.level(a.det))                                  # the det recipe (its attention element: det.DET_ATTN) — no selector
    patch_rec = PT.apply()                                              # upstream bug 0002: applied unconditionally, every arm (the unpatched file cannot run on the pinned stack; upstream_issues/EF2INV-0002)
    print(f"[ef2inv-opt {a.arm}] PATCH {patch_rec['name']} applied={patch_rec['applied']} sha={patch_rec['function_sha256'][:12]}", file=sys.stderr)
    patch_recs = [patch_rec, kit_rec] + ([rope_rec] if rope_rec else [])
    esmcs = {id(m): m for m in [getattr(app, "esmc_model", None)] + [getattr(m, "_esmc", None) for m in list(app.inversion_models.values()) + list(getattr(app, "hf_critic_models", {}).values())] if m is not None}
    attn_rec = AT.read(esmc_models=list(esmcs.values()), det=det_rec)   # THE words of the run: the classes, flags and callables bound now, after det / patches, on the loaded models
    attn_rec["require_fast_env"] = bool(a.require_fast_env); attn_rec["esmc_rope_pin"] = rope_pin; attn_rec["upstream_fix"] = UF.ids_of(fix_recs)
    sentences = fast_env_problems(attn_rec)                             # the guard: what the built models hold (esmc_mlp is decided at build time)
    info_line = AT.guard_ruling(sentences, a.arm)                       # every mode: the one info line on any sentence, never a refusal
    R.log(R.attn_line(AT.words(attn_rec), attn_rec, bool(a.require_fast_env)))
    if info_line:
        R.log(info_line); attn_rec["fast_env_report_only"] = sentences  # the sentences worded once and recorded (run.json attention); the levers engage unchanged
    if a.arm == "off":
        proof["attention"] = attn_rec
        json.dump(proof, open(os.path.join(a.out, "stock_env_proof.json"), "w"), indent=1)
    counter = None
    before = {}
    if mode.is_kit:
        if getattr(app, "_fast_kit_tokens", None) is not None:
            print(f"[ef2inv-opt {a.arm}] REFUSED F7 app._fast_kit_tokens already set before the design", file=sys.stderr); return 3
        try:
            import ef2_autograd_kernels as agk
        except Exception as e:
            print(f"[ef2inv-opt {a.arm}] REFUSED the kit's lever module does not import: {e!r}", file=sys.stderr); return 3
        before = EV.probe_before(a.arm, agk=agk, torch=torch)
        if before["refusals"]:
            print(f"[ef2inv-opt {a.arm}] REFUSED before the loop: {before['refusals']}", file=sys.stderr)
            json.dump({"before": before}, open(os.path.join(a.out, "activation.json"), "w"), indent=1); return 3
        counter = EV.KernelCounter(); counter.install(agk, list(app.inversion_models.values()), torch)
        try:                                                            # the kit's ONE composition for this complex, enabled here so that its loop-level levers and its per-step
            app._enable_fast_kit(n_tokens)                              # guard sit INSIDE the fold observer below (the design() hook's own call is then the named no-op)
        except Exception as e:  # noqa: BLE001 — named, recorded, exit 3 (a lever that cannot install is never skipped silently)
            print(traceback.format_exc(), file=sys.stderr)
            print(f"[ef2inv-opt {a.arm}] REFUSED the kit did not enable for {n_tokens} tokens: {type(e).__name__}: {e}", file=sys.stderr)
            json.dump({"before": before, "enable_error": f"{type(e).__name__}: {e}"}, open(os.path.join(a.out, "activation.json"), "w"), indent=1); return 3
    else:
        agk = None
    names_by_id = {id(m): n for n, m in app.inversion_models.items()}
    names_by_id.update({id(m): n for n, m in app.hf_critic_models.items()})
    calls = []
    orig_fold = BD.fold_and_get_distogram                               # on a kit arm: the kit's outermost fold wrapper (the state guard's per-step check)

    def fold_observed(model, *args, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig_fold(model, *args, **kw)
        torch.cuda.synchronize(); w = time.perf_counter() - t0
        conf = bool(kw.get("calculate_confidence", False)); loops = kw.get("num_loops", 0)
        calls.append({"k": len(calls), "kind": "critic" if (conf and loops == 3) else "design", "wall_s": round(w, 6), "tokens": n_tokens,
                      "confidence": conf, "num_loops": loops, "num_sampling_steps": kw.get("num_sampling_steps"), "model": names_by_id.get(id(model))})
        return r
    BD.fold_and_get_distogram = fold_observed

    if a.warm_only:
        # the one-time costs: import, load, the first fold of each inversion model on the case (the kit enabled above on a kit arm) — in the loop's
        # own grad MODE on a design that requires no grad (no graph is recorded; an enclosing no_grad would be a process-state change the kit's
        # state guard names at the next design fold)
        target_one_hot = BD.sequence_to_one_hot(target_seq)          # [1, L, tokens] (stock file l.215-222)
        logits = BD.build_initial_soft_sequence_logits("#" * st.binder_len, 1)
        design_soft = torch.softmax(logits.detach().cuda(), dim=-1)
        firsts = {}
        for name, m in app.inversion_models.items():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            orig_fold(m, target_seq, target_one_hot, design_soft, num_loops=1, num_sampling_steps=1, calculate_confidence=False, seed=a.seed)
            torch.cuda.synchronize(); firsts[name] = round(time.perf_counter() - t0, 3)
        # then ONE design-step-shaped grad pass per inversion model — the loop's own calls at temperature 1 (run_step, stock file l.1046-1101: the fold
        # under grad + the structure loss backward w.r.t. the logits, then the pseudo-perplexity term fwd+bwd): the GRAD-path kernels a design step runs
        # (stock's compiled blocks with their backward, the kit's Triton kernels on a kit arm) are built into the JIT caches here instead of at design
        # step 0; the CUDA graphs a mode captures are per process and are captured again by the design (seconds), the caches are what carries over
        with torch.device("cuda"):
            score_mask = BD.build_gradient_mask("#" * st.binder_len, 1).sum(dim=-1) > 0      # the loop's own mask rule (l.1040, 1082)
        steps = {}
        for name, m in app.inversion_models.items():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            lg = logits.detach().to("cuda").requires_grad_(True)
            fr = orig_fold(m, target_seq, target_one_hot, torch.softmax(lg, dim=-1), num_loops=1, num_sampling_steps=1, calculate_confidence=False, seed=a.seed)
            sl = BD.compute_structure_losses(fr["distogram_logits"], st.binder_len, target_sequence=target_seq, target_hotspot_ids=hotspots, epitope_contact_distance=st.epitope_contact_distance)["total_loss"]
            torch.autograd.grad(sl.mean(), lg)
            with BD.seed_context(a.seed):
                pl = BD.compute_esmc_pseudoperplexity_nll(esmc_model=app.esmc_model, binder_design=torch.softmax(lg, dim=-1), score_mask=score_mask, batch_size=BD.LM_LOSS_BATCH_SIZE, n_passes=BD.LM_MASK_PASSES)
            torch.autograd.grad(pl.mean(), lg)
            del fr, sl, pl, lg
            torch.cuda.synchronize(); steps[name] = round(time.perf_counter() - t0, 3)
        w = {"arm": a.arm, "import_s": round(t_import, 3), "load_s": round(t_load, 3), "first_folds_s": firsts, "first_steps_s": steps, "n_tokens": n_tokens,
             "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3), "device": device_name}
        json.dump(w, open(os.path.join(a.out, "warm_result.json"), "w"), indent=1)
        with open(os.path.join(a.out, "run.log"), "w") as f:
            f.write(cap.text() + "\n")
        print(f"[ef2inv-opt {a.arm}] WARM import_s={w['import_s']} load_s={w['load_s']} first_folds_s={firsts} first_steps_s={steps}", file=sys.stderr)
        return 0

    # ---- the loop, as shipped ----
    torch.cuda.reset_peak_memory_stats()
    t_design = time.time(); err = None
    try:
        seqs, trajectory, results = app.design(target_name=target_name, target_sequence=target_sequence, binder_name=binder_name, binder_sequence=a.binder_sequence,
                                               seed=a.seed, target_hotspot_ids=hotspots, **st.design_kwargs())
    except Exception as e:
        err = traceback.format_exc(); seqs = trajectory = results = None
    torch.cuda.synchronize(); t_design = time.time() - t_design
    peak_alloc_b, peak_reserved_b = R.cuda_peaks(torch)                  # the item's peaks: reset right before the loop (above), read after its last step
    for line in R.peak_lines(a.target_name, a.seed, peak_alloc_b, peak_reserved_b):
        R.log(line)
    peak_gib = peak_alloc_b / (1024 ** 3)
    log_text = cap.text()
    with open(os.path.join(a.out, "run.log"), "w") as f:
        f.write(log_text + "\n")
    if err:
        n_design = sum(1 for c in calls if c["kind"] == "design")
        print(f"[ef2inv-opt {a.arm}] the loop failed after {n_design} design fold(s) (last call: {calls[-1] if calls else None}):\n{err}", file=sys.stderr)
        oom = is_cuda_oom(err)
        if oom:                                                              # named, never a silent downgrade or a retry: the operator picks the mode
            print(oom_line(a.arm, n_tokens, n_design), file=sys.stderr)
        json.dump({"error": err, "exit_reason": ("cuda_oom" if oom else "exception"), "case": case, "proof": proof, "overrides": overrides, "model_switches": model_switches, "patches": patch_recs, "upstream_fix": fix_recs, "attention": attn_rec,
                   "design_folds_before_failure": n_design, "last_call": (calls[-1] if calls else None), "calls": calls}, open(os.path.join(a.out, "run.json"), "w"), indent=1)
        O.write_steps(os.path.join(a.out, "steps.jsonl"), calls)
        return 4

    # ---- outputs ----
    STEPS = BD.STEPS; TMIN = BD.TEMPERATURE_MIN
    temps = {s: TMIN + (1 - TMIN) * (0.5 * (1 + math.cos(math.pi * ((s + 1) / STEPS)))) for s in range(STEPS)}   # the file's schedule, l.1132-1134
    design_calls = [c for c in calls if c["kind"] == "design"]
    reps = {i: c["model"] for i, c in enumerate(design_calls)}
    n_traj = O.write_trajectory(os.path.join(a.out, "trajectory.jsonl"), trajectory, temps, reps)
    O.write_steps(os.path.join(a.out, "steps.jsonl"), calls)
    binder_seq = seqs[0].split("|")[-1]
    O.write_fasta(os.path.join(a.out, "design.fasta"), a.target_name, target_seq, binder_seq)
    crit_rows = O.write_critics(os.path.join(a.out, "critics.json"), results, a.out)
    first = results[0] if results else None
    if first is None or first.get("complex") is None:
        print(f"[ef2inv-opt {a.arm}] the first hero critic returned no complex", file=sys.stderr); return 4
    O.write_complex(os.path.join(a.out, "design.pdb"), os.path.join(a.out, "design.cif"), first["complex"])
    steady = sorted(c["wall_s"] for c in design_calls[len(design_calls) // 2:]) if design_calls else []
    per_step = [t.get("time") for t in trajectory.values()]
    timing = {"import_s": round(t_import, 3), "load_s": round(t_load, 3), "design_s": round(t_design, 3), "wall_s": round(time.time() - t_start, 3),
              "n_design_folds": len(design_calls), "n_critic_folds": len(calls) - len(design_calls),
              "fold_wall_first_s": design_calls[0]["wall_s"] if design_calls else None,
              "first_calls_s": [design_calls[0]["wall_s"]] + ([c["wall_s"] for c in calls if c["kind"] == "critic"][:1]) if design_calls else [],
              "fold_wall_steady_median_s": (steady[len(steady) // 2] if steady else None),
              "step_time_median_s": (sorted(per_step)[len(per_step) // 2] if per_step else None),
              "step_time_steady_median_s": (sorted(per_step[len(per_step) // 2:])[len(per_step) // 4] if per_step else None),
              "peak_allocated_gib_process": round(peak_gib, 3), "peak_reserved_gib_process": round(peak_reserved_b / (1024 ** 3), 3),
              "peak_allocated_gib_steps_max": max((t.get("peak_allocated_gib", 0) for t in trajectory.values()), default=None)}
    act = {"before": before}
    rc = 0
    if mode.is_kit:
        try:
            import ef2_pppl_graph
            pf = ef2_pppl_graph.stats
        except Exception:
            pf = None
        after = EV.collect_after(a.arm, BD, app, counter, log_text, agk=agk, n_tokens=n_tokens, pppl_stats_fn=pf, n_steps=n_traj, critic_switches=critic_switches,
                                 user_switches=user_switches, batch_size=st.batch_size)   # + the hero critics' switches read back after their folds; the user's pair-stack switches (a lever yields to them by name); B (the pPPL graph's per-step call count)
        act["after"] = after
        for rec in after.get("levers") or []:                           # one LEVER line per lever of the composition, from its own counters
            R.log(R.lever_line(rec))
        if after["refusals"]:                                           # a lever that did not install or confirm WITHOUT a declared reason: not active, exit 3 (a lever that stood aside by its own word is an `asides` record and a LEVER state word, exit 0)
            R.log(R.evidence_line(after, a.arm))
            print(f"[ef2inv-opt {a.arm}] REFUSED after the loop: {after['refusals']}", file=sys.stderr); rc = 3
        else:
            R.log(R.evidence_line(after, a.arm))
    else:
        lines = EV.stock_lever_lines(log_text)
        act["stock_lever_lines"] = lines
        if lines:
            print(f"[ef2inv-opt off] REFUSED a stock arm whose log carries kit lines: {lines[:5]}", file=sys.stderr); rc = 3
        else:
            R.log(R.evidence_line({"applied": True, "fallback": [], "refusals": [], "events": []}, "off"))
    json.dump(act, open(os.path.join(a.out, "activation.json"), "w"), indent=1)
    run = {"arm": a.arm, "mode": {"name": mode.name, "kit_switch": mode.kit_switch, "levers": mode.levers, "numerics_class": mode.numerics_class},
           "cookbook": {"path": os.path.abspath(a.cookbook), "sha256": got, "module": BD.__name__},
           "settings": st.as_dict(), "shipped_constants": {k: getattr(BD, k) for k in S.SHIPPED_CONSTANTS}, "overrides": overrides, "model_switches": model_switches, "case": case, "det": {"level": D.level(a.det), "applied": det_rec}, "patches": patch_recs, "upstream_fix": fix_recs, "attention": attn_rec,
           "launch": {"proof": proof, "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "device": device_name, "nvidia_smi_memory_total_mib": smi_mib, "torch_total_mib": device_mib,
                      "argv": sys.argv[1:], "isolated": bool(sys.flags.isolated)},
           "stock_patch": "upstream_bug_0002" if patch_rec["applied"] else None, "stock_patch_function_sha256": patch_rec["function_sha256"] if patch_rec["applied"] else None,
           "weights": {"hf_home": os.environ.get("HF_HOME"), "snapshots": {repo: w.get("snapshot_commit") for repo, w in (pins.get("weights") or {}).items()}, "pin_note": "stock/PINS.json weights (digested by stock/check_pins.py at launch — sha256 per file through a (path, size, mtime_ns) memo — and afresh under `check`; an unknown checkpoint is named NOT PINNED and runs)"},
           "timing": timing, "n_trajectory_rows": n_traj, "best_sequence": seqs[0], "binder_sequence": binder_seq,
           "critics": [{k: r[k] for k in ("critic_name", "iptm", "final_loss")} for r in crit_rows], "design_structure": "critic 0 (" + str(crit_rows[0]["critic_name"]) + ") complex",
           "exit_code": rc}
    json.dump(run, open(os.path.join(a.out, "run.json"), "w"), indent=1)
    pdb_sha = hashlib.sha256(open(os.path.join(a.out, "design.pdb"), "rb").read()).hexdigest()
    last = trajectory[max(trajectory)] if trajectory else {}
    R.log(R.run_summary_line(a.target_name, a.seed, n_tokens, n_traj, timing, last.get("total_loss"), crit_rows[0].get("iptm"), pdb_sha))
    return rc


if __name__ == "__main__":
    sys.exit(main())
