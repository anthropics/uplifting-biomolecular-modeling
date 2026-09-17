"""ef2inv-opt — design | check | warm (run.sh is a thin wrapper; `python -m ef2inv_opt <verb>` is the same).

  design  --mode off|exact|fast|big --target-name NAME [--target-sequence SEQ] [--binder-len L | --binder-name minibinder] --seed S --out DIR
          [--target-hotspot-ids 56 57] [--det 0|1] [--chunk-size none|N] [--kernel-backend fused|cuequivariance|None]
          [--batch-size 1] [--is-antibody 0|1] [--epitope-contact-distance 12.0] [--binder-sequence SEQ] [--use-scaling-critics 0|1]
          [--upstream-fix ID[,ID...]] [--no-compile] [--allow-partial]
          the second-to-last row = the cookbook's own main()/design() knobs passed through verbatim on every mode (settings.STOCK_KNOBS);
          the cookbook's design in a clean subprocess (launch.py -> stock_design.py); writes DIR/{design.pdb,design.cif,critic_*.pdb,critic_*.cif,critics.json,
          design.fasta,trajectory.jsonl,steps.jsonl,run.json,run.log,launch.log,activation.json,opt_manifest.json} (+ stock_env_proof.json on off;
          launch.log = the subprocess's stderr as the launcher captured it).
          exit 0 = complete (a lever that stood aside by a declared word, or yielded to a user --chunk-size / --kernel-backend, is named on its
          LEVER line and the run completes); 1 = incomplete output set; 2 = usage (among them --batch-size > 1 on a kit mode: supported with
          --mode off only, one sentence); 3 = refused by name (proof, pins, a lever that did not install or confirm without a declared reason)
          unless --allow-partial (recorded as allow_partial: true); 4 = the loop failed.
  check   [--mode M] [--json]   the mode row, the pins (stock/check_pins.py), the cookbook files, the device. ACTIVE <mode>
          ... | REFUSED ...; exit 0 / 3. The card is compared with the pinned card (hardware.py) — any other card is one NOTE line on
          stderr and ACTIVE; a box with no CUDA device is worded (device=None), the design verb refuses it by name.
  warm    --mode M --target-name NAME [--target-sequence SEQ] --binder-len L --out DIR [--no-compile]   the one-time costs (import, load, the first fold and
          the first design-step-shaped grad pass of each inversion model: the JIT caches filled) through the mode's line; writes DIR/warm_result.json and prints WARM ...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List

from . import attention as AT, det as D, fastkit as FK, launch as L, manifest as M, modes as MD, outputs as O, patches as PT, settings as S, report as R, hardware as HW, upstream_fix as UF


UPSTREAM_FIX_FLAG = "--upstream-fix"                 # == upstream_fix.FLAG (the tests lock the two); design and warm carry it, check applies none
UPSTREAM_FIX_HELP = ("apply the named upstream-issue fixes, one file per ID under upstream_issues/ (README 'Known upstream issues'): the arm process installs them "
                     "before any model is built, every mode, off included, prints UPSTREAM-FIX <ID> applied, and run.json / opt_manifest.json record the IDs "
                     "(upstream_fix). Default: none (the upstream library as installed); an unknown ID is refused by name (exit 2), nothing launched")


def core_gate() -> dict:
    """THE pin gate, statement one of every verb (main): the ``opt_core`` this interpreter would import is the one
    opt/pyproject.toml [tool.opt_core] pins — ``_core_gate.gate`` (the shared core's kit_template/_core_gate.py carried byte-identical
    inside this package; standard library only, imports nothing of the core). Absent / older / newer / edited core or an unreadable pin:
    one ``[ef2inv_opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …`` line and exit 3. On a match
    the facts ``{"pinned": …, "installed": …}``; no second pin call is made anywhere in the kit."""
    from ._core_gate import gate
    return gate(__file__, tag="ef2inv_opt")
def _model_opt() -> str:
    return os.environ.get("MODEL_OPT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _cookbook_stock() -> str:
    p = os.environ.get("EF2INV_COOKBOOK_STOCK")
    if p:
        return p
    try:
        import esm
        return os.path.abspath(os.path.join(os.path.dirname(esm.__file__), "..", "cookbook", "tutorials", "binder_design.py"))
    except Exception:
        return ""


def _pins(model_opt: str) -> dict:
    return json.load(open(os.path.join(model_opt, "stock", "PINS.json")))


def require_fast_env() -> bool:
    """``EF2INV_REQUIRE_FAST_ENV=1`` (configs/*.env): every route compares the attention / MLP / rotary words with the pinned stack's
    (attention.FAST_ENV) and words any difference on one info line — the run proceeds, levers unchanged (attention.guard_ruling); unset / 0
    (no config sourced, CPU tests, other stacks): the words report what is bound without comparing."""
    return os.environ.get(AT.REQUIRE_VAR, "0").strip() == "1"


def _mode_from(args_mode):
    env_mode = os.environ.get(MD.ENV_MODE)
    if args_mode and env_mode and args_mode != env_mode:                             # a usage error: exit 2 with the disagreement named (3 is a refusal / NOT ACTIVE)
        print(f"ef2inv-opt: --mode {args_mode} disagrees with {MD.ENV_MODE}={env_mode}", file=sys.stderr); raise SystemExit(2)
    return MD.resolve(args_mode)


def jit_cache_report(environ, key, problems: list) -> dict:
    """The JIT cache facts of ``check``: the switch (``EF2INV_JIT_CACHE``), the shared root (``MODEL_OPT_JIT_ROOT``), the two dirs, the form they imply (modes.jit_cache_state), and
    ``notes`` — worded, never refused, the run proceeds on the dirs as given: a switch and dirs that disagree, and — in the shared form only — a
    dir not keyed by the running stack (``key``; None without a GPU / core; per-box dirs are private and fresh, no key rule applies to them).
    The one refusal appended to ``problems`` is a switch naming an unshipped form (``frozen:<sha>``): a usage error of the switch itself."""
    st = MD.jit_cache_state(environ)
    rep = {"key": key, "switch": st["switch"], "shared_root": st["shared_root"], "form": st["form"], "dirs": st["dirs"], "notes": []}
    if st["switch"] is not None:
        try:
            named = MD.jit_cache_form(st["switch"])
            if st["form"] != "unset" and named != st["form"]:
                rep["notes"].append(f"{MD.JIT_CACHE_VAR}={st['switch']} but the cache dirs imply {st['form']} ({st['dirs']})")
        except ValueError as e:
            problems.append(str(e))
    if st["form"] == "shared":
        for var, d in st["dirs"].items():
            if key and f"/{key}/" not in d.rstrip("/") + "/":
                rep["notes"].append(f"{var}={d} is not keyed by the running stack ({key})")
    return rep


class _Unset:
    def __repr__(self):
        return "UNSET"


UNSET = _Unset()                                                          # argparse default of --chunk-size / --kernel-backend: the flag absent (None is a legal value of both)


def effective_switches(mode, chunk_size=UNSET, kernel_backend=UNSET):
    """The two model switches of a design run = upstream's setters, accepted on EVERY mode exactly as stock accepts them. ``off``: no call unless a
    flag names one (a flag on off is a SETTING, never an override: the stock arm's documented speed settings are ``--chunk-size none --kernel-backend
    cuequivariance``); ``exact``: chunk None + cuequivariance unless a flag names another value — then that value is every model's, as on the stock
    arm with the same flag, and the one exact lever with nothing to attach to (the cuEquivariance tile table under another backend) steps aside by
    name (settings.EXACT_SWITCH_NOTES; never a refusal); ``fast`` / ``big``: no call at load, a differing flag is a user override the kit's own
    pair-stack choice yields to by name (the chunk of 64, the fused triangle multiplication's tile-table companion). UNSET = the flag not passed.
    Returns (effective, user) — ``user`` holds only the flags that differ from the mode's value (settings.user_switches: opt_manifest.json
    ``overrides``, the NOTE lines, and the kit's enable step, which reads them to step aside by name)."""
    eff = {"chunk_size": mode.chunk_size, "kernel_backend": mode.kernel_backend}
    given = {k: v for k, v in (("chunk_size", chunk_size), ("kernel_backend", kernel_backend)) if v is not UNSET}
    for key, val in given.items():
        eff[key] = val
    user = S.user_switches(mode, **given)                                    # off has no value of its own to override: the flag IS the setting (no NOTE, no `overrides` record)
    return eff, user


def switch_notes(mode, user: dict) -> List[str]:
    """The NOTE lines a design / warm run prints for the user's pair-stack switches before the launch: a ``--kernel-backend`` override of any kit
    mode = ONE line of what the value reaches under grad (settings.KERNEL_BACKEND_NOTES); under ``exact`` one more line per switch saying what the
    exact lever set does with it (settings.EXACT_SWITCH_NOTES). A setting on off and a mode's own value are silent."""
    lines = []
    if "kernel_backend" in user:
        lines.append(R.note_line(*S.kernel_backend_note(user["kernel_backend"])))
    if mode.name == "exact":
        lines += [R.note_line(*S.exact_switch_note(k, user[k])) for k in ("chunk_size", "kernel_backend") if k in user]
    return lines


def hardware_facts(pins: dict, probe=None) -> dict:
    """The card comparison of a GPU verb (hardware.gate), worded: the pinned card is silent; any other card — or no nvidia-smi reading — is
    ONE ``NOTE hardware … — <unpinned|unread> card: …; proceeding`` line on stderr, kept as ``note`` in the returned facts (design records them
    in opt_manifest.json ``stack.hardware_gate``), and the verb proceeds: nothing exits on the card's name or memory. No CUDA device at all is
    refused by name where torch is asked (``_check_facts``: ``no CUDA device``), not here."""
    hw = HW.gate(pins, probe)
    line = HW.note(hw)
    if line:
        R.log(line)
        hw["note"] = line
    return hw


WEIGHTS_CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "ef2inv_opt", "weights_sha256.json")   # the design launch's sha256 memo (stock/check_pins.py --weights-cache: an unchanged file is hashed once per box, never identified by size alone)


def _check_facts(mode, model_opt, pins=None, cb=None, require_gpu=False, weights=True, cache=None, jit_environ=None, alias=None) -> dict:
    """The check report of a mode: stock/check_pins.py, the cookbook files, the device and stack. `weights`: the
    weights digested (sha256 per file) or read by presence and size; `cache`: digest through check_pins' (path, size, mtime_ns) memo (the design's launch);
    `jit_environ`: the environment whose JIT-cache dirs are judged (a launch passes the ARM's, launch.arm_jit_view; absent = this process's, the `check` verb);
    `alias`: the alias word the run was invoked by (modes.alias_word; ALIASES is empty as shipped), None under the mode's own name — the ACTIVE line's `alias=`."""
    rep = {"mode": mode.name, "alias": alias, "kit_switch": mode.kit_switch, "levers": mode.levers, "numerics_class": mode.numerics_class, "model_opt": model_opt, "kit_switch_var": MD.KIT_SWITCH}
    rep["opt_core"] = dict(CORE_FACTS) if CORE_FACTS else core_gate()
    cp = subprocess.run([sys.executable, "-I", os.path.join(model_opt, "stock", "check_pins.py"), "--json"] + ([] if weights else ["--no-weights"]) + (["--weights-cache", cache] if (weights and cache) else []), capture_output=True, text=True)
    try:
        rep["pins"] = json.loads(cp.stdout)
    except Exception:
        rep["pins"] = {"ok": False, "stdout": cp.stdout[-500:], "stderr": cp.stderr[-500:]}
    rep["esm"] = f"{rep['pins'].get('facts', {}).get('esm_version', '?')}@{(pins or {}).get('upstream', {}).get('commit', '?')[:8]}" if isinstance(rep["pins"], dict) else "?"
    cb = cb if cb is not None else _cookbook_stock()
    rep["cookbook_stock"] = {"path": cb, "present": bool(cb and os.path.isfile(cb))}
    kp = MD.kit_paths(model_opt)
    rep["kit_presence"] = MD.kit_presence(model_opt)
    try:
        import torch
        rep["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        rep["torch"] = torch.__version__
        rep["stack"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0], "jit_cache_key": MD.jit_cache_key() if torch.cuda.is_available() else None}
    except Exception as e:
        rep["device"] = None; rep["torch"] = "?"; rep["stack"] = {"error": str(e)[:200]}
    rep["attention"] = AT.read()                                          # the upstream modules' flags as this interpreter imports them (the arm pins esmc_rope and applies det_attn itself)
    rep["attention"]["require_fast_env"] = require_fast_env()
    rep["upstream_fixes"] = sorted(UF.available(model_opt))                   # the switchable upstream-issue fixes this tree carries (design | warm --upstream-fix ID); check applies none
    problems = []
    rep["attention"]["fast_env_report_only"] = []
    if rep["attention"]["require_fast_env"]:
        sentences = AT.require_problems(rep["attention"], rope_pin=None)
        line = AT.guard_ruling(sentences, mode.name)                    # every mode: worded on one info line, never a problem
        if line:
            rep["attention"]["fast_env_report_only"] = sentences; rep["attention"]["fast_env_line"] = line
    if not rep["pins"].get("ok"):
        problems.append("pins")
    if not rep["cookbook_stock"]["present"]:
        problems.append("stock cookbook file")
    if mode.is_kit and not rep["kit_presence"]["root_present"]:
        problems.append(f"design kit not found: {rep['kit_presence']['root']}")
    elif mode.is_kit and not rep["kit_presence"]["entry_present"]:
        problems.append(f"design kit incomplete: {rep['kit_presence']['entry']} missing")
    if require_gpu and not rep["device"]:
        problems.append("no CUDA device")
    key = (rep.get("stack") or {}).get("jit_cache_key")
    rep["jit_cache"] = jit_cache_report(os.environ if jit_environ is None else jit_environ, key, problems)
    rep["jit_cache"]["judged"] = "process" if jit_environ is None else "arm"
    if require_gpu:
        rep["hardware"] = hardware_facts(pins or {})                   # worded, never a problem: off the pinned card = one NOTE line, the verb proceeds
    pfacts = rep["pins"].get("facts") or {}
    rep["weights"] = {"hf_home": pfacts.get("hf_home"), "checked": ("sha256 per file" + (" (digest memo)" if cache else "")) if weights else "presence and size",
                      "gate": "warn-and-run: an unknown checkpoint (another revision, differing files) is named NOT PINNED and runs, never refused; weights genuinely missing (no snapshot of the repo, a pinned file absent) is the one refusal by name (stock/check_pins.py weights_gate)",
                      "snapshots": {repo: {"snapshot_commit": w.get("snapshot_commit"), "status": (pfacts.get("weights") or {}).get(repo),
                                           "word": ((pfacts.get("weights_census") or {}).get(repo) or {}).get("word"), "sha256": ((pfacts.get("weights_census") or {}).get(repo) or {}).get("sha256")}
                                    for repo, w in ((pins or {}).get("weights") or {}).items()},
                      "lines": list(pfacts.get("weights_lines") or [])}
    rep["stack_lines"] = list(pfacts.get("stack_lines") or [])                # the pinned stack's two wheels, warn-and-run: one NOT PINNED line per differing version (stock/check_pins.py dist_pins_check)
    rep["weights"]["word"] = ("unpinned" if any(v["word"] == "unpinned" for v in rep["weights"]["snapshots"].values()) else
                              "missing" if any(v["word"] in ("missing", None) for v in rep["weights"]["snapshots"].values()) else "pinned")
    rep["ok"] = not problems; rep["problems"] = problems
    return rep


def cmd_check(a) -> int:
    model_opt = _model_opt()
    try:
        mode = _mode_from(a.mode)
    except ValueError as e:
        print(f"REFUSED {e}"); return 3
    rep = _check_facts(mode, model_opt, _pins(model_opt), require_gpu=False, jit_environ=dict(os.environ), alias=MD.alias_word(a.mode))   # the deployment's dirs as this process was started (read before the facts import torch); a missing device is worded, never refused here
    rep["jit_cache"]["judged"] = "process"
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        for line in rep["stack_lines"] + rep["weights"]["lines"]:
            if "NOT PINNED" in line:
                print(line)                                               # warn-and-run: an unpinned wheel / unpinned weights are named, never refused
        for note in rep["jit_cache"].get("notes", []):
            print(R.note_line(f"JIT cache {note}", "the dirs are used as given"))
        if rep["attention"].get("fast_env_line"):
            print(rep["attention"]["fast_env_line"])                      # a box without the fast environment: the one info line, check proceeds to its result
        print(("ACTIVE " if rep["ok"] else "REFUSED ") + f"{mode.name}{R.alias_token(rep.get('alias'))} switch={mode.kit_switch} class={mode.numerics_class!r} device={rep['device']} cache={rep['jit_cache']['form']} weights={rep['weights']['word']} "
              f"flash_attn={rep['attention'].get('flash_attn_version') or 'none'} transformer_engine={rep['attention'].get('transformer_engine_version') or 'none'} {AT.words(rep['attention'])} "
              f"require_fast_env={int(rep['attention']['require_fast_env'])} problems={rep['problems']}")
    return 0 if rep["ok"] else 3


NO_COMPILE_HELP = ("the opt-out of KIT-ADDED torch.compile levers (the same as MODEL_OPT_LEVERS_OFF=compile). This kit adds none — stock's own compile of the inversion "
                   "models' MSA encoder and pair-update blocks is inherited untouched in every mode — so the flag is accepted, noted once (compile=stock:not_kit_added on a NOTE "
                   "line and on design's ACTIVE line) and changes nothing")
NO_COMPILE_FLAG = "--no-compile"                     # the sanctioned opt-out of KIT-ADDED torch.compile levers, an alias of MODEL_OPT_LEVERS_OFF=compile (fastkit.COMPILE_LEVER): one mechanism


def compile_optout(a, word: str):
    """``--no-compile`` / the ablation word naming ``compile``: this kit adds no torch.compile lever — the only torch.compile of any mode is stock's own
    (the cookbook's COMPILE), inherited untouched — so either spelling is ACCEPTED (never a refusal), worded on ONE NOTE line and on the ACTIVE line
    (``compile=stock:not_kit_added``), and changes nothing of the arm. Returns ``(asked, the word without `compile`)``."""
    names = [w.strip() for w in (word or "").split(",") if w.strip()]
    asked = bool(getattr(a, "no_compile", False)) or FK.COMPILE_LEVER in names
    if asked:
        R.log(R.note_line(f"{NO_COMPILE_FLAG} ({FK.LEVERS_OFF_VAR}={FK.COMPILE_LEVER}): compile={FK.compile_word(True)}",
                          "this kit adds no torch.compile lever to switch off; stock's own compile (the cookbook's COMPILE: the inversion models' MSA encoder and pair-update blocks) is inherited untouched in every mode"))
    return asked, ",".join(n for n in names if n != FK.COMPILE_LEVER)


def launch_context(a, det: int):
    """What a launch establishes before composing the arm's argv (the design verb; any caller launching an arm of this package the same way):
    the base environment read before anything imports torch (launch.launch_base: the JIT-cache pair completed by the shared-root rule, one
    NOTE per completed variable), the mode, the pins, the stock cookbook, the det-arm cache word (+ the DEVIATION line of a det
    arm on a shared JIT cache), the arm's environment (launch.arm_env from that base: the per-box pair at --det >= 1), the launch facts with
    the JIT dirs judged on what THE ARM runs with (refused by name -> rc 3), the NOT PINNED weight lines, the effective model switches, the
    and one JIT line naming the judged pair. ``a``: the verb's namespace (mode, chunk_size / kernel_backend when present).
    Returns (ctx, rc): rc != 0 = the verb returns it (the refusal already printed)."""
    base, jit_notes = L.launch_base()                                       # first: _check_facts imports torch, whose inductor writes its /tmp default into os.environ when TORCHINDUCTOR_CACHE_DIR is unset
    model_opt = _model_opt()
    try:
        mode = _mode_from(a.mode)
    except ValueError as e:
        R.log(R.not_active_line(f"REFUSED {e}")); return None, 3
    for what, consequence in jit_notes:
        R.log(R.note_line(what, consequence))
    ablated: List[str] = []                                                  # the ablation word MODEL_OPT_LEVERS_OFF, resolved by name ONCE here (a typo or a lever the arm does not compose: refused before any launch)
    no_compile, word = compile_optout(a, base.get(FK.LEVERS_OFF_VAR) or "")   # `compile` (or --no-compile): accepted and worded, every mode; the rest of the word is the ablation
    if FK.LEVERS_OFF_VAR in base and word != (base.get(FK.LEVERS_OFF_VAR) or "").strip():
        base[FK.LEVERS_OFF_VAR] = word                                        # the arm's word without `compile` (nothing there to switch): an arm asked only that runs a plain run's environment
        if not word:
            del base[FK.LEVERS_OFF_VAR]
    if word and not mode.is_kit:
        R.log(R.note_line(f"{FK.LEVERS_OFF_VAR}={word} names kit levers and --mode off installs none", "the stock arm ignores it (the variable is stripped like every kit variable)"))
    elif word:
        try:
            res = FK.resolve_levers_off(word, mode)
        except FK.LeversOffError as e:
            R.log(R.not_active_line(f"REFUSED {e}")); return None, 3
        ablated = res["off"]
        R.log(R.note_line(f"{FK.LEVERS_OFF_VAR}={word}: levers {','.join(ablated)} switched off by name" + (f" ({', '.join(f'{k} serves {v}' for k, v in res['chained'].items())})" if res["chained"] else ""),
                          f"an ablation of --mode {mode.name}, not the mode (ACTIVE ablated=…; LEVER state=ablated)"))
    pins = _pins(model_opt)
    cb = _cookbook_stock()
    if not (cb and os.path.isfile(cb)):
        R.log(R.not_active_line(f"the stock cookbook file is not installed ({cb!r}): set EF2INV_COOKBOOK_STOCK")); return None, 3
    try:
        cache = L.det_cache(base, det)                             # --det >= 1: the per-box JIT cache unless the operator's switch names a form
    except ValueError as e:
        R.log(R.not_active_line(str(e))); return None, 3
    env = L.arm_env(mode, pins["stock_environment"]["must_be_absent_prefixes"], base=base, det=det)
    facts = _check_facts(mode, model_opt, pins, cb, require_gpu=True, weights=True, cache=WEIGHTS_CACHE, jit_environ=L.arm_jit_view(base, env), alias=MD.alias_word(a.mode))   # digested at launch (memoized); the JIT dirs judged = the arm's; `alias`: the word given when it is an alias (modes.ALIASES, empty as shipped), for the ACTIVE line
    if facts["problems"]:
        R.log(R.not_active_line("; ".join(facts["problems"]))); return None, 3
    for line in facts.get("stack_lines", []) + (facts.get("weights") or {}).get("lines", []):
        if "NOT PINNED" in line:
            R.log("[ef2inv-opt] " + line)                                   # warn-and-run: the arm runs on an unpinned wheel / unpinned weights, named on this line and in opt_manifest.json
    for note in (facts.get("jit_cache") or {}).get("notes", []):
        R.log(R.note_line(f"JIT cache {note}", "the dirs are used as given"))   # worded, never refused
    facts["cache"] = R.cache_word(cache["form"], cache["explicit"])
    facts["ablated"] = ablated
    facts["compile"] = FK.compile_word(no_compile)                          # the ACTIVE line's compile= word: stock (inherited) | stock:not_kit_added (the opt-out asked; nothing kit-added to switch)
    if det >= 1 and cache["form"] != "per-box":
        R.log(f"[ef2inv-opt] DEVIATION det arm (--det {det}) on a {cache['form']} JIT cache ({MD.JIT_CACHE_VAR} set by the operator): "
              "kernel and autotune choices written by other boxes are visible to this process")
    jc = facts.get("jit_cache") or {}
    R.log(f"JIT det={det} cache={facts['cache']} judged={jc.get('judged')} form={jc.get('form')} key={jc.get('key')} "
          + " ".join(f"{v}={env.get(v) or 'unset'}" for v in MD.JIT_DIR_VARS) + f" base=" + ",".join(f"{v}={base.get(v) or 'unset'}" for v in MD.JIT_DIR_VARS) + L.env_words(env, det))
    switches, user = effective_switches(mode, getattr(a, "chunk_size", UNSET), getattr(a, "kernel_backend", UNSET))   # every mode accepts both switches (a user value under a kit mode is recorded, worded, and yielded to by name)
    return {"model_opt": model_opt, "mode": mode, "pins": pins, "cookbook_stock": cb, "facts": facts, "cache": cache, "switches": switches, "user_overrides": user, "env": env,
            "base": base, "jit_notes": jit_notes}, 0


def stock_knobs(a) -> dict:
    """The cookbook's pass-through knobs of a design namespace (settings.STOCK_KNOBS; absent attributes = the defaults) as ``launch.argv_for`` /
    ``settings.for_run`` keywords."""
    return {"batch_size": getattr(a, "batch_size", 1) if getattr(a, "batch_size", None) is not None else 1, "is_antibody": getattr(a, "is_antibody", None),
            "epitope_contact_distance": getattr(a, "epitope_contact_distance", 12.0) if getattr(a, "epitope_contact_distance", None) is not None else 12.0,
            "binder_sequence": getattr(a, "binder_sequence", None), "use_scaling_critics": int(getattr(a, "use_scaling_critics", 0) or 0)}


def target_len_word(a):
    """The RUN line's ``target_len``: the explicit --target-sequence's length, or ``built-in`` (the cookbook resolves its own target by name in the arm)."""
    seq = getattr(a, "target_sequence", None)
    return len(seq) if seq else "built-in"


def resolve_upstream_fix(a, model_opt: str, verb: str) -> List[str]:
    """``--upstream-fix``: the requested IDs, each with a file under upstream_issues/ and its run precondition met (upstream_fix.resolve with the
    run's words ``{"verb", "mode"}``); ``[]`` when the flag is absent. Unknown / refused IDs raise ``ValueError`` subclasses naming them — the
    verbs turn that into a usage error (exit 2) before anything launches."""
    ids = UF.parse(getattr(a, "upstream_fix", None))
    return [fid for fid, _ in UF.resolve(ids, model_opt, run={"verb": verb, "mode": getattr(a, "mode", None)})]


def cmd_design(a) -> int:
    knobs = stock_knobs(a)
    problem = S.batch_size_problem(_mode_from(a.mode), knobs["batch_size"])  # a design batch above one on a kit mode: ONE sentence, exit 2 (usage), before anything launches or loads — the kit's memory plan and graph pools are sized for one trajectory per process; --mode off passes B through
    if problem:
        print(problem, file=sys.stderr); return 2
    try:                                                                    # usage errors (a non-positive size, a binder sequence of the wrong length) exit 3 by name before any launch
        S.for_run(a.binder_len, binder_name=getattr(a, "binder_name", None), **knobs)
    except ValueError as e:
        print(f"REFUSED {e}", file=sys.stderr); return 3
    try:                                                                    # --upstream-fix: every ID a file under upstream_issues/, resolved before anything launches (an unknown ID is a usage error by name)
        fix_ids = resolve_upstream_fix(a, _model_opt(), "design")
    except (UF.UnknownUpstreamFix, UF.UpstreamFixRefused) as e:
        print(f"run.sh design: {e}", file=sys.stderr); return 2
    ctx, rc = launch_context(a, a.det)
    if rc:
        return rc
    model_opt, mode, pins, cb, facts, switches, user = (ctx[k] for k in ("model_opt", "mode", "pins", "cookbook_stock", "facts", "switches", "user_overrides"))
    os.makedirs(a.out, exist_ok=True)
    R.log(R.active_line(mode, facts))
    R.log(R.run_line(mode.name, a.target_name, target_len_word(a), S.binder_word(a.binder_len, getattr(a, "binder_name", None), knobs["binder_sequence"]), None, a.seed, S.SHIPPED_CONSTANTS["STEPS"]))
    for line in switch_notes(mode, user):                                   # a user pair-stack switch over a kit mode's value: the backend's grad facts, and under exact what the lever set does with it — worded before the launch, never refused; a setting on off and a mode's own value are silent
        R.log(line)
    argv = L.argv_for(mode, model_opt, cookbook_stock=cb, target_name=a.target_name, target_sequence=getattr(a, "target_sequence", None), binder_len=a.binder_len,
                      seed=a.seed, out=os.path.abspath(a.out), target_hotspot_ids=getattr(a, "target_hotspot_ids", None), det=a.det,
                      binder_name=getattr(a, "binder_name", None), chunk_size=switches["chunk_size"], kernel_backend=switches["kernel_backend"],
                      require_fast_env=int(require_fast_env()), upstream_fix=fix_ids, **knobs)
    env = ctx["env"]
    lp = os.path.join(a.out, "run.log")
    if os.path.exists(lp):
        os.remove(lp)
    res = L.run(argv, env, os.path.join(a.out, "launch.log"))
    stack_key = (facts.get("stack") or {}).get("jit_cache_key")            # the running stack named by what it is: torch<v>-cu<NNN>-sm<cc> (modes.jit_cache_key; None without a CUDA device)
    man = M.write(a.out, mode, model_opt, pins, cb, res, stack_key, allow_partial=a.allow_partial, facts=facts, switches=switches, user_overrides=user)
    rc = res["exit_code"]
    if rc == 0 and man["missing_outputs"]:
        rc = 1
    if rc == 3 and a.allow_partial:
        print(f"[ef2inv-opt] partial by name, allowed: {man['refusals']}", file=sys.stderr); rc = 0
    R.log(R.exit_line(rc, mode, man["evidence"], a.out, facts["cache"]))
    return rc


def cmd_warm(a) -> int:
    model_opt = _model_opt()
    try:
        mode = _mode_from(a.mode)
    except ValueError as e:
        print(f"REFUSED {e}", file=sys.stderr); return 3
    pins = _pins(model_opt); cb = _cookbook_stock()
    try:                                                                  # --upstream-fix resolved before anything launches, as on design
        fix_ids = resolve_upstream_fix(a, model_opt, "warm")
    except (UF.UnknownUpstreamFix, UF.UpstreamFixRefused) as e:
        print(f"run.sh warm: {e}", file=sys.stderr); return 2
    hardware_facts(pins)                                                  # the card worded before any pass (one NOTE line off the pinned card); nothing exits on it
    os.makedirs(a.out, exist_ok=True)
    switches, user = effective_switches(mode, getattr(a, "chunk_size", UNSET), getattr(a, "kernel_backend", UNSET))   # the warm folds run the same setters the design run will (--chunk-size / --kernel-backend as on design, every mode)
    for line in switch_notes(mode, user):
        R.log(line)
    argv = L.argv_for(mode, model_opt, cookbook_stock=cb, target_name=a.target_name, target_sequence=getattr(a, "target_sequence", None), binder_len=a.binder_len,
                      seed=0, out=os.path.abspath(a.out), chunk_size=switches["chunk_size"], kernel_backend=switches["kernel_backend"],
                      require_fast_env=int(require_fast_env()), upstream_fix=fix_ids) + ["--warm-only"]
    _asked, word = compile_optout(a, os.environ.get(FK.LEVERS_OFF_VAR) or "")   # `compile` (or --no-compile): accepted and worded; nothing kit-added to switch
    env = L.arm_env(mode, pins["stock_environment"]["must_be_absent_prefixes"])
    if FK.LEVERS_OFF_VAR in env:                                            # the arm's word without `compile`; named alone, the arm's environment is a plain run's
        env[FK.LEVERS_OFF_VAR] = word
        if not word:
            del env[FK.LEVERS_OFF_VAR]
    res = L.run(argv, env, os.path.join(a.out, "launch.log"))
    wr = os.path.join(a.out, "warm_result.json")
    w = json.load(open(wr)) if os.path.isfile(wr) else {}
    w["exit_code"] = res["exit_code"]; json.dump(w, open(wr, "w"), indent=1)
    print(f"WARM {'PASS' if res['exit_code'] == 0 else 'FAIL'} mode={mode.name} import_s={w.get('import_s')} load_s={w.get('load_s')} first_folds_s={w.get('first_folds_s')} first_steps_s={w.get('first_steps_s')} exit={res['exit_code']}")
    return res["exit_code"]


CORE_FACTS: dict = {}


def main(argv=None) -> int:
    CORE_FACTS.update(core_gate())                                        # statement one: the pinned core, else NOT ACTIVE (exit 3) before anything else runs
    ap = argparse.ArgumentParser(prog="ef2inv-opt", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design"); d.add_argument("--mode"); d.add_argument("--target-name", required=True, help="the cookbook's main(target_name): one of its built-in target names (e.g. pd-l1; then no --target-sequence), else your name for the explicit --target-sequence"); d.add_argument("--target-sequence", default=None, help="the cookbook's main(target_sequence): the target's amino-acid sequence (required unless --target-name is a built-in)")
    d.add_argument("--binder-len", type=int, default=None, help="a fixed binder length L (a PromptFactory with length_ranges (L, L)); absent = the cookbook's own binder route, --binder-name")
    d.add_argument("--binder-name", default=None, help=f"the cookbook's main(binder_name): a registered BINDER_PROMPT_FACTORIES entry whose length the cookbook samples per seed (used when --binder-len is absent; default {S.STOCK_BINDER_NAME}, stock's __main__ value, lengths 60-200)")
    d.add_argument("--seed", type=int, required=True); d.add_argument("--out", required=True)
    d.add_argument("--target-hotspot-ids", nargs="+", default=None, metavar="ID", help="the cookbook's design(target_hotspot_ids): the target's hotspot residue ids, e.g. --target-hotspot-ids 56 57 (absent = None, unconditioned)")
    d.add_argument("--det", type=int, default=0); d.add_argument("--allow-partial", action="store_true")
    d.add_argument("--chunk-size", default=UNSET, type=S.as_argparse_type(S.parse_chunk_size), help="upstream's set_chunk_size(v) on every loaded ESMFold2 model right after load: none (unchunked) | N; flag absent = no setter call, each model keeps the fork's own chunk (64) — the flag itself has no default value; the stock arm passes none, exact's own value is none, a differing flag on exact / fast / big is a recorded override every model takes (exact: one NOTE line; fast / big: the kit's chunk of 64 yields to it by name on the LEVER chunk line)")
    d.add_argument("--batch-size", type=int, default=1, help="the cookbook's design(batch_size) (stock default 1): B trajectories in one process — design.fasta / design.pdb = trajectory 0, every trajectory in critics.json (batch_idx) and trajectory.jsonl")
    d.add_argument("--is-antibody", type=int, default=None, choices=(0, 1), help="the cookbook's design(is_antibody); unset = stock's None (the factory's value / auto-detected from --binder-sequence)")
    d.add_argument("--epitope-contact-distance", type=float, default=12.0, help="the cookbook's design(epitope_contact_distance) (stock default 12.0)")
    d.add_argument("--binder-sequence", default=None, help="the cookbook's design(binder_sequence): a starting binder (stock default None); its length must equal --binder-len")
    d.add_argument("--use-scaling-critics", type=int, default=0, choices=(0, 1), help="the cookbook's load(use_scaling_critics); base 0 — a named difference from stock's main() default 1: the 15 scaling-critic snapshots (STOCK.md Weights) are not among the six pinned weight snapshots; 1 loads them from HF_HOME as the cookbook does")
    d.add_argument("--kernel-backend", default=UNSET, type=S.as_argparse_type(S.parse_kernel_backend), help="upstream's set_kernel_backend(v) on every loaded ESMFold2 model right after load: cuequivariance | fused | None; absent = no call (the cookbook's own fused call); the stock arm passes cuequivariance, exact's own value is cuequivariance, a differing flag on exact / fast / big is a recorded override every model takes, with one NOTE line saying what it reaches under grad (exact: one more NOTE — the cuEquivariance tile table steps aside by name under another backend)")
    d.add_argument(UPSTREAM_FIX_FLAG, dest="upstream_fix", default=None, metavar="ID[,ID...]", help=UPSTREAM_FIX_HELP)
    d.add_argument(NO_COMPILE_FLAG, dest="no_compile", action="store_true", help=NO_COMPILE_HELP); d.set_defaults(fn=cmd_design)
    c = sub.add_parser("check"); c.add_argument("--mode"); c.add_argument("--json", action="store_true"); c.set_defaults(fn=cmd_check)
    w = sub.add_parser("warm"); w.add_argument("--mode"); w.add_argument("--target-name", required=True, help="the cookbook's main(target_name): one of its built-in target names (e.g. pd-l1; then no --target-sequence), else your name for the explicit --target-sequence"); w.add_argument("--target-sequence", default=None, help="the cookbook's main(target_sequence): the target's amino-acid sequence (required unless --target-name is a built-in)"); w.add_argument("--binder-len", type=int, required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--chunk-size", default=UNSET, type=S.as_argparse_type(S.parse_chunk_size)); w.add_argument("--kernel-backend", default=UNSET, type=S.as_argparse_type(S.parse_kernel_backend))
    w.add_argument(UPSTREAM_FIX_FLAG, dest="upstream_fix", default=None, metavar="ID[,ID...]", help=UPSTREAM_FIX_HELP)
    w.add_argument(NO_COMPILE_FLAG, dest="no_compile", action="store_true", help=NO_COMPILE_HELP); w.set_defaults(fn=cmd_warm)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
