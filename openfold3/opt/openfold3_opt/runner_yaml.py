"""A caller's ``--runner-yaml`` under a kit mode is COMPOSED UNDER THE MODE'S CONFIGURATION: every execution key the mode's own runner yaml
(its member: cli.member_yaml) writes is laid on top of the caller's document and each override is named on stderr; the caller's other keys run
as written (execution keys the member does not write are named where they differ); workload sections (seeds, template / MSA / output settings —
everything but ``pl_trainer_args`` and ``model_update``) are never touched. ``model_update.presets`` composes as the ordered union (the caller's,
then the member's missing ones), named only when that changes it. ``--mode off`` composes it the same way under the stock configuration (cli.row_yaml); one of the tree's own configuration files named under off is the base as given (cli.kit_base). The graphed line (`fast`)
additionally pins upstream's four alternative attention kernel flags off (its member leaves three of them at upstream's off default; the CUDA
graphs do not capture those kernels). The row-sharded launcher lays its run plan on top of the member's keys with the same primitive
(tp.pinned_yaml).
"""
import os
import sys
from typing import List, Optional, Tuple

TAG = "openfold3-opt"
EXECUTION_SECTIONS = ("pl_trainer_args", "model_update")              # upstream's runner-yaml sections that say HOW the model runs; every other section says WHAT is run
KERNEL_EVAL_FLAGS = ("use_deepspeed_evo_attention", "use_cueq_triangle_kernels", "use_triton_triangle_kernels", "use_lma")   # settings.memory.eval: upstream's third-party / alternative attention kernels
EVAL_PATH = ("model_update", "custom", "settings", "memory", "eval")
PRESETS_PATH = ("model_update", "presets")                            # upstream's preset list: composed as an ordered union, not replaced
COMPOSED_PREFIX = "runner_composed_"                                  # <out_dir>/runner_composed_<caller basename>: the written composition
CALLERS_PREFIX = "runner_callers_"                                    # <out_dir>/runner_callers_<n>_<last basename>: several --runner-yaml laid in argv order (merge_callers), then composed as one caller
MISSING = "unset"                                                     # the word for a key the member yaml does not write (upstream's default applies there)


def line_pins(line, member: str) -> dict:
    """The runner-yaml document a mode lays over a caller's yaml: the execution sections of its member yaml (every key the mode's own yaml
    writes), plus — on a graphed line over the kernels-off base (fast) — upstream's four alternative attention kernel flags off. The exact
    line captures its graphs over stock's own kernels (its graphed sampler turns the DS4Sci attention off inside the rollout itself): its pins
    are its member's keys alone."""
    import copy
    from . import modes
    doc = _load(member)
    pins: dict = {s: copy.deepcopy(doc[s]) for s in EXECUTION_SECTIONS if isinstance(doc.get(s), dict)}
    if line is not None and modes.line_graphed(line) and not modes.stock_kernel_line(line):
        node = pins
        for key in EVAL_PATH:
            node = node.setdefault(key, {})
        for flag in KERNEL_EVAL_FLAGS:
            node.setdefault(flag, False)
    return pins


def deep_overlay(base, top, _path: Tuple[str, ...] = ()):
    """`top` laid over `base` (dicts merge, anything else replaces). Returns (merged, overrides, added): overrides = [(dotted key, base value,
    top value)] where base held a different value; added = [(dotted key, top value)] where base did not hold the key."""
    import copy
    out = dict(base) if isinstance(base, dict) else {}
    overrides: List[tuple] = []
    added: List[tuple] = []
    for k, v in top.items():
        p = _path + (str(k),)
        if isinstance(v, dict):
            if isinstance(out.get(k), dict):                             # both mappings: merge below
                sub, o, a = deep_overlay(out[k], v, p)
                overrides.extend(o); added.extend(a); out[k] = sub
            elif k in out and out[k] is not None:                        # a scalar / list where the line needs a mapping: replaced whole, named
                overrides.append((".".join(p), out[k], v)); out[k] = copy.deepcopy(v)
            else:                                                        # absent (or an empty `key:`): the line's mapping added
                added.extend(_leaves(v, p)); out[k] = copy.deepcopy(v)
        else:
            if p == PRESETS_PATH and isinstance(v, list) and isinstance(out.get(k), list):   # model_update.presets: the ordered union (the caller's, then the line's missing ones), named only when that changes it
                v = list(out[k]) + [x for x in v if x not in out[k]]
            if k in out:
                if out[k] != v:
                    overrides.append((".".join(p), out[k], v))
            else:
                added.append((".".join(p), v))
            out[k] = v
    return out, overrides, added


def _leaves(doc, path: Tuple[str, ...] = ()) -> List[tuple]:
    if isinstance(doc, dict) and doc:
        out = []
        for k, v in doc.items():
            out.extend(_leaves(v, path + (str(k),)))
        return out
    return [(".".join(path), doc)]


def execution_leaves(doc) -> dict:
    """{dotted key: value} for every leaf under the EXECUTION_SECTIONS of a runner-yaml document (a list is a leaf)."""
    out = {}
    for section in EXECUTION_SECTIONS:
        if isinstance(doc, dict) and section in doc:
            for k, v in _leaves(doc[section], (section,)):
                out[k] = v
    return out


def member_differences(caller_doc, member_doc, pinned=()) -> Tuple[List[tuple], List[tuple]]:
    """(kept, absent): kept = [(dotted key, caller value, member value | MISSING)] for the caller's execution leaves whose value differs from the
    member's; absent = [(dotted key, member value)] for the member's execution leaves the caller's yaml does not write (upstream's default applies
    there, not the member's value). Keys the line pins are excluded from both (they are named as overrides / additions)."""
    mine, member = execution_leaves(caller_doc), execution_leaves(member_doc)
    pinned = set(pinned)
    kept = [(k, v, member.get(k, MISSING)) for k, v in mine.items() if k not in pinned and (k not in member or member[k] != v)]
    absent = [(k, v) for k, v in member.items() if k not in pinned and k not in mine]
    return kept, absent


def difference_words(kept, absent) -> List[str]:
    """The naming clauses of member_differences (shared by compose and the row-sharded launcher)."""
    words = []
    if kept:
        words.append("caller keys kept as written that differ from the member: " + ", ".join(f"{k}={_word(v)} (member {_word(m) if m != MISSING else MISSING})" for k, v, m in kept))
    if absent:
        words.append("member keys the caller's yaml does not set take upstream's defaults: " + ", ".join(f"{k} (member {_word(v)})" for k, v in absent))
    return words


def override_words(route: str, overrides, added) -> List[str]:
    """`<route> overrides runner-yaml keys: <dotted.key> (<caller value> -> <line value>); …` and `<route> sets runner-yaml keys the caller's yaml
    leaves unset: …`."""
    words = []
    if overrides:
        words.append(f"{route} overrides runner-yaml keys: " + "; ".join(f"{k} ({_word(old)} -> {_word(new)})" for k, old, new in overrides))
    if added:
        words.append(f"{route} sets runner-yaml keys the caller's yaml leaves unset: " + ", ".join(f"{k}={_word(v)}" for k, v in added))
    return words


def note(words: List[str]) -> None:
    sys.stderr.write(f"[{TAG}] note: " + "; ".join(words) + "\n"); sys.stderr.flush()


def as_given_note(home: str, caller: str, route: str = "off") -> None:
    """One of the tree's own configuration files named under `--mode off` (cli.kit_base: the `default` tier's shipped_predict.yml …) is the configuration the stock caller runs, as given: said once."""
    note([f"runner yaml {_rel(home, caller)} as given under {route} (upstream reads it as written; keys it does not set take upstream's defaults, not the stock member's)"])


def merge_callers(home: str, callers: List[str], out_dir: str) -> str:
    """Several ``--runner-yaml`` (the flag repeated) laid in ARGV ORDER into one caller document — each later yaml on top of the earlier ones
    (deep_overlay: mappings merge, a later leaf wins, named) — written to ``<out_dir>/runner_callers_<n>_<last basename>``; that one document is
    then the call's caller yaml (cli.row_yaml composes it under the mode's configuration like a single --runner-yaml). One path: returned as is."""
    callers = [c for c in callers if c]
    if not callers:
        raise ValueError("merge_callers: no runner yaml given")
    if len(callers) == 1:
        return callers[0]
    import yaml
    doc: dict = {}
    words = [f"runner yamls {', '.join(_rel(home, c) for c in callers)} laid in argv order (a later yaml's keys on top of the earlier ones')"]
    for c in callers:
        doc, overrides, _added = deep_overlay(doc, _load(c))
        if overrides:
            words.append(f"{_rel(home, c)} overrides: " + "; ".join(f"{k} ({_word(old)} -> {_word(new)})" for k, old, new in overrides))
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"{CALLERS_PREFIX}{len(callers)}_{os.path.basename(callers[-1])}")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(f"# {len(callers)} runner yamls laid in argv order ({', '.join(_rel(home, c) for c in callers)}); written by openfold3_opt.runner_yaml.merge_callers\n")
        fh.write(yaml.safe_dump(doc, default_flow_style=False, sort_keys=False))
    note(words + [f"-> {dst}"])
    return dst


def _load(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: a runner yaml is a mapping, got {type(doc).__name__}")
    return doc


def _rel(home: str, path: str) -> str:
    return os.path.relpath(path, home) if path.startswith(os.path.join(home, "")) else path


def _word(v) -> str:
    return "null" if v is None else repr(v)


def compose(home: str, caller: str, member: str, pins: dict, out_dir: Optional[str], route: str) -> Tuple[str, dict]:
    """The caller's yaml composed under the `route` configuration (route = the mode word: fast | exact | big). Returns (path, record): path =
    the caller's own file when the pins change nothing in it (the argv is then byte-identical to an as-given call), else
    `<out_dir>/runner_composed_<basename>` written here; record = {"caller", "member", "overrides", "added", "kept", "absent"} for the run record.
    ONE stderr note names it: `note: runner yaml <caller> composed under the <route> configuration (<member>) ->
    <composed>; <route> overrides runner-yaml keys: <key> (<caller value> -> <line value>); …`, then the keys the mode set that the caller left
    unset and the caller's kept execution keys the member does not write."""
    doc = _load(caller)
    merged, overrides, added = deep_overlay(doc, pins)
    pinned = [k for k, _ in _leaves(pins)] if pins else []
    kept, absent = member_differences(doc, _load(member), pinned)
    record = {"caller": caller, "member": member, "overrides": [[k, old, new] for k, old, new in overrides], "added": [[k, v] for k, v in added],
              "kept": [[k, v, m] for k, v, m in kept], "absent": [[k, v] for k, v in absent]}
    if not overrides and not added:                                      # the caller's yaml already carries every key the mode writes: its own path (byte-identical argv)
        note([f"runner yaml {_rel(home, caller)} composed under the {route} configuration ({_rel(home, member)}): every key the mode writes already as written, the file runs as is"]
             + difference_words(kept, absent))
        return caller, record
    if not out_dir:
        raise ValueError("compose: an output directory is required to write the composed runner yaml")
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, COMPOSED_PREFIX + os.path.basename(caller))
    note([f"runner yaml {_rel(home, caller)} composed under the {route} configuration ({_rel(home, member)}) -> {dst}"]
         + override_words(route, overrides, added) + difference_words(kept, absent))
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(f"# {_rel(home, caller)} composed under the line ({_rel(home, member)}): the line's keys laid on top; written by openfold3_opt.runner_yaml.compose\n")
        fh.write(yaml.safe_dump(merged, default_flow_style=False, sort_keys=False))
    record["composed"] = dst
    return dst, record
