"""The command line: ``complexa-opt design|check ...`` (``python -m complexa_opt ...``; ``run.sh`` is a thin wrapper).

    design --out <dir> [--mode off|exact|fast|big] [--input <entry.json|yaml>] [<Hydra overrides>]
           [-- <upstream's own `complexa generate` arguments, verbatim: ++generation.dataloader.dataset.nres.nsamples=<n> ++seed=<s>
               ++generation.dataloader.batch_size=<B> … any Hydra override, --job-id N>]
    (Hydra overrides in upstream's grammar — `key=value`, `++key=value`, `~key` — may also stand among the flags, as `complexa generate`
    takes them positionally: they join the tail after `--` in order.)
    check  [--mode M]      dry run: pins, weights, checkout, GPU vs target, the mode's route and levers, the hook — one DRY-RUN line

Everything after ``--`` reaches upstream unchanged and after the tokens the package composes, so upstream — not this command line — answers
for it (a key the package also spells is simply given twice; Hydra applies the later, the caller's). Exit codes (``opt_core.report``): 0 ok ·
1 failed or incomplete (the child failed, designs written short) · 2 usage (an unknown option or mode; an entry file that is not one
item — each ONE named ``USAGE refused:`` line) · 3 not active (the core pin gate; pins, checkout or weights not as pinned; a stock process
that is NOT STOCK; on the kit route the hook absent from upstream's interpreter, a lever that could not be installed — the generation
process refuses the whole mode by name — or a lever that never engaged). The mode: ``--mode``, else ``COMPLEXA_OPT``, else the default
``fast``.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import ActivationError, modes, report, settings as _settings, stack


class _Parser(argparse.ArgumentParser):
    """argparse whose errors are this package's one named usage line (exit 2), not argparse's two-line usage dump."""

    def error(self, message):
        report.emit(report.usage_refused(message))
        raise SystemExit(report.EXIT_USAGE)


def _parser() -> argparse.ArgumentParser:
    p = _Parser(prog="complexa-opt", description="Proteina-Complexa binder generation: mode off (stock) | exact | fast | big (the kit's lever sets inside upstream's own command).")
    sub = p.add_subparsers(dest="command", parser_class=_Parser)
    d = sub.add_parser("design", help="one generation run")
    d.add_argument("--mode", default=None, help=f"{'|'.join(modes.MODES)} (else ${modes.ENV_MODE}, else {modes.DEFAULT_MODE}); {modes.OFF} runs stock; " + "; ".join(f"{m}: {','.join(modes.KIT_MODES[m])}" for m in modes.KIT_MODES))
    d.add_argument("--input", default=None, help="optional target entry file: one item in upstream's targets_dict form (JSON or YAML); a relative target_path inside it is resolved against this file's directory. Without it the target is the overrides' (++generation.task_name=…) or the configuration's own")
    d.add_argument("--out", required=True, help="the run directory (the child's working directory): inference/<config>_<item>_<run>/job_*/*.pdb, logs/, opt_manifest.json, design.log, stock_env_proof.json (off) or kit_records/ (exact|fast|big)")
    d.add_argument("overrides", nargs="*", help=f"upstream's own `complexa generate` arguments after `--`, verbatim and last, e.g. -- "
                                                  f"++run_name=r1 ++{_settings.NSAMPLES_KEY}=32 ++seed=5 ++generation.dataloader.batch_size=32 (shipped: nsamples 4, seed 5, batch 16)")
    c = sub.add_parser("check", help="dry run: the mode's route, pins, checkout, weights, GPU")
    c.add_argument("--mode", default=None)
    return p


def _resolve_mode(value: Optional[str]) -> str:
    """exit 2 for an unknown or conflicting mode — raised as SystemExit after the line."""
    try:
        return modes.resolve(value)
    except modes.ModeError as e:
        report.emit(report.usage_refused(str(e)))
        raise SystemExit(report.EXIT_USAGE)


def cmd_check(a) -> int:
    mode = _resolve_mode(a.mode)
    route = modes.route_of(mode)
    cp = stack.check_pins()
    pins = stack.pins()
    reasons: List[str] = []
    bad, detail = cp.check_upstream(pins)
    reasons += bad
    cbad, cdetail = cp.check_configs(pins, detail.get("checkout"))
    reasons += cbad
    try:
        w = stack.weights_gate()
        wlabel = "pinned-bytes" if w["pinned"] else "UNPINNED"
        reasons += w["bad"]
        report.emit(w["line"])
    except ActivationError as e:
        w, wlabel = None, "none"
        reasons.append(str(e))
    try:
        cs = stack.console_script()["path"]
    except ActivationError as e:
        cs = "none"
        reasons.append(str(e))
    g = stack.gpu()
    tg = stack.target_gate(g)
    from opt_core.report import gpu_label
    fields = dict(mode=mode, route=route, upstream=(f"{detail.get('version')}@{(detail.get('commit') or 'nogit')[:8]}" if detail.get("version") else "none"),
                  dirty=(detail.get("dirty") if detail.get("dirty") is not None else "none"), complexa=cs, weights=wlabel, gpu=gpu_label(g),
                  target=(tg.details or {}).get("target") or "none", match=("none" if not (tg.details or {}).get("target") else ("yes" if tg.ok else "no")))
    if route == "kit":                                                                   # the hook must be live in upstream's interpreter, else the levers cannot reach the generation process
        from . import stock_design
        env, _removed = stock_design.clean_env(export={modes.ENV_MODE: mode}, keep_kit_paths=True)
        probe = stack.hook_probe(env)
        fields.update(levers=",".join(modes.levers_of(mode)), hook=("armed" if probe.get("finder_armed") else ("loaded" if probe.get("autoload_loaded") else "missing")),
                      pth=probe.get("pth") or "none")
        if not probe.get("finder_armed"):
            reasons.append(f"hook_missing: {stack.PTH_FILE} did not arm complexa_opt._autoload in {probe.get('python')} (run.sh install)")
    ok = not reasons
    fields["ok"] = ok
    if reasons:
        fields["reason"] = "; ".join(reasons)
    report.emit(report.dry_run_line(**fields))
    if not ok:
        report.emit(report.refused("pins not met: " + "; ".join(reasons), mode))
        return report.EXIT_NOT_ACTIVE
    return report.EXIT_OK


def cmd_design(a) -> int:
    mode = _resolve_mode(a.mode)
    from . import inputs as _inputs, kit_design, stock_design
    route = kit_design if modes.route_of(mode) == "kit" else stock_design
    try:
        rec = route.run(mode=mode, out_dir=a.out, input_path=a.input, overrides=a.overrides)
    except _inputs.InputError as e:
        report.emit(report.usage_refused(str(e)))
        return report.EXIT_USAGE
    except (ActivationError, stock_design.StockError) as e:
        report.emit(report.refused(str(e), mode))
        return report.EXIT_NOT_ACTIVE
    return int(rec["exit_code"])


def _is_override(tok: str) -> bool:
    """A Hydra override in upstream's own grammar (``key=value``, ``+key=…``, ``++key=…``, ``~key[=…]``) — what ``complexa generate <config>``
    takes positionally. Given among this command line's flags it joins the tail after ``--`` in order (upstream's surface, unchanged)."""
    return bool(tok) and not tok.startswith("-") and (tok.startswith(("+", "~")) or "=" in tok)


def split_overrides(argv: List[str]) -> List[str]:
    """``argv`` with every Hydra override token found before ``--`` moved, in order, to just after it (a ``--`` is added when none was given)."""
    head = argv[:argv.index("--")] if "--" in argv else list(argv)
    tail = argv[argv.index("--") + 1:] if "--" in argv else []
    moved = [t for t in head if _is_override(t)]
    if not moved:
        return list(argv)
    return [t for t in head if not _is_override(t)] + ["--"] + moved + tail


def main(argv: Optional[List[str]] = None) -> int:
    argv = split_overrides(list(sys.argv[1:] if argv is None else argv))
    p = _parser()
    try:
        a = p.parse_args(argv)
    except SystemExit as e:                                   # _Parser.error printed the named line; --help exits 0
        return int(e.code) if isinstance(e.code, int) else report.EXIT_USAGE
    if a.command == "design":
        return cmd_design(a)
    if a.command == "check":
        return cmd_check(a)
    report.emit(report.usage_refused("no command: complexa-opt design|check ..."))
    return report.EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
