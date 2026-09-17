#!/usr/bin/env python
"""big_launch.py <core dir> <pallas patches dir> <fpf launcher.py> --line big --base <mode> [--off <lever,…>] <script.py> [flags...] — the model process of
mode ``big``: installs the FAST base (FlashPairformer imported with its switches OFF, then exact's Pallas levers at the vacated sites),
then the kit's memory levers (big_levers.py) via ``opt_core.mem.apply`` — every lever of the line except those named on ``--off``, applied or refused by
name; the selection is the command line's alone (``switches=`` to opt_core.mem.apply, settings the kit's: nothing of the process
environment selects a lever or a setting) — then, above one GPU,
row-sharded pair-stack tensor parallelism. At exit: a fail-closed gate on the trace census (the BIG census line). Run by the fork's interpreter; never imports af3_jax_opt."""
import atexit
import os
import sys

PREFIX = "[af3-jax-opt]"
TAG = "af3-jax-opt"
INSTALL = {"big": "fast_disengaged"}                                    # line -> the base's installation (the base MODE comes on the command line: modes.BIG_LINES): fast's import with its switches at their OFF values by property + exact's Pallas levers at the vacated sites
ENV_N_GPU = "AF3_JAX_N_GPU"                                                 # the resource axis of the mode (stack.ENV_N_GPU spells the same): 1 = the single-card levers; > 1 adds inprocess/rowpair.py over that many cards
USAGE = "usage: big_launch.py <core dir> <pallas patches dir> <fpf launcher.py> --line big --base <mode> [--off <lever,...>] <script.py> [flags...]"
OFF_ARG = "--off"                                                            # modes.OFF_ARG (a test holds the pair equal): the memory levers the wrapper switches off for a --n_gpu P > 1 composition
                                                                             # (the row-sharded pair stack supersedes them)
NO_OPT_OUT = "a mode is all of its levers: nothing"                          # the census gate's words where a kit's opt-out flag would stand (opt_core.mem.record PARTIAL_REFUSED: `exit 3 (<…> records and
                                                                             # proceeds)`): this kit has none — a lever the core cannot apply refuses the mode by name
_STATE = {"record": None, "exited": False}


def _at_exit() -> None:
    if _STATE["exited"]:
        return
    _STATE["exited"] = True
    rec = _STATE["record"]
    if rec is None:
        return
    import big_levers
    for n in big_levers.exit_notes():
        rec.note(n)
    v = rec.exit_gate(0, expect_units=False)                              # no opt-out exists (NO_OPT_OUT): a partial census is verdict=partial and the refusal line names it — the wrapper exits 3 by name on it (levers short)
    line = rec.exit_line(TAG, v)
    sys.stdout.flush()
    print(f"{PREFIX} BIG census units={v['census'].get('n_units')} applied={','.join(rec.levers) or 'none'} refused={','.join(rec.refused_names) or 'none'} "
          f"traced={','.join(f'{k}={n}' for k, n in big_levers.TRACED.items())} verdict={'partial' if v['partial'] else 'ok'} "
          f"n_gpu={rec.extra.get('n_gpu', 1)}" + (f" sharding={rec.extra['sharding']}" if rec.extra.get("sharding") else ""), flush=True)
    if line:
        print(line, flush=True)


def install_fast(script: str, fpf_launcher: str) -> None:
    """fast's own installation: the template census guard (fpf_launch.install_template_guard, before anything of the model is bound), the
    FlashPairformer package imported as its launcher does (it installs per AF3_FLASHPAIRFORMER / AF3_DIFFUSION_HOIST — modes.mode_env sets
    them at their OFF values on the mode's composition: nothing of the add-on is bound), the tree's SERVED line at exit (fpf_launch._report:
    the served/fallback census; fused=0 on the composition — a disengaged kernel that served is named by the wrapper)."""
    import fpf_launch
    fpf_launch.install_template_guard()
    fpf_launch._install_tree_levers()                                        # the base's in-process levers (DATTN at n_gpu 1; TTR is disengaged by property and DATTN superseded under --n_gpu > 1: their switches are off there), before the add-on import as on fast
    sys.path.insert(0, os.path.dirname(script))
    sys.path.insert(0, os.path.dirname(os.path.abspath(fpf_launcher)))
    import af3_flashpairformer  # noqa: F401
    fpf_launch._install_tree_levers(late=True)                               # levers that rebind the add-on (HOIST_LOGITS), after its import as on fast
    atexit.register(fpf_launch._report)


def install_base(line: str, levers_dir: str, fpf_launcher: str, script: str) -> None:
    """The base's own installation (the FAST base): the mode's composition = fast's import with its switches at their OFF values (the fast
    levers disengaged by property) THEN exact's Pallas levers at the vacated sites (levers_launch.install: GLUT, ATTNCFG)."""
    import levers_launch
    kind = INSTALL.get(line)
    if kind == "fast_disengaged":
        # the wrapper decided the composition (modes.big_env): the hoist is the memory holder (always 0); the FlashPairformer word is `triatt`
        # when the fused triangle attention is kept (one GPU, at or below the kernels' serve edge) or `off` — trimul_chunk owns the other site
        if os.environ.get("AF3_FLASHPAIRFORMER", "off").lower() not in ("off", "triatt") or os.environ.get("AF3_DIFFUSION_HOIST", "0") != "0":
            raise SystemExit(f"{PREFIX} BIG FAILED: the composition needs the hoist off and the trimul site free (AF3_FLASHPAIRFORMER=off|triatt, "
                             f"AF3_DIFFUSION_HOIST=0: the fast levers disengaged by property) — got "
                             f"AF3_FLASHPAIRFORMER={os.environ.get('AF3_FLASHPAIRFORMER')!r} AF3_DIFFUSION_HOIST={os.environ.get('AF3_DIFFUSION_HOIST')!r}")
        install_fast(script, fpf_launcher)
        levers_launch.install(levers_dir, script)
    else:
        raise SystemExit(f"{PREFIX} BIG FAILED: --line {line!r} is not a line ({' | '.join(INSTALL)})")


def parse_argv(argv):
    """(core_dir, levers_dir, fpf_launcher, line, base, off, script, flags) from the command line; SystemExit(usage) on anything else.
    ``off``: the lever names of ``--off a,b,…`` (the memory levers the wrapper switches off for a --n_gpu P > 1 composition)."""
    if len(argv) < 9 or argv[4] != "--line" or argv[6] != "--base":
        raise SystemExit(USAGE)
    if argv[5] not in INSTALL:
        raise SystemExit(f"{PREFIX} BIG FAILED: --line {argv[5]!r} is not a line of this kit ({'|'.join(INSTALL)}) (rc 2)")
    i, off = 8, ()
    if i < len(argv) and argv[i] == OFF_ARG:
        if i + 1 >= len(argv):
            raise SystemExit(USAGE)
        off = tuple(x for x in argv[i + 1].split(",") if x)
        i += 2
    if i >= len(argv) or not argv[i].endswith(".py"):
        raise SystemExit(USAGE)
    core_dir, levers_dir, fpf_launcher = (os.path.abspath(x) for x in argv[1:4])
    return core_dir, levers_dir, fpf_launcher, argv[5], argv[7], off, os.path.abspath(argv[i]), list(argv[i + 1:])


def lever_switches(off, registry) -> dict:
    """The explicit switch mapping handed to the core (opt_core.mem.apply ``switches=``): each lever named on --off at OFF (``{name: False}``);
    a name that is no registered lever exits by name (rc 2) before anything is applied."""
    known = set(registry.names())
    unknown = [x for x in off if x not in known]
    if unknown:
        raise SystemExit(f"{PREFIX} BIG FAILED: --off names no registered lever: {','.join(unknown)} (registered: {','.join(sorted(known))}) (rc 2)")
    return {name: False for name in off}


def n_gpu_requested(environ=None) -> int:
    """``AF3_JAX_N_GPU`` as an int >= 1 (default 1); anything else exits by name (rc 2) — the wrapper validates and counts the visible cards
    before the launch (stack.py), this is the model process's own reading of the same variable."""
    environ = os.environ if environ is None else environ
    raw = environ.get(ENV_N_GPU, "1")
    try:
        n = int(str(raw).strip())
        if n < 1:
            raise ValueError
    except ValueError:
        raise SystemExit(f"{PREFIX} BIG FAILED: {ENV_N_GPU}={raw!r} is not an integer >= 1 (rc 2)")
    return n


def main(argv):
    core_dir, levers_dir, fpf_launcher, line, base, off, script, flags = parse_argv(argv)
    for p in (os.path.join(core_dir, "opt_core", "mem", "registry.py"), fpf_launcher, script):          # the levers dir is checked by levers_launch.install
        if not os.path.isfile(p):
            sys.exit(f"{PREFIX} BIG FAILED: no such file {p}")
    n_gpu = n_gpu_requested()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(script))
    sys.path.insert(0, here)
    sys.path.insert(0, core_dir)
    import fpf_launch
    fpf_launch.install_cache_key()                                            # the persistent-cache key portability rebinding, before the base installs anything
    install_base(line, levers_dir, fpf_launcher, script)
    from opt_core import mem
    from opt_core.mem import registry
    from opt_core.mem.jax_mem import xla_env_record
    import big_levers
    import levers_launch
    ctx = registry.Ctx(prefix="AF3_JAX", tag=TAG, framework="jax", hooks=big_levers.HOOKS, settings=big_levels_settings(big_levers),
                       environ=os.environ, graphs=False, opt_out=NO_OPT_OUT,               # environ: the process environment the core's allocator/XLA levers act on — never read for the selection or a setting
                       extra={"line": line, "base": base, "script": os.path.basename(script), "script_sha256": levers_launch.sha256(script),
                              "xla_env": xla_env_record(os.environ), "levers_module_sha256": levers_launch.sha256(big_levers.__file__)})
    rec = mem.apply(big_levers.LINE_LEVERS[line], ctx, base=base, strict=False, switches=lever_switches(off, registry),
                    allow_partial=False)                                      # the selection = the line minus --off, settings the kit's (ctx.settings); a lever the core cannot apply refuses the
                                                                              # mode by name (the census gate, exit 3) — a mode is all of its levers, nothing opts out
    _STATE.update(record=rec)
    atexit.register(_at_exit)
    tp = {"n_gpu": n_gpu}
    rowpair = None
    if n_gpu > 1:                                                             # the resource axis: row-sharded pair-stack tensor parallelism over n_gpu cards (inprocess/rowpair.py) — installed
        import fpf_launch                                                     # after the base and the memory levers so it rebinds the FINAL classes under its mesh; at 1 nothing of it is imported
        rowpair = fpf_launch.load_inprocess("rowpair")
        rowpair.install(n_gpu, b21=big_levers)                             # mesh + the shared recipe (b21 = this kit's pair-conditioning transcription); refusals exit 5 by name
        sys.modules["af3_jax_opt.inprocess.templates"].set_form("row_born")  # the installed census guard's word (install_big_base loaded it): under the row-sharded pair stack each device forms the template pair features for its row block only
        tp["sharding"] = "rowpair"
    rec.extra.update(tp)
    print(rec.active_line(TAG, line=line, **tp), flush=True)
    if rowpair is not None:                                                   # n_gpu > 1: the adapter loads the script as a module, binds its ModelRunner to the mesh and runs its main (absl)
        rowpair.run_script(script, flags, n_gpu, b21=big_levers)
        return
    import runpy
    sys.argv = [script] + flags
    runpy.run_path(script, run_name="__main__")


def big_levels_settings(mod):
    return {k: dict(v) for k, v in mod.SETTINGS.items()}


if __name__ == "__main__":
    main(sys.argv)
