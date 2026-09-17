"""The mode line — the one line a caller greps, first on stdout, flushed, one per process before the job:

    [chrombpnet-opt] ACTIVE mode=<mode> route=<k1|tf_function|keras_predict_fileorder> kit=<version> gpu=<class> det=<0|1> precision=<tf32|fp32> [multi=<N>]
    [chrombpnet-opt] NOT ACTIVE mode=off (stock)
    [chrombpnet-opt] NOT ACTIVE mode=<mode> reason=<...>

and the accounting lines on stderr (one formatter for every entry point; `warm` prints them through its pred_bw child):

    [chrombpnet-opt] IGNORED names=<N1,N2,…> reason=<...>     at activation: kit-internal names found in the caller's environment,
                                                               removed from the run (the mode is the whole composition) — never a refusal
    [chrombpnet-opt] GATED <lever>: <reason>                   after the job: a lever the kit's tables keep off on this class / mode, by name
    [chrombpnet-opt] NOT ACTIVE mode=<mode> reason=the kit refused by name: <…>; exit 3
                                                               after the child: a lever of the class's set could not start (the K1 stack or
                                                               kernels, the component models) — the kit refused the mode, ran nothing
    [chrombpnet-opt] NOT ACTIVE: partial activation — <detail>; exit 3 (--mode off runs stock)
                                                               the backstop (partial_line): the record shows a forward other than the route

`route` is the kit's route BY ITS OWN TABLES for this class and mode (stack.table_route: the kit's arch_tiles.json / class table through
the kit's own functions — no torch probe in the wrapper); the kit's own line states the forward it actually ran and why. `check` (the dry run) prints the kit's full resolve()
instead, probe included. `kit` is the kit's __version__ read from its file; `gpu` is the kit's own class word from
fastdefault.detect_gpu() (H100|H200|A100|L40S|B200|unknown), `none` when nvidia-smi is unavailable. The exit tally goes to stderr.
"""
import os
import sys

PREFIX = "[chrombpnet-opt]"
OFF_LINE = PREFIX + " NOT ACTIVE mode=off (stock)"


def gpu_word(gpu) -> str:
    g = gpu or {}
    cls = g.get("class")
    if cls and cls != "unknown":
        return cls
    return "none" if not g.get("compute_cap") and not g.get("available", True) else "unknown"


def mode_line(rep: dict) -> str:
    mode = rep.get("mode")
    if mode == "off" and not rep.get("reason"):
        return OFF_LINE
    if not rep.get("active"):
        return "{} NOT ACTIVE mode={} reason={}".format(PREFIX, mode, rep.get("reason"))
    line = "{} ACTIVE mode={} route={} kit={} gpu={} det={} precision={}".format(PREFIX, mode, rep.get("route"), rep.get("kit_version"), gpu_word(rep.get("gpu")), 1 if rep.get("det") else 0, "fp32" if rep.get("det") else "tf32")   # the conv arithmetic follows --det: fp32 under the recipe (bitwise equal to stock's deterministic run), TF32 at shipped numerics — stock's own default precision class (its cuDNN convs run TF32 on these cards); the K1 route uses the tensor cores
    if rep.get("multi"):
        line += " multi={}".format(int(rep["multi"]))            # `pred_bw --items <file>`: N items in one process
    return line


def print_mode_line(rep: dict, stream=None) -> str:
    line = mode_line(rep)
    out = stream or sys.stdout
    out.write(line + "\n")
    out.flush()
    return line


def ignored_line(names, reason: str) -> str:
    """Kit-internal names found in the caller's environment: removed for the run and named (never a refusal)."""
    return "{} IGNORED names={} reason={}".format(PREFIX, ",".join(names), reason)


def print_ignored(rep: dict, stream=None):
    names = list(rep.get("ignored") or [])
    if not names:
        return None
    line = ignored_line(names, "kit-internal names set in the environment are removed for the run (the mode is the whole composition)")
    print(line, file=stream or sys.stderr, flush=True)
    return line


def partial_line(detail: str, exit_code: int = 3) -> str:
    """The partial verdict line (the family grammar, byte-literal outside `<detail>` and the exit code): a lever of the mode could not
    engage on this machine, so the run does not read as the mode; `--mode off` (stock) is the way to run without the kit's levers."""
    return "{} NOT ACTIVE: partial activation — {}; exit {} (--mode off runs stock)".format(PREFIX, detail, exit_code)


def print_partial_line(detail: str, exit_code: int = 3, stream=None) -> str:
    line = partial_line(detail, exit_code)
    out = stream or sys.stdout
    out.write(line + "\n")
    out.flush()
    return line



def exit_line(mode: str, **fields) -> str:
    body = " ".join("{}={}".format(k, v) for k, v in fields.items())
    return "{} EXIT pid={} mode={} {}".format(PREFIX, os.getpid(), mode, body).rstrip()


def print_exit(mode: str, stream=None, **fields) -> str:
    line = exit_line(mode, **fields)
    print(line, file=stream or sys.stderr, flush=True)
    return line
