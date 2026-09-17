"""opt_manifest.json — what was active when a pred_bw job's outputs were produced, written beside them (the directory of the -op prefix).

Carries the activation report (mode, route, the kit's resolution, the GPU), the command as run (the arm's argv and the wrapper's), the
kit's identity (root, version, integrity verdict), the deterministic recipe block, the stock arm's environment proof, the driver-cache
source, the exit code and a UTC timestamp. Everything volatile about a run lives here and nowhere else.
"""
import datetime as _dt
import json
import os
import platform as _platform

SCHEMA = "chrombpnet_opt.manifest/1"
FILENAME = "opt_manifest.json"
EXCLUDED_REPORT_KEYS = ()


def build(report, *, command=None, argv=None, arm_argv=None, exit_code=None, det=None, env_proof=None, env_stripped=None, extra=None) -> dict:
    rep = {k: v for k, v in (report or {}).items() if k not in EXCLUDED_REPORT_KEYS}
    return {
        "schema": SCHEMA,
        "written_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": rep.get("mode"),
        "route": rep.get("route"),
        "active": bool(rep.get("active")),
        "reason": rep.get("reason"),
        "levers_applied": list(rep.get("levers_applied") or []),  # the levers the kit's stamp evidences (stack.applied); [] on the stock arm
        "partial": list(rep.get("partial") or []),                # the levers that fell back by the kit's stamp (exit 3)
        "partial_reasons": list(rep.get("partial_reasons") or []),
        "partial_detection": rep.get("partial_detection"),
        "refused": rep.get("refused"),                            # the kit refused the mode by name (a lever of the class's set could not start); exit 3
        "ignored": list(rep.get("ignored") or []),                # kit-internal names found in the caller's environment, removed for the run
        "gated": list(rep.get("gated") or []),                    # the kit's declared rules taken (exit-neutral, by name)
        "incomplete": rep.get("incomplete"),                      # "<ok>/<n>" when the outputs fall short of the request (exit 1), else None
        "det": det,
        "kit": {"root": rep.get("kit"), "version": rep.get("kit_version"), "integrity": rep.get("kit_integrity"), "documented_form": rep.get("documented_form")},
        "gpu": rep.get("gpu"),
        "stack": {"package_version": rep.get("package_version"), "python": _platform.python_version(), "target_gpu": os.environ.get("MODEL_OPT_TARGET_GPU") or None,
                  "notes": list(rep.get("notes") or [])},
        "cache_tar": rep.get("cache_tar"),
        "env_proof": env_proof,
        "env_stripped": env_stripped,
        "command": command,
        "argv": list(argv) if argv is not None else None,
        "arm_argv": list(arm_argv) if arm_argv is not None else None,
        "exit_code": exit_code,
        "chrombpnet_opt_env": os.environ.get("CHROMBPNET_OPT"),
        "activation_report": rep,
        **(extra or {}),
    }


def output_dir(output_prefix: str) -> str:
    d = os.path.dirname(os.path.abspath(output_prefix))
    return d or os.getcwd()


def write(out_dir: str, report, **kw) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FILENAME)
    man = build(report, **kw)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=str, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def read(out_dir_or_path: str) -> dict:
    path = out_dir_or_path if out_dir_or_path.endswith(".json") else os.path.join(out_dir_or_path, FILENAME)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
