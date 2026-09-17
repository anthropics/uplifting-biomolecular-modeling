"""The stock command (`off`) and its proof; the logged launch every route uses. `off` = the fork's own command line at its defaults — the stock
``run_alphafold.py`` with the kit's COMMON-row input flags and the caller's input/output flags, no launcher, no second script, no kit-row
flag. The model process runs in a clean environment: every variable of the must-be-absent prefixes is stripped (the package's own switches
and the add-ons' lever switches) and the image's own XLA variables are set. The proof — printed before launch, the
PROOF line — is that this stripped-prefix scan is empty and argv carries no kit-row flags: nothing from the add-ons is on the path.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

from opt_core import process as _process                                # the model-process runner (transcript pump, process-group timeout)
from . import modes as _modes, settings as _settings, variants as _variants
from .stack import pins, repo_dir, venv_python


def caller_cache_dir(user_args: Optional[List[str]]) -> Optional[str]:
    """The ``--cache_dir`` the caller states (the last one, as absl reads it), or None."""
    return effective_flag(list(user_args or []), _settings.CACHE_FLAG_NAME)[0]


def cache_flags(script: str, cache_dir: Optional[str], user_args: Optional[List[str]] = None) -> List[str]:
    """The cache class the package composes: the kit script always ``--cache_dir=<class>`` (its levers' class: stack.cache_dir); the stock
    script the EMPTY class ``--cache_dir=`` (settings.EMPTY_CACHE_FLAG) unless the caller names a ``--cache_dir`` of their own, which is then
    the only one on the line (it rides in the caller's flags)."""
    if script != _modes.STOCK_SCRIPT:
        return [f"--{_settings.CACHE_FLAG_NAME}={cache_dir}"]
    return [] if caller_cache_dir(user_args) is not None else [_settings.EMPTY_CACHE_FLAG]


def compose(variant: str, cache_dir: Optional[str], user_args: Optional[List[str]] = None,
            script: str = _modes.STOCK_SCRIPT, mode_flags: Optional[List[str]] = None, launcher: Optional[List[str]] = None,
            py: Optional[str] = None, model_dir: Optional[str] = None) -> List[str]:
    """argv of a model process: interpreter, [launcher prefix,] script (in the repo dir), COMMON, [the cache class,] the variant's weights
    flags, the mode row, then the caller's flags verbatim (the stock command line's own knobs, absl's last-wins order)."""
    argv = [py or venv_python(), *(launcher or []), os.path.join(repo_dir(), script), *_modes.common_flags(), *cache_flags(script, cache_dir, user_args),
            *_variants.model_flags(variant, model_dir=model_dir), *(mode_flags or []), *(user_args or [])]
    return argv



def effective_flag(argv: List[str], name: str) -> Tuple[Optional[str], int]:
    """(value, occurrences) of ``--<name>`` on a COMPOSED argv, parsed the way absl parses it: the LAST ``--<name>=<v>`` / ``--<name> <v>`` on the
    line wins (a caller's flag comes after the composed ones in compose, so it wins); (None, 0) when the flag is not on the line."""
    flag = f"--{name}"
    vals = []
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            vals.append(argv[i + 1])
        elif tok.startswith(flag + "="):
            vals.append(tok.split("=", 1)[1])
    return (vals[-1] if vals else None), len(vals)


def effective_buckets(argv: List[str], tree: Optional[str] = None) -> dict:
    """The ``--buckets`` the model process will run with (effective_flag): none on the line = the fork's default list as shipped
    (settings.upstream_buckets). ``{"buckets": [int, ...], "source": "caller" | "mode" | "fork_default", "occurrences": n}`` — ``mode`` = the
    last occurrence equals the mode row's own flag (modes.pad_flags), ``caller`` otherwise."""
    val, n = effective_flag(argv, _settings.BUCKETS_FLAG_NAME)
    if val is None:
        return {"buckets": list(_settings.upstream_buckets(tree)), "source": "fork_default", "occurrences": 0}
    last = [int(t) for t in val.split(",") if t.strip()]
    mode_lists = {tuple(int(t) for t in f.split("=", 1)[1].split(",")) for md in _modes.PAD_POLICY for f in _modes.pad_flags(md, tree)}
    return {"buckets": last, "source": "mode" if n == 1 and tuple(last) in mode_lists else "caller", "occurrences": n}


def effective_flash_impl(argv: List[str], tree: Optional[str] = None) -> dict:
    """The ``--flash_attention_implementation`` the model process is asked for (effective_flag): none on the line = the script's own default,
    read from the stock source (settings.upstream_defaults: ``triton``). ``{"impl": str, "source": "line" | "fork_default", "occurrences": n}``.
    The request only: what the process RESOLVES (the fork's CPU auto-downgrade can rewrite it) is the KERNELS probe's reading (kernels.py)."""
    val, n = effective_flag(argv, _settings.FLASH_FLAG_NAME)
    if val is None:
        return {"impl": _settings.upstream_defaults(tree, (_settings.FLASH_FLAG_NAME,))[_settings.FLASH_FLAG_NAME], "source": "fork_default", "occurrences": 0}
    return {"impl": val, "source": "line", "occurrences": n}


def buckets_line(eff: dict) -> str:
    """`[af3-jax-opt] BUCKETS buckets=<csv> source=caller|mode|fork_default occurrences=n` — the effective padding buckets of the composed
    command (effective_buckets), printed after composition on pred and check."""
    from .report import line
    return line("BUCKETS", buckets=",".join(str(b) for b in eff["buckets"]), source=eff["source"], occurrences=eff["occurrences"])


def proof(env: dict, argv: List[str]) -> dict:
    prefixes = tuple(pins().get("stock_proof", {}).get("must_be_absent_prefixes", ["AF3_JAX_"]))
    present = sorted(k for k in env if k.startswith(prefixes))
    script = os.path.basename(argv[1]) if len(argv) > 1 else None
    exact_flags = _modes.exact_only(argv)
    ok = not present and script == _modes.STOCK_SCRIPT and not exact_flags
    return {"prefixes": list(prefixes), "env_present": present, "script": script, "exact_only_flags": exact_flags, "ok": ok}


def proof_line(p: dict) -> str:
    from .report import line
    return line("STOCK", script=p["script"], env_prefixes_absent=",".join(p["prefixes"]), env_present=p["env_present"] or "none",
                exact_only_flags=p["exact_only_flags"] or "none", proof="ok" if p["ok"] else "FAILED")


def run_logged(argv: List[str], env: dict, cwd: str, on_line: Callable[[str], None]) -> int:
    """Run argv (the model process) to its end, streaming every output line to on_line; its exit code. The core's runner
    (opt_core.process.run_logged): the process leads its own session, so its process group (the kit script's featurisation workers) ends with it."""
    return _process.run_logged(list(argv), timeout_s=None, on_line=on_line, env=env, cwd=cwd).rc
