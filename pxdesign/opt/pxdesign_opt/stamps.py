"""The kit's line writer and its one configuration-evidence line, KERNELS.

Every line the kit prints is ``<PREFIX> <TAG> key=value …`` on stderr — the TAG word after the caller's prefix (``[pxdesign-opt]`` on the kit
routes, ``[pxdesign-opt stock]`` on the stock caller), then whitespace-separated ``key=value`` tokens whose values hold no blank (`kv`).
`emit` is the one writer of those lines on every route: each starts on a fresh line (`FRESH`), so a reader that splits the stream on ``\n``
finds every line at column 0 even after a ``\r``-terminated progress fragment of upstream's.

    KERNELS    once per pass, once the runner is built (kit modes and the stock caller's in-process route: ``source=process``) or before the
               console script is exec'd (the stock caller's cli route: ``source=caller_process`` — the child is the same interpreter and the
               same, proven environment): torch / CUDA versions, the TF32 switches as torch reads them back, ``PYTORCH_CUDA_ALLOC_CONF``, the
               LayerNorm kind — ``fused`` = Protenix's CUDA LayerNorm (`FUSED_CLASS`), ``plain`` = `PLAIN_CLASS` — read from the BUILT model's first
               LayerNorm leaf (`layernorm_of_model`) or, on the cli route, from the environment Protenix will read at its import
               (`layernorm_of_env`, ``primitives.py:49-51``) — deepspeed's version, the ``use_deepspeed_evo_attention`` flag as resolved, and
               whether flash_attn / cuequivariance / xformers are importable (find_spec, never imported).

Timing and memory are not printed per item: the DONE line carries the job's designs, s/design and load / wall seconds, and `opt_manifest.json`
the same timings. This module imports nothing of the core, torch or upstream at module level (the stock caller imports it before its
environment proof); the probes import torch inside the function that needs it.
"""
import importlib.util
import os
import re
import sys
from typing import Dict, Optional, Tuple

FUSED_CLASS = "protenix.model.layer_norm.layer_norm.FusedLayerNorm"            # Protenix's fused CUDA LayerNorm (protenix/model/layer_norm/layer_norm.py:145)
PLAIN_CLASS = "protenix.openfold_local.model.primitives.OpenFoldLayerNorm"      # OpenFold's LayerNorm (protenix/openfold_local/model/primitives.py:232)
LAYERNORM_ENV = "LAYERNORM_TYPE"
FAST_LAYERNORM = "fast_layernorm"                                               # the value primitives.py:49 compares against; anything else selects PLAIN_CLASS
ABSENT, UNSET, NA, NONE = "absent", "unset", "n/a", "none"
KERNELS = "KERNELS"
TAGS = (KERNELS,)
FRESH = "\n"                                                                     # every line the kit prints starts on a fresh line: a progress bar's "\r"-terminated fragment left on the
                                                                                 # same stream cannot glue onto it (a line-anchored reader would lose the glued line); the empty line is harmless


def emit(text: str, stream=None) -> str:
    """Write ``text`` as one line that starts at column 0 whatever the stream's last byte was: a newline first, then the line, flushed (stderr by
    default). The one writer of the kit's own lines on every route (report.log*, the stock caller's log). Returns ``text``."""
    s = sys.stderr if stream is None else stream
    s.write(f"{FRESH}{text}\n")
    s.flush()
    return text


def _value(v) -> str:
    if v is None:
        return NONE
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (list, tuple, set, frozenset)):
        s = ",".join(_value(x) for x in v)
    else:
        s = str(v)
    s = "_".join(s.split())                                                     # a value holds no blank: the line is split on whitespace
    return s if s else NONE


def kv(**fields) -> str:
    """``k=v k2=v2`` in keyword order — the tokens after the TAG: booleans ``True``/``False``, ``None``/empty → ``none``, sequences comma-joined,
    blanks inside a value replaced by ``_``. The same ``key=value`` grammar as the core's line primitives, kept import-free for the stock caller."""
    return " ".join(f"{k}={_value(v)}" for k, v in fields.items())


def body(tag: str, **fields) -> str:
    """``<TAG> k=v …`` — the text a caller prints behind its own prefix (``log(body(...))``)."""
    text = kv(**fields)
    return f"{tag} {text}" if text else tag


# --------------------------------------------------------------------------------------------------------------------- formatters


def kernels(facts: Dict[str, object]) -> str:
    return body(KERNELS, **facts)


def batch_facts(n_sample: int, chunk, n_tasks: int, n_seeds: int) -> Dict[str, object]:
    """The job's batch facts, recorded in `opt_manifest.json` / `stock_env_proof.json` under ``stamps.batch`` (no line): ``chunk`` an int or None
    (upstream: a null chunk size runs all samples in one batch), ``batch`` = min(n_sample, chunk), ``expected_designs`` = tasks × seeds × n_sample."""
    c = None if chunk in (None, "", NONE) else int(chunk)
    n = int(n_sample)
    return {"n_sample": n, "chunk": c, "batch": min(n, c) if c else n, "tasks": int(n_tasks), "seeds": int(n_seeds), "expected_designs": int(n_tasks) * int(n_seeds) * n}


# ------------------------------------------------------------------------------------------------------------------------ grammar

_P = r"^\[(?P<tag>pxdesign-opt(?: stock)?)\] "                                  # the two prefixes: report.PREFIX and the stock caller's
KERNELS_RE = re.compile(_P + r"KERNELS torch=(?P<torch>\S+) cuda=(?P<cuda>\S+) tf32_matmul=(?P<tf32_matmul>True|False|n/a) tf32_cudnn=(?P<tf32_cudnn>True|False|n/a) "
                        r"alloc_conf=(?P<alloc_conf>\S+) layernorm=(?P<layernorm>fused|plain) deepspeed=(?P<deepspeed>\S+) ds4sci_evo_attention=(?P<ds4sci_evo_attention>on|off) "
                        r"flash_attn=(?P<flash_attn>yes|no) cuequivariance=(?P<cuequivariance>yes|no) xformers=(?P<xformers>yes|no) compile=(?P<compile>\S+) source=(?P<source>process|caller_process|metadata)$")
RES = {KERNELS: KERNELS_RE}

EXAMPLES = {                                                                    # the line exactly as printed (the kit prefix; the stock caller's differs in the prefix and source= only)
    KERNELS: "[pxdesign-opt] " + kernels({"torch": "2.3.1+cu121", "cuda": "12.1", "tf32_matmul": False, "tf32_cudnn": True, "alloc_conf": UNSET, "layernorm": "fused",
                                          "deepspeed": "0.15.4", "ds4sci_evo_attention": "off", "flash_attn": "no", "cuequivariance": "no", "xformers": "no", "compile": "off",
                                          "source": "process"}),
}


def tokens(line: str, tag: str):
    """The ``[(key, value)]`` after ``tag`` on ``line`` (``value`` None for a token without ``=``), or None when the TAG word is absent — the downstream
    log reducer's tokenizer rule (a TAG at line start or after whitespace / ``]``, then whitespace-separated tokens)."""
    m = re.search(r"(?:^|[\s\]])" + re.escape(tag) + r"(?=\s|$)", line)
    if not m:
        return None
    return [tuple(t.split("=", 1)) if "=" in t else (t, None) for t in line[m.end():].split()]


# ------------------------------------------------------------------------------------------------------------------------- probes

def qualname(cls) -> str:
    return f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__qualname__', getattr(cls, '__name__', '?'))}"


def kind_of_class(name: str) -> str:
    """``fused`` for Protenix's fused CUDA LayerNorm class, ``plain`` for any other (OpenFold's on the stock stack)."""
    return "fused" if name == FUSED_CLASS else "plain"


def layernorm_of_model(model) -> Tuple[str, str]:
    """(kind, module.qualname) of the LayerNorm class the BUILT model normalises with: the first leaf module (one without submodules) whose class
    name contains ``LayerNorm`` — Protenix's wrappers that hold a LayerNorm (``AdaptiveLayerNorm``) are containers, not the class in use; (``plain``,
    ``none``) when the model holds no such module."""
    first_container = None
    for m in model.modules():
        if "LayerNorm" not in type(m).__name__:
            continue
        if next(iter(m.children()), None) is None:                             # a leaf: the normalisation itself (FusedLayerNorm | OpenFoldLayerNorm)
            name = qualname(type(m))
            return kind_of_class(name), name
        first_container = first_container or qualname(type(m))
    return "plain", first_container or NONE


def layernorm_of_env(environ=None) -> Tuple[str, str, Optional[str]]:
    """(kind, module.qualname, value-or-None) the environment selects at Protenix's import (primitives.py:49-51: ``fast_layernorm`` → the fused class)."""
    environ = os.environ if environ is None else environ
    v = environ.get(LAYERNORM_ENV) or None                       # an empty value selects OpenFold's exactly as an absent one does; printed as absent
    cls = FUSED_CLASS if v == FAST_LAYERNORM else PLAIN_CLASS
    return kind_of_class(cls), cls, v


def _importable(name: str) -> str:
    try:
        return "yes" if importlib.util.find_spec(name) is not None else "no"
    except (ImportError, ValueError):
        return "no"


def _dist_version(name: str) -> Optional[str]:
    from importlib import metadata
    try:
        return metadata.version(name)
    except (metadata.PackageNotFoundError, ValueError):
        return None


def kernels_facts(*, layernorm_kind: str, ds4sci: bool, source: str, environ=None) -> Dict[str, object]:
    """The KERNELS fields in line order. ``source=process``/``caller_process`` read the live torch module (imported here if not yet): ``tf32_matmul``
    and ``tf32_cudnn`` are the readbacks ``torch.backends.cuda.matmul.allow_tf32`` / ``torch.backends.cudnn.allow_tf32`` taken in this process after
    the import (torch folds `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` into the first at its first query); when torch is not importable the versions come from the
    distribution metadata and ``source=metadata`` (flags ``n/a``)."""
    environ = os.environ if environ is None else environ
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        tv, cu = torch.__version__, getattr(torch.version, "cuda", None) or NONE
        tf32_mm, tf32_cudnn = bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32)
    else:
        tv, cu, tf32_mm, tf32_cudnn, source = _dist_version("torch") or ABSENT, NONE, NA, NA, "metadata"
    return {"torch": tv, "cuda": cu, "tf32_matmul": tf32_mm, "tf32_cudnn": tf32_cudnn, "alloc_conf": environ.get("PYTORCH_CUDA_ALLOC_CONF") or UNSET,
            "layernorm": layernorm_kind, "deepspeed": _dist_version("deepspeed") or ABSENT, "ds4sci_evo_attention": "on" if ds4sci else "off",
            "flash_attn": _importable("flash_attn"), "cuequivariance": _importable("cuequivariance_torch"), "xformers": _importable("xformers"),
            "compile": "off", "source": source}


TRUE_WORDS, FALSE_WORDS = ("true", "t", "yes", "y", "1"), ("false", "f", "no", "n", "0")   # upstream's boolean option words (get_bool_value, protenix/config/extend_types.py:48-55) — the one table (options.py renders them)


def truthy(v) -> bool:
    """upstream's boolean option words (`TRUE_WORDS`) and Python booleans."""
    if isinstance(v, str):
        return v.strip().lower() in TRUE_WORDS
    return bool(v)


def option_value(argv, name: str):
    """The value following ``--<name>`` (or ``--<dotted.alias>``) in an upstream argument list, last occurrence wins; None when absent."""
    val = None
    keys = {f"--{name}", f"--infer_setting.{name}"} if name == "sample_diffusion_chunk_size" else {f"--{name}"}
    for i, a in enumerate(argv or ()):
        if a in keys and i + 1 < len(argv):
            val = argv[i + 1]
        elif any(a.startswith(k + "=") for k in keys):
            val = a.split("=", 1)[1]
    return val
