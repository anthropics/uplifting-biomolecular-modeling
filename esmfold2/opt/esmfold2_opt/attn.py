"""The fast-environment reader: which kernels ESMFold2 runs in THIS interpreter — the atom transformers' attention and the ESMC language model's
MLP / attention / rotary paths — read at run time from upstream's own module switches and from the classes and forwards actually bound on the
loaded model, never inferred from "is importable". One reader for every route: the stock caller's ``SETTINGS`` line,
the kit arms' ``ACTIVE`` / ``APPLIED`` lines. The fail-loud switch of the pinned stack's accelerated paths lives here too
(:data:`ENV_REQUIRE`, :func:`require_refusal`).

What upstream decides, and where (the transformers fork @ef32577f; ``common.py`` = models/esmfold2/modeling_esmfold2_common.py, ``esmc.py`` =
models/esmc/modeling_esmc.py, the module ESMFold2 loads its LM from — modeling_esmfold2.py:596):

* ``atom_attn`` — common.py:25-41 try-imports ``flash_attn`` and sets the module constant ``FLASH_ATTN_AVAILABLE``; ``SWA3DRoPEAttention.forward``
  (:561) runs ``flash_attn_varlen_func`` (:578-604) when it is set — the atom encoder always builds the five-entry ``attention_params`` (:868-872) and
  the decoder reuses it, so the two-entry ``flash_attn_func`` branch (:605-612) is not reached — else ``F.scaled_dot_product_attention`` over a dense
  ``[B, N, N]`` window mask (:613-632). Word: ``flash_attn`` | ``sdpa``, from (that flag) × (the forward each instance resolves,
  :func:`forward_resolution`: the stock def; the fast line's U1 ``_swa_forward_cached``, driver/ef2_opt.py:607, passes through to the stock
  forward under the flag; atom_swa's banded forward is SDPA by construction and never installed under the flag).
* ``esmc_mlp`` — esmc.py:64-75 try-imports ``transformer_engine.pytorch`` (``_te_available``); the block factories bind ``te.LayerNormMLP`` /
  ``te.LayerNormLinear`` / ``te.Linear`` when it is set, else the pure-PyTorch ``_PyTorchLayerNormMLP`` / ``_PyTorchLayerNormLinear`` / ``nn.Linear``
  (:543-580), at model construction. Word: ``te`` | ``torch``, from the classes bound in the loaded LM (the flag before a model exists).
* ``esmc_attn`` — esmc.py:607-658 ``_scaled_dot_product_attention``: xformers (:629) and ``flash_attn_func`` (:638) ONLY ``if seq_id is None``, else
  ``F.scaled_dot_product_attention`` with the chain-aware mask (:651-656). ESMFold2's ``compute_lm_hidden_states`` (common.py:2313-2328) always
  builds and passes ``sequence_id`` (esmc.py:1241-1242 forwards it as the mask), so inside ESMFold2 the word is ``sdpa(chain_mask)`` on every image:
  xformers is importable-but-unreachable here (it serves standalone single-sequence ESMC); the packed varlen class ``_FlashMultiHeadAttention``
  needs ``attn_implementation="flash_attention_2"`` at load (:1027), which ``load_esmc`` does not pass (modeling_esmfold2.py:609-611) and which
  raises on a multi-chain ``sequence_id`` (esmc.py:1226-1233). Reported, never required.
* ``esmc_rope`` — esmc.py:56-62: ``_flash_attn_rotary_available = torch.cuda.is_available()`` when ``flash_attn.ops.triton.rotary`` imports;
  ``RotaryEmbedding.forward`` (:424-426) runs flash-attn's Triton ``apply_rotary`` on CUDA then, else the torch rotary. Word: ``flash_attn_triton`` |
  ``torch``.
* metadata — ``flash_attn=<version>|absent transformer_engine=… xformers=…`` from distribution metadata: the only words a dry run (``check``, no
  torch) can print, and what stock/check_pins.py pins against stock/PINS.json "image".

The switch: ``ESMFOLD2_OPT_REQUIRE_FAST_ENV=1`` (configs/h100.env; declared in _autoload.ENV_NAMES) makes the stock route and every kit arm REFUSE —
exit 3, one sentence naming each failing word with its expected value and the import error behind it, and the pinned stack's accelerated
layer (stock/PINS.json "image") — unless :data:`REQUIRED` holds: ``atom_attn=flash_attn``, ``esmc_mlp=te``, ``esmc_rope=flash_attn_triton``. Never a
silent slow path under the switch. Unset or ``0`` (CPU tests, other stacks): nothing is refused and the words print what runs (``atom_attn=sdpa
esmc_mlp=torch …``). Any other value is refused by name.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

COMMON_MODULE = "transformers.models.esmfold2.modeling_esmfold2_common"    # FLASH_ATTN_AVAILABLE (:25-41), SWA3DRoPEAttention (:547)
ESMC_MODULE = "transformers.models.esmc.modeling_esmc"                      # _te_available, _xformers_available, _flash_attn_available, _flash_attn_rotary_available (:47-86)
FLAG = "FLASH_ATTN_AVAILABLE"
ATOM_CLASS = "SWA3DRoPEAttention"
ENV_REQUIRE = "ESMFOLD2_OPT_REQUIRE_FAST_ENV"                               # =1: refuse (exit 3) unless REQUIRED holds; unset/0: report only
WORD_FLASH, WORD_SDPA, WORD_UNREAD = "flash_attn", "sdpa", "unread"
WORD_TE, WORD_TORCH = "te", "torch"
WORD_ROPE_TRITON = "flash_attn_triton"
WORD_ESMC_SDPA, WORD_ESMC_VARLEN = "sdpa(chain_mask)", "flash_attn_varlen"  # esmc_attn: the chain-masked F.sdpa (the only branch ESMFold2 reaches); the packed varlen class
STOCK_WORD, BANDED_WORD = "stock", "banded"                                 # forward_resolution words: the class's own def; atom_swa's banded forward
FORWARD_KERNEL: Dict[str, tuple] = {                                        # resolution word -> the kernel it runs with the flag (set, unset)
    STOCK_WORD: (WORD_FLASH, WORD_SDPA),                                    # common.py:578 / :613
    "instance:_swa_forward_cached": (WORD_FLASH, WORD_SDPA),                # driver/ef2_opt.py:607 — pass-through under the flag; the memoized dense mask otherwise
    "class:_swa_forward_cached": (WORD_FLASH, WORD_SDPA),
    BANDED_WORD: (WORD_SDPA, WORD_SDPA),                                    # atom_swa._forward_banded — banded SDPA (never installed under the flag: path stock_flash)
}
PACKAGES: Dict[str, dict] = {                                               # metadata word -> distribution names to try, the import that upstream's switch tries
    "flash_attn": {"dists": ("flash-attn", "flash_attn"), "module": "flash_attn"},
    "transformer_engine": {"dists": ("transformer-engine", "transformer_engine", "transformer_engine_torch"), "module": "transformer_engine.pytorch"},
    "xformers": {"dists": ("xformers",), "module": "xformers.ops"},
}
REQUIRED: Tuple[Tuple[str, str, str], ...] = (                             # (word, the accelerated value on the pinned stack, the import behind it) — esmc_attn is reported, never required (docstring)
    ("atom_attn", WORD_FLASH, "flash_attn"),
    ("esmc_mlp", WORD_TE, "transformer_engine.pytorch"),
    ("esmc_rope", WORD_ROPE_TRITON, "flash_attn.ops.triton.rotary"),
)
TE_CLASS_PREFIX = "transformer_engine"                                      # te.LayerNormMLP / LayerNormLinear / Linear: type(m).__module__ starts with it
TORCH_MLP_CLASSES = ("_PyTorchLayerNormMLP", "_PyTorchLayerNormLinear")     # esmc.py:486, :508 — the pure-PyTorch fallbacks the factories bind without TE
ESMC_ATTN_CLASSES = {"MultiHeadAttention": WORD_ESMC_SDPA, "_FlashMultiHeadAttention": WORD_ESMC_VARLEN}   # esmc.py:663 / :747
REQUIRE_VALUES = {"1": True, "0": False, "": False}


# ----------------------------------------------------------------------------------------------------------------- metadata (no import)
def dist_version(key: str) -> Optional[str]:
    """The installed distribution version behind a metadata word (:data:`PACKAGES`), or None — package METADATA only."""
    for name in PACKAGES[key]["dists"]:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def versions() -> Dict[str, Optional[str]]:
    return {k: dist_version(k) for k in PACKAGES}


def metadata_words(vers: Optional[Mapping] = None) -> str:
    """``flash_attn=<version>|absent transformer_engine=… xformers=…`` — the words a dry run prints (it imports nothing of torch)."""
    vers = versions() if vers is None else vers
    return " ".join(f"{k}={vers.get(k) or 'absent'}" for k in PACKAGES)


def import_error(module: str) -> Optional[str]:
    """Why ``import <module>`` fails here (``<ExceptionType>: <message>``), or None when it imports — asked only to NAME the reason beside a False
    switch; upstream's switches, not this probe, decide the paths."""
    try:
        importlib.import_module(module)
        return None
    except Exception as e:  # noqa: BLE001 — an ABI-broken wheel raises ImportError/OSError with the symbol in the text; named verbatim
        return f"{type(e).__name__}: {e}"[:300]


# ----------------------------------------------------------------------------------------------------------------- upstream's modules
def _module(name: str, load: bool = False):
    mod = sys.modules.get(name)
    if mod is None and load:
        mod = importlib.import_module(name)
    return mod


def common_module(load: bool = False):
    """Upstream's common modeling module if imported in this process; ``load=True`` imports it (torch and transformers with it)."""
    return _module(COMMON_MODULE, load)


def esmc_module(load: bool = False):
    """The fork's ESMC module if imported in this process; ``load=True`` imports it (it try-imports TE / xformers / flash-attn at import)."""
    return _module(ESMC_MODULE, load)


def flag_state(common=None, load: bool = False) -> dict:
    """Upstream's atom-attention switch as this process has it: ``available`` = bool(FLASH_ATTN_AVAILABLE) (None = module not imported)."""
    mod = common if common is not None else common_module(load)
    if mod is None:
        return {"available": None, "read": False, "module": COMMON_MODULE}
    if not hasattr(mod, FLAG):
        raise AttributeError(f"{getattr(mod, '__name__', COMMON_MODULE)} has no attribute {FLAG!r}: upstream's flash-attn switch moved")
    return {"available": bool(getattr(mod, FLAG)), "read": True, "module": getattr(mod, "__name__", COMMON_MODULE)}


def esmc_flags(esmc=None, load: bool = False) -> Optional[dict]:
    """The fork's ESMC switches (``_te_available``, ``_xformers_available``, ``_flash_attn_available``, ``_flash_attn_rotary_available``) as imported
    here, or None when the module is not in the process (or, with ``load``, does not import: the words read ``unread`` and the switch refuses them)."""
    try:
        mod = esmc if esmc is not None else esmc_module(load)
    except ImportError:
        mod = None
    if mod is None:
        return None
    return {k: (bool(getattr(mod, k)) if hasattr(mod, k) else None)
            for k in ("_te_available", "_xformers_available", "_flash_attn_available", "_flash_attn_rotary_available")}


# ----------------------------------------------------------------------------------------------------------------- the bound forwards / classes
def _forward_name(f) -> str:
    """The function name behind a bound method / plain function stored as a ``forward``."""
    return str(getattr(getattr(f, "__func__", f), "__name__", type(f).__name__))


def forward_resolution(instances: Iterable, words: Optional[Mapping] = None) -> Dict[str, int]:
    """What ``nn.Module.__call__`` resolves ``self.forward`` to on each instance, counted by word: ``stock`` (the class's own ``forward`` def),
    ``instance:<name>`` (an attribute in the instance's ``__dict__`` — e.g. ``instance:_swa_forward_cached``, the fast line's U1 forward),
    ``class:<name>`` (a class-attribute patch), or the word ``words`` gives for a
    specific function object (atom_swa passes ``{_forward_banded: "banded"}``). The one census every caller uses."""
    words = dict(words or {})
    out: Dict[str, int] = {}
    for m in instances:
        f = vars(m).get("forward")
        fn, where = (getattr(type(m), "forward", None), "class") if f is None else (f, "instance")
        base = getattr(fn, "__func__", fn)
        if base in words:
            word = words[base]
        elif where == "class" and _forward_name(fn) == "forward":                  # the class's own def (its source digest is what atom_swa's install guard pins)
            word = STOCK_WORD
        else:
            word = f"{where}:{_forward_name(fn)}"
        out[word] = out.get(word, 0) + 1
    return dict(sorted(out.items()))


def _modules(model) -> List:
    return list(model.modules()) if model is not None and callable(getattr(model, "modules", None)) else []


def atom_instances(model, class_name: str = ATOM_CLASS) -> List:
    """Every module under ``model`` whose class is named ``class_name`` (the atom attention of all three atom transformers)."""
    return [m for m in _modules(model) if type(m).__name__ == class_name]


def banded_words() -> dict:
    """atom_swa's word for its forward, when that module is imported in the process (it is under the row-sharded line; never imported here)."""
    swa = sys.modules.get((__package__ or "esmfold2_opt") + ".atom_swa")
    fb = getattr(swa, "_forward_banded", None) if swa is not None else None
    return {fb: BANDED_WORD} if fb is not None else {}


def kernel_word(available: Optional[bool], resolution: Mapping[str, int]) -> str:
    """``flash_attn`` | ``sdpa`` from the flag and the resolved forwards (:data:`FORWARD_KERNEL`); ``mixed(...)`` when instances run different
    kernels; ``unknown(<word>)`` for a forward the table does not know; ``unread`` when the flag is not read. With no instances (no model yet)
    the word is the stock forward's kernel under the flag — what a model loaded in this process runs unless a lever rebinds it."""
    if available is None:
        return WORD_UNREAD
    if not resolution:
        return FORWARD_KERNEL[STOCK_WORD][0 if available else 1]
    kernels: Dict[str, int] = {}
    for word, n in resolution.items():
        k = FORWARD_KERNEL.get(word)
        kw = k[0 if available else 1] if k else f"unknown({word})"
        kernels[kw] = kernels.get(kw, 0) + int(n)
    if len(kernels) == 1:
        return next(iter(kernels))
    return "mixed(" + ",".join(f"{k}x{v}" for k, v in sorted(kernels.items())) + ")"


def esmc_bound(model) -> dict:
    """The LM paths from the classes BOUND in the loaded model: ``mlp`` = {te: n, torch: n} over TE modules (type(m).__module__ under
    ``transformer_engine``) and the pure-PyTorch fallbacks (:data:`TORCH_MLP_CLASSES`); ``attn`` = {word: n} over the attention classes
    (:data:`ESMC_ATTN_CLASSES`). Empty counts without a model."""
    mlp: Dict[str, int] = {}; attn: Dict[str, int] = {}
    for m in _modules(model):
        cls = type(m)
        if (cls.__module__ or "").startswith(TE_CLASS_PREFIX):
            mlp[WORD_TE] = mlp.get(WORD_TE, 0) + 1
        elif cls.__name__ in TORCH_MLP_CLASSES:
            mlp[WORD_TORCH] = mlp.get(WORD_TORCH, 0) + 1
        if cls.__name__ in ESMC_ATTN_CLASSES:
            w = ESMC_ATTN_CLASSES[cls.__name__]; attn[w] = attn.get(w, 0) + 1
    return {"mlp": dict(sorted(mlp.items())), "attn": dict(sorted(attn.items()))}


def _one_word(counts: Mapping[str, int], default: str) -> str:
    if not counts:
        return default
    if len(counts) == 1:
        return next(iter(counts))
    return "mixed(" + ",".join(f"{k}x{v}" for k, v in sorted(counts.items())) + ")"


def esmc_state(model=None, esmc=None, load: bool = False) -> dict:
    """``esmc_mlp`` / ``esmc_attn`` / ``esmc_rope`` and their evidence: from the bound classes when a model is given (``esmc_source=bound``), else from
    the module switches (``esmc_source=module``: what the factories WILL bind); ``unread`` when the module is not imported."""
    flags = esmc_flags(esmc, load)
    bound = esmc_bound(model) if model is not None else {"mlp": {}, "attn": {}}
    if flags is None and not bound["mlp"]:
        return {"esmc_mlp": WORD_UNREAD, "esmc_attn": WORD_UNREAD, "esmc_rope": WORD_UNREAD, "esmc_flags": None, "esmc_bound": bound, "esmc_source": "unread"}
    if bound["mlp"] or bound["attn"]:
        mlp = _one_word(bound["mlp"], WORD_UNREAD); attn_w = _one_word(bound["attn"], WORD_ESMC_SDPA); source = "bound"
    else:
        mlp = WORD_TE if (flags or {}).get("_te_available") else WORD_TORCH; attn_w = WORD_ESMC_SDPA; source = "module"
    rope = WORD_ROPE_TRITON if (flags or {}).get("_flash_attn_rotary_available") else (WORD_TORCH if flags is not None else WORD_UNREAD)   # RotaryEmbedding.forward reads the module switch per call (esmc.py:424)
    lm = getattr(model, "_esmc", None) if model is not None else None
    impl = getattr(getattr(lm, "config", None), "_attn_implementation", None) if lm is not None else None
    return {"esmc_mlp": mlp, "esmc_attn": attn_w, "esmc_rope": rope, "esmc_flags": flags, "esmc_bound": bound, "esmc_source": source, "esmc_attn_implementation": impl}


# ----------------------------------------------------------------------------------------------------------------- the state and its words
def state(model=None, common=None, load: bool = False) -> dict:
    """Every path of this process in one dict: ``atom_attn`` (:func:`kernel_word`), ``upstream_flag``, ``atom_forward`` (the census over the model's
    SWA3DRoPEAttention instances; {} without a model), ``atom_modules``; the ESMC words (:func:`esmc_state`); ``versions`` (metadata) and
    ``import_errors`` (for each package of :data:`PACKAGES` whose module does not import — named, informational). ``load=True`` imports upstream's
    two modules when they are not yet in the process (torch with them)."""
    fs = flag_state(common, load)
    inst = atom_instances(model) if model is not None else []
    res = forward_resolution(inst, banded_words()) if inst else {}
    out = {"atom_attn": kernel_word(fs["available"], res), "upstream_flag": fs["available"], "atom_forward": res, "atom_modules": len(inst), "module": fs["module"]}
    out.update(esmc_state(model, load=load))
    out["versions"] = versions()
    errs: Dict[str, str] = {}
    if fs["read"] or out.get("esmc_flags") is not None:                              # upstream's modules are in the process: name why each accelerated package does not import (None when it does)
        for k, spec in PACKAGES.items():
            e = import_error(spec["module"])
            if e:
                errs[k] = e
    out["import_errors"] = errs
    return out


def forward_words(resolution: Mapping[str, int]) -> str:
    return ",".join(f"{k}x{v}" for k, v in resolution.items()) or "-"


def words(st: Mapping, forward: Optional[bool] = None) -> str:
    """``atom_attn=<w> [atom_forward=<census>] esmc_mlp=<w> esmc_attn=<w> esmc_rope=<w> flash_attn=<v|absent> transformer_engine=<v|absent> xformers=<v|absent>``
    — the words every route prints from one state dict (``forward``: include the census; default = when the state carries one)."""
    parts = [f"atom_attn={st.get('atom_attn', WORD_UNREAD)}"]
    res = st.get("atom_forward") or {}
    if forward or (forward is None and res):
        parts.append(f"atom_forward={forward_words(res)}")
    for k in ("esmc_mlp", "esmc_attn", "esmc_rope"):
        parts.append(f"{k}={st.get(k, WORD_UNREAD)}")
    parts.append(metadata_words(st.get("versions") or {}))
    return " ".join(parts)


# ----------------------------------------------------------------------------------------------------------------- the fail-loud switch
def require_value(environ: Optional[Mapping] = None) -> bool:
    """``ESMFOLD2_OPT_REQUIRE_FAST_ENV``: ``1`` → True; unset, empty or ``0`` → False; anything else → ValueError (refused by name, never read as off)."""
    environ = os.environ if environ is None else environ
    raw = (environ.get(ENV_REQUIRE) or "").strip()
    if raw not in REQUIRE_VALUES:
        raise ValueError(f"{ENV_REQUIRE}={raw!r}: expected 1 (refuse unless the pinned stack's accelerated paths are live) or 0/unset")
    return REQUIRE_VALUES[raw]


def pinned_image() -> dict:
    """stock/PINS.json "image" — the pinned stack's accelerated layer (python_packages, wheels) — for the refusal sentence; {} when the pins
    cannot be read here."""
    try:
        from . import stack                                                             # lazily: the stock caller imports this module before torch and reads the pins only to word a refusal
        return dict(stack.pins().get("image") or {})
    except Exception:  # noqa: BLE001
        return {}


def _image_phrase(image: Optional[Mapping]) -> str:
    img = pinned_image() if image is None else dict(image)
    pk = img.get("python_packages") or {}
    have = ", ".join(f"{k} {v}" for k, v in pk.items()) or "its pinned packages"
    return f"the pinned stack's accelerated layer ({have}; stock/PINS.json \"image\") is not live here"


def failing_words(st: Mapping) -> List[Tuple[str, str, str, str]]:
    """[(word, got, expected, import behind it)] for every :data:`REQUIRED` word the state does not meet (``unread`` counts as not met)."""
    return [(w, str(st.get(w, WORD_UNREAD)), want, mod) for w, want, mod in REQUIRED if st.get(w, WORD_UNREAD) != want]


def require_refusal(st: Mapping, environ: Optional[Mapping] = None, image: Optional[Mapping] = None) -> Optional[str]:
    """The refusal sentence when :data:`ENV_REQUIRE` is 1 and the state misses any :data:`REQUIRED` word, else None. Each failing word is named with
    its expected value and the import error behind it (from ``st["import_errors"]`` or probed here)."""
    if not require_value(environ):
        return None
    bad = failing_words(st)
    if not bad:
        return None
    errs = dict(st.get("import_errors") or {})
    named = []
    for w, got, want, mod in bad:
        key = mod.split(".")[0]
        err = errs.get(key) or import_error(mod)
        named.append(f"{w}={got} (expected {want}" + (f"; import {mod}: {err}" if err else f"; {mod} imports but upstream's switch is off in this process") + ")")
    return (f"{ENV_REQUIRE}=1 requires the pinned stack's accelerated paths and this interpreter lacks " + ", ".join(named)
            + f" — {_image_phrase(image)}: refused (rc 3), no fallback to the slow paths")


def require_refusal_metadata(environ: Optional[Mapping] = None, image: Optional[Mapping] = None, vers: Optional[Mapping] = None) -> Optional[str]:
    """The dry-run form (``check`` imports nothing of torch): refuse by name when :data:`ENV_REQUIRE` is 1 and a distribution behind a required word
    is absent from the metadata (flash_attn, transformer_engine). A present distribution is not proof the path is live — the run-time gates read
    the switches — but an absent one is proof it is not."""
    if not require_value(environ):
        return None
    vers = versions() if vers is None else dict(vers)
    need = sorted({mod.split(".")[0] for _, _, mod in REQUIRED})
    missing = [k for k in need if not vers.get(k)]
    if not missing:
        return None
    return (f"{ENV_REQUIRE}=1 requires the pinned stack's accelerated paths and the distribution metadata lacks " + ", ".join(missing)
            + f" ({metadata_words(vers)}) — {_image_phrase(image)}: refused (rc 3)")


__all__ = ["COMMON_MODULE", "ESMC_MODULE", "FLAG", "ATOM_CLASS", "ENV_REQUIRE", "WORD_FLASH", "WORD_SDPA", "WORD_UNREAD", "WORD_TE", "WORD_TORCH",
           "WORD_ROPE_TRITON", "WORD_ESMC_SDPA", "WORD_ESMC_VARLEN", "STOCK_WORD", "BANDED_WORD", "FORWARD_KERNEL", "PACKAGES", "REQUIRED",
           "dist_version", "versions", "metadata_words", "import_error", "common_module", "esmc_module", "flag_state", "esmc_flags", "forward_resolution",
           "atom_instances", "kernel_word", "esmc_bound", "esmc_state", "state", "words", "forward_words", "require_value", "pinned_image",
           "failing_words", "require_refusal", "require_refusal_metadata"]
