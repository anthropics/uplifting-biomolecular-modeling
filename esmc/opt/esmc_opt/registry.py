"""The levers — one registry, read by the modes table, the activation, the dry run and the tests.

Every lever is a kit: a subpackage of ``esmc_opt.kits`` (``stack.kit_dir()``). A lever is applied ONLY by its kit's own ``apply()`` (``stack.py``); nothing here transcribes a
kit value — the entries name the kit directory, its class, its numerics tier as the kit's own bytes state it, its switches, and the
variants (checkpoint sizes) it serves.

Classes (the release README's vocabulary): ``forward`` = a lever on the forward's kernels; ``datapath`` = boot, load and tokenisation.
Tiers: ``exact`` = byte-identical to stock under the model's deterministic recipe (by construction and/or by the kit's own record); no lever
of this model is tier 2.

Regimes: the call shapes a command serves — ``b1`` = the SDK's one-protein call (``client.logits`` on one ``ESMProteinTensor``);
``batched`` = the caller's own padded batched forward (``client.model(input_ids[B, Lmax], attention_mask=...)``, the upstream cookbook
form). The fused kit serves every call shape at every size (eager: no graph capture, nothing that depends on a shape recurring); the regime is a
fact of the run (the KERNELS line, the manifest), not a lever selector.
"""
from __future__ import annotations

KIT_PACKAGE = "esmc_opt.kits"                         # the kits' package (esmc_opt/kits/); a lever's module = f"{KIT_PACKAGE}.{dir}"
KIT_ENV = "ESMC_KIT"                                  # the kits' shared marker (esmc_opt/kits/__init__.py:11): set by every apply(), never by this package
# The kits' OWN environment switches, by exact name, read from the kits' bytes (grep os.environ / KIT_ENV in esmc_opt/kits; file:line per name):
# under a named mode any of these present in the environment is refused BY NAME (one mode table) — never a prefix, so a name the kits never
# read (a caller's own bookkeeping) is ignored.
KIT_SWITCHES = {   # name -> what it switches — <the read site under esmc_opt/kits>:<line> (the tests lock every site to the bytes)
    "ESMC_KIT": "the kits' shared marker / flag variable (KIT_ENV: set by every apply, asserted absent by the stock arm) — esmc_opt/kits/__init__.py:11, set by each kit's apply",
}
# Environment names the kits READ FOR THEIR RECORDS ONLY (labels in their own record files; nothing they run changes with them) —
# not switches: the one-mode table ignores them, the stock arm does not strip them; file:line each. The kits read none.
KIT_BOOKKEEPING_ENV = {}
KIT_ENV_NAMES = tuple(KIT_SWITCHES)                    # the exact names (the stock arm proves every one absent; a kit mode refuses any present)

LEVERS = {
    "pipe": {
        "dir": "pipe",
        "class": "datapath",
        "tier": "exact",
        "what": ("the pipeline levers of the wall outside the forward, on the SDK route: 'boot' (the device-side safetensors loader + "
                 "meta-init skip, composed by import from boot), 'tok' (the exact character-table tokeniser behind tokenizer([seq], "
                 "return_tensors='pt', padding=True)); both patch at the load; the kit's own per-route table decides the set (apply(route, levers='auto'))"),
        "apply": "esmc_opt/kits/pipe/__init__.py:592 apply(route, levers, weights_dir); :749 from_pretrained_sdk (the one call swap)",
        "switches": {"in_mode": "exact", "kit_marker": "ESMC_KIT=pipe"},
        "regime": "any",
        "variants": ("300m", "600m", "6b"),
        "composes": ("boot",),
        "route_condition": "apply() precedes ESMC.from_pretrained (both levers patch at the load); before or after the upstream import alike",
    },
    "fused": {
        "dir": "fused",
        "class": "forward",
        "tier": "exact",
        "what": ("the fused eager forward, every call shape (no graph capture, no bucketing): the stock's kernels on the [b*L, d] rows with "
                 "fewer launches — the varlen metadata once per forward, the fp32 LayerNorm weight copies made once at engage, the QKV stack "
                 "removed, unpad/pad as views at batch 1 — plus the composed items (its COMPOSED_ITEMS): residual_ln (fused residual+LayerNorm "
                 "CUDA kernels, built once from kernels.cu), qk_rotary (the Triton qk-LayerNorm+rotary kernel), thin "
                 "(thin launchers for the Transformer Engine templates), composed per shape; every ESM C size; serves the SDPA-constructed and the flash-constructed model"),
        "apply": "esmc_opt/kits/fused/__init__.py:272 apply(model) (no knobs); :320 remove(model); _patch.py engage/disengage",
        "switches": {"in_mode": "exact", "kit_marker": "ESMC_KIT=fused", "knobs": "none (apply(model) takes no options)"},
        "regime": "any",
        "variants": ("300m", "600m", "6b"),
        "composes": ("residual_ln", "qk_rotary", "thin"),
        "route_condition": "bitwise on the three ESM C shapes; the composition is fixed per shape (fused SERVED_CONFIGS / ITEM_SERVES): every size gets the base path + the residual kernel + the thin launchers; the residual+LayerNorm kernel (TE's <8192,1,4,16> config: 2048 < hidden <= 8192) and the q/k-LayerNorm+rotary kernel (hidden a multiple of 512) compose at 6B only and are named out by shape on 300M / 600M",
        "placement": "the item qk_rotary is the stripped item (family dir qk_rotary_stripped), placed at the module path the kit's COMPOSED_ITEMS names",
    },
}

LEVER_ORDER = ("pipe", "fused")                       # apply order in a process: datapath before the load, the forward lever on the loaded model


def lever_module(name: str) -> str:
    return f"{KIT_PACKAGE}.{LEVERS[name]['dir']}"


def levers_for(variant: str, regime: str, names) -> tuple:
    """Of ``names``, the levers that serve ``variant`` in ``regime`` (a lever that serves neither is left to the mode table's caller)."""
    out = []
    for n in names:
        lv = LEVERS[n]
        if variant not in lv["variants"]:
            continue
        if lv["regime"] not in ("any", regime):
            continue
        out.append(n)
    return tuple(out)
