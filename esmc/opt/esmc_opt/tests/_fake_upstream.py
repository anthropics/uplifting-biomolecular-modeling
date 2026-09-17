"""A duck-typed stand-in for a BUILT upstream model, on a box without torch or the upstream: the two upstream modules the reader inspects
(esm.models.esmc.kernels / .layers: the flags, the bound function objects, the block classes) and a client whose blocks bind either the
accelerators' classes (`engaged`) or upstream's fallbacks — enough for kernels.census to read real bindings, nothing more.
`install(kind)` puts the fake modules in sys.modules and returns a client; `uninstall()` removes them."""
import sys
import types

KERNELS, LAYERS = "esm.models.esmc.kernels", "esm.models.esmc.layers"
FAKES = ("esm", "esm.models", "esm.models.esmc", KERNELS, LAYERS, "flash_attn", "flash_attn.flash_attn_interface", "flash_attn.ops", "flash_attn.ops.triton",
         "flash_attn.ops.triton.rotary", "transformer_engine", "transformer_engine.pytorch")
FLASH_VERSION, TE_VERSION = "2.7.4.post1", "2.15.0"        # == stock/PINS.json stacks.accel (the tests assert it)


def _cls(name, module, base=object):
    return type(name, (base,), {"__module__": module, "__qualname__": name})


def _fn(name, module):
    f = types.FunctionType((lambda *a, **k: None).__code__, {}, name)
    f.__module__ = module
    f.__qualname__ = name
    return f


class _Param:
    def __init__(self, device, dtype):
        self.device = types.SimpleNamespace(type=device)
        self.dtype = dtype


def install(kind: str = "engaged", *, device: str = "cuda", attn_impl: str = "flash_attention_2", xformers: bool = False):
    """kind: 'engaged' (flash class + Triton rotary + TE classes bound, flags True, the accelerator modules importable-by-name),
    'fallback' (sdpa class + torch rotary + torch LN/MLP, flags False, no accelerator module), 'te_off' (flash engaged, TE fallen back)."""
    uninstall()
    L = types.ModuleType(LAYERS)
    L.EsmcMultiHeadAttention = _cls("EsmcMultiHeadAttention", LAYERS)
    L.EsmcFlashMultiHeadAttention = _cls("EsmcFlashMultiHeadAttention", LAYERS, L.EsmcMultiHeadAttention)
    L.EsmcRotaryEmbedding = _cls("EsmcRotaryEmbedding", LAYERS)
    L.EsmcTritonRotaryEmbedding = _cls("EsmcTritonRotaryEmbedding", LAYERS, L.EsmcRotaryEmbedding)
    L.EsmcLayerNormLinear = _cls("EsmcLayerNormLinear", LAYERS)
    L.EsmcLayerNormMLP = _cls("EsmcLayerNormMLP", LAYERS)
    TorchLinear = _cls("Linear", "torch.nn.modules.linear")
    TE_LNL = _cls("LayerNormLinear", "transformer_engine.pytorch.module.layernorm_linear")
    TE_L = _cls("Linear", "transformer_engine.pytorch.module.linear")
    TE_MLP = _cls("LayerNormMLP", "transformer_engine.pytorch.module.layernorm_mlp")
    K = types.ModuleType(KERNELS)
    flash_on = kind in ("engaged", "te_off")
    te_on = kind == "engaged"
    K.FLASH_ATTN_INSTALLED = flash_on
    K.FLASH_ATTN_ROTARY_INSTALLED = flash_on
    K.TE_INSTALLED = te_on
    K.XFORMERS_INSTALLED = bool(xformers)
    K.flash_attn_varlen_qkvpacked_func = _fn("flash_attn_varlen_qkvpacked_func", "flash_attn.flash_attn_interface") if flash_on else None
    K.flash_attn_func = _fn("flash_attn_func", "flash_attn.flash_attn_interface") if flash_on else None
    K.unpad_input = _fn("unpad_input", "flash_attn.bert_padding") if flash_on else None
    K.pad_input = _fn("pad_input", "flash_attn.bert_padding") if flash_on else None
    K.apply_triton_rotary = _fn("apply_rotary", "flash_attn.ops.triton.rotary") if flash_on else None
    K.te = types.ModuleType("transformer_engine.pytorch") if te_on else None
    K.xops = types.ModuleType("xformers.ops") if xformers else None
    mods = {LAYERS: L, KERNELS: K}
    for parent in ("esm", "esm.models", "esm.models.esmc"):
        mods[parent] = types.ModuleType(parent)
    if flash_on:
        fa = types.ModuleType("flash_attn"); fa.__version__ = FLASH_VERSION
        mods.update({"flash_attn": fa, "flash_attn.flash_attn_interface": types.ModuleType("flash_attn.flash_attn_interface"), "flash_attn.ops": types.ModuleType("flash_attn.ops"),
                     "flash_attn.ops.triton": types.ModuleType("flash_attn.ops.triton"), "flash_attn.ops.triton.rotary": types.ModuleType("flash_attn.ops.triton.rotary"),
                     "flash_attn_2_cuda": types.ModuleType("flash_attn_2_cuda")})
    if te_on:
        te = types.ModuleType("transformer_engine"); te.__version__ = TE_VERSION
        mods.update({"transformer_engine": te, "transformer_engine.pytorch": K.te})
    if xformers:
        xf = types.ModuleType("xformers"); xf.__version__ = "0.0.35"
        mods.update({"xformers": xf, "xformers.ops": K.xops})
    for n, m in mods.items():
        m.__dict__.setdefault("__path__", [])                     # a package, so find_spec on its children consults sys.modules
        sys.modules[n] = m
    use_flash = flash_on and device == "cuda" and attn_impl == "flash_attention_2"
    attn_cls = L.EsmcFlashMultiHeadAttention if use_flash else L.EsmcMultiHeadAttention
    rot_cls = L.EsmcTritonRotaryEmbedding if use_flash else L.EsmcRotaryEmbedding
    use_te = te_on and device == "cuda"

    def block():
        b = types.SimpleNamespace()
        b.attn = attn_cls()
        b.attn.layernorm_qkv = (TE_LNL if use_te else L.EsmcLayerNormLinear)()
        b.attn.out_proj = (TE_L if use_te else TorchLinear)()
        b.attn.rotary = rot_cls()
        b.ffn = (TE_MLP if use_te else L.EsmcLayerNormMLP)()
        return b

    core = types.SimpleNamespace()
    core.transformer = types.SimpleNamespace(blocks=[block(), block()])
    core._use_flash_attn = use_flash
    core.parameters = lambda: iter([_Param(device, "torch.bfloat16")])
    model = types.SimpleNamespace(esmc=core, config=types.SimpleNamespace(attn_implementation=attn_impl))
    client = types.SimpleNamespace(model=model)
    return client


def uninstall():
    for n in FAKES + ("flash_attn_2_cuda", "xformers", "xformers.ops"):
        sys.modules.pop(n, None)
