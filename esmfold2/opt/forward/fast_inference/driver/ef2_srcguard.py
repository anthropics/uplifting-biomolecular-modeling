"""ef2_srcguard — the install-time STOCK-MISMATCH guard of the levers that re-issue an upstream function statement by statement (the recycle
loop, the MSA-encoder block, the pair Transition, the TriMul block, the atom encoder / decoder, the diffusion sampler and transformer, the MSA
featuriser): each such lever is written against ONE upstream source (the pinned Biohub transformers / esm commits, stock/PINS.json); on a
different upstream the re-issued statements could silently diverge from the library's.  ``check(lever)`` reads the source FILE of the imported
upstream module (never the live attribute: other levers patch those), extracts the pinned functions with ``ast`` and compares a normalised digest
(sha256 over the stripped, non-blank, non-comment lines — decorators and docstrings included) with the table below; a mismatch raises
``StockMismatch`` (a RuntimeError) naming the lever and the function — the mode refuses by name at install, never at fold time.  Hooks and pure
kernel swaps behind an unchanged call signature need no entry.  Regenerate the table only together with a pins move (stock/PINS.json).
"""
import ast, hashlib, importlib

PINNED = {                                                                # (module, qualname) -> digest of the pinned upstream source
    ("transformers.models.esmfold2.modeling_esmfold2", "ESMFold2Model._run_one_loop"): "9ccf4df2ca6a3f78d37f9a61bc2fe03f7a7467df9f5c40fa0f69d327826fce72",
    ("transformers.models.esmfold2.modeling_esmfold2", "MSAEncoderBlock.forward"): "63bb15c263042db45ced9913d7b4f14ef609fdb6df579c7d343e0487c36ead07",
    ("transformers.models.esmfold2.modeling_esmfold2", "MSAEncoder.forward"): "7b2e95a956b96a5fb6600a1598792af5363ed3d9cb6818d9fea9dbb02958a310",
    ("transformers.models.esmfold2.modeling_esmfold2", "PairTransition.forward"): "4202077deb89da19d787645ad55d84d144c039e63af57a57f1b2b1262f8624bf",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "maybe_subsample_msa"): "e813f7e9d88ebd86658ea7483b0b576be4fb81498349eb5d70d2c139b73470fa",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "TriangleMultiplicativeBlock.forward"): "53218226e8d262dc8822baa4e9d9e84a7c38f6b65faacc8b065adf23ff783bca",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "Transition.forward"): "701cf3821504a470f1e6d97012b5fcf8e44dbf2c1f18fd88d92bf769e167af9a",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "SWAAtomBlock.forward"): "ef56729f46fa5a21ec4962c9e8eed00e454bcc65582051b444c18cd756db0390",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "SWA3DRoPEAttention.forward"): "df2f9fd170dda7f4d4a679147a6e60ffaa4e29e9112e9cb7532178092fdfbf0a",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "ESMFold2AtomEncoder.forward"): "29d2b064a6f45d20d47f354cd9fdbe6857ff329ff0c97698e0b481488d6c2faa",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "ESMFold2AtomDecoder.forward"): "11b48dfc0ec1fef9323d34865419e4d0d433c5d313f6bfeb9ab441979c36e4f3",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionStructureHead.sample"): "89a2d396d877a2d88e21d0700d3ad910eaba9c015f30fefaacf17472d363c297",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionStructureHead._center_random_augmentation"): "2b5dd87ffee23743e145bd030e939e3cff80efde7462f5bdba449139a55d1864",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionStructureHead._weighted_rigid_align"): "71a5a20fa7332c55000d50c4295e40eed04927603be53f26424247b0ac9f58e0",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionTransformer.forward"): "2c40cd2d08f46bf7bdc407139583564391affe8290fcc546fee23f380f1582f7",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "AttentionPairBias.forward"): "4a22471fdcd86a13543d3e46b418a38984ce2c96f32651c210c5518f2918fc75",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "ConditionedTransitionBlock.forward"): "8accea6a37862d8eb9bb520fc6db802b56a888db1b8bbec22d0913cd8df7f5a5",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionConditioning.forward"): "fbfb5bcac75b16008bbdfcab2580305ec0d2c5dee9989a9c5a4848e6fdfb2b9f",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionModule.forward"): "031b1ed93f63148ba1a52b39cc77af49d6a4f5b7575a8f53225c4453d1d1bbe4",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "OuterProductMean.forward"): "24681f35fb7e60ccd6a3e1d06b3449f8fbae78f8650653689728c1c25b49cca4",
    ("transformers.models.esmfold2.modeling_esmfold2_common", "MSAPairWeightedAveraging.forward"): "26509319e795e2e6c55a665d1c04504bec81b5e284e4f2d0d96664efeb780f09",
    ("esm.models.esmfold2.paired_msa", "msa_to_res_type_and_deletions"): "7850d1bbd5a9839f91a21c4b0bc71de56747a5877872ee333c90dfc61dda960a",
    ("esm.models.esmfold2.paired_msa", "construct_paired_msa"): "6e6f2ac8093f3f32fa9dcf41054ca91c20af8861ac016b96aefe2efda9c28e9e",
}
LEVER_SOURCES = {                                                         # lever -> the upstream functions its code re-issues or wraps with assumptions on their statements
    "ls": (("transformers.models.esmfold2.modeling_esmfold2", "ESMFold2Model._run_one_loop"),),
    "rg": (("transformers.models.esmfold2.modeling_esmfold2", "ESMFold2Model._run_one_loop"),),
    "m15": (("transformers.models.esmfold2.modeling_esmfold2", "MSAEncoderBlock.forward"), ("transformers.models.esmfold2.modeling_esmfold2", "PairTransition.forward")),
    "m16": (("transformers.models.esmfold2.modeling_esmfold2", "MSAEncoderBlock.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "MSAPairWeightedAveraging.forward")),
    "m17": (("transformers.models.esmfold2.modeling_esmfold2", "MSAEncoderBlock.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "OuterProductMean.forward")),
    "mh": (("transformers.models.esmfold2.modeling_esmfold2_common", "maybe_subsample_msa"), ("transformers.models.esmfold2.modeling_esmfold2", "MSAEncoder.forward"), ("transformers.models.esmfold2.modeling_esmfold2", "ESMFold2Model._run_one_loop")),
    "t15": (("transformers.models.esmfold2.modeling_esmfold2_common", "Transition.forward"),),
    "t15msa": (("transformers.models.esmfold2.modeling_esmfold2", "PairTransition.forward"), ("transformers.models.esmfold2.modeling_esmfold2", "MSAEncoderBlock.forward")),
    "t16": (("transformers.models.esmfold2.modeling_esmfold2_common", "Transition.forward"), ("transformers.models.esmfold2.modeling_esmfold2", "PairTransition.forward")),
    "t6s": (("transformers.models.esmfold2.modeling_esmfold2_common", "Transition.forward"),),
    "t6i": (("transformers.models.esmfold2.modeling_esmfold2_common", "Transition.forward"),),
    "xtr": (("transformers.models.esmfold2.modeling_esmfold2", "PairTransition.forward"),),
    "trimul": (("transformers.models.esmfold2.modeling_esmfold2_common", "TriangleMultiplicativeBlock.forward"),),
    "glue": (("transformers.models.esmfold2.modeling_esmfold2_common", "TriangleMultiplicativeBlock.forward"),),
    "ax": (("transformers.models.esmfold2.modeling_esmfold2_common", "SWAAtomBlock.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "SWA3DRoPEAttention.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "ESMFold2AtomEncoder.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "ESMFold2AtomDecoder.forward")),
    "af": (("transformers.models.esmfold2.modeling_esmfold2_common", "SWAAtomBlock.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "SWA3DRoPEAttention.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "ESMFold2AtomEncoder.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "ESMFold2AtomDecoder.forward")),
    "ro": (("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionStructureHead.sample"), ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionStructureHead._center_random_augmentation"), ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionStructureHead._weighted_rigid_align"), ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionModule.forward")),
    "kd": (("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionStructureHead._weighted_rigid_align"),),
    "dit": (("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionTransformer.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "AttentionPairBias.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "ConditionedTransitionBlock.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionConditioning.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionModule.forward")),
    "fz": (("esm.models.esmfold2.paired_msa", "msa_to_res_type_and_deletions"), ("esm.models.esmfold2.paired_msa", "construct_paired_msa")),
}
_FILES = {}                                                              # module name -> (path, {qualname: digest}) cache: each upstream file is parsed once per process


class StockMismatch(RuntimeError):
    """An upstream function a lever re-issues does not match the pinned source (or cannot be found): the lever refuses by name."""


def normalised_digest(text):
    lines = [l.strip() for l in str(text).splitlines()]
    return hashlib.sha256("\n".join(l for l in lines if l and not l.startswith("#")).encode()).hexdigest()


def _function_text(tree, source, qualname):
    nodes, node = tree.body, None
    for part in qualname.split("."):
        node = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part), None)
        if node is None:
            return None
        nodes = node.body
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "\n".join(source.splitlines()[start - 1: node.end_lineno])


def file_digest(module_name, qualname):
    """The normalised digest of ``qualname`` in the source file of the imported module ``module_name`` (None when the function is absent)."""
    ent = _FILES.get(module_name)
    if ent is None:
        mod = importlib.import_module(module_name)
        path = getattr(mod, "__file__", None)
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
        ent = (path, source, ast.parse(source, filename=path), {})
        _FILES[module_name] = ent
    path, source, tree, memo = ent
    if qualname not in memo:
        text = _function_text(tree, source, qualname)
        memo[qualname] = None if text is None else normalised_digest(text)
    return memo[qualname]


def check(lever, keys=None):
    """Raise StockMismatch naming ``lever`` unless every pinned upstream function it depends on (LEVER_SOURCES[lever], or ``keys``) has the pinned
    digest in this process's upstream source files. Returns the list of (module, qualname) checked."""
    keys = list(keys if keys is not None else LEVER_SOURCES.get(lever, ()))
    for module_name, qualname in keys:
        want = PINNED[(module_name, qualname)]
        got = file_digest(module_name, qualname)
        if got is None:
            raise StockMismatch(f"{lever}: stock mismatch: {module_name}:{qualname} not found in {_FILES[module_name][0]} (the lever re-issues this upstream function; refusing by name)")
        if got != want:
            raise StockMismatch(f"{lever}: stock mismatch: {module_name}:{qualname} source digest {got[:12]} != pinned {want[:12]} "
                                f"({_FILES[module_name][0]}; the lever re-issues this upstream function statement by statement; refusing by name)")
    return keys


def check_many(levers):
    """check() every lever of ``levers`` (a name -> bool mapping or an iterable of names); returns {lever: [keys]}."""
    items = [k for k, v in levers.items() if v] if isinstance(levers, dict) else list(levers)
    return {lv: check(lv) for lv in items if lv in LEVER_SOURCES}
