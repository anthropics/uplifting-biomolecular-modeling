"""af3_torch_api.py — documented entry points for the PyTorch AlphaFold3 port (xfold @22bdeed + OF3-layout patches).

Weights: the sokrypton/alphafold3 OF3-ported haiku records (of3_ported_weights.bin.zst, layout 'of3'); they are READ from the
params directory into GPU memory; nothing is ever written.  DeepMind's af3.bin.zst layout is key/shape-compatible by
construction but must not be used with this port.

    import sys; sys.path += ["<af3t>/af3_torch", "<af3t>/kernels", "<af3t>/kernels/third_party"]
    import af3_torch_api as A
    model = A.build_model("<checkpoint file or its dir>", levers="fastest")  # or levers="eager" / () for the eager port; see LEVER_SETS
    batch = A.batch_from_npz("batch.npz")                               # numeric AF3 features (featurise_input output) -> device dict
    with A.inference():                                                 # inference_mode + bf16 autocast (the kit's numerics class)
        tf   = A.run_target_feat(model, batch)                          # [N, 447] fp32
        emb  = A.run_trunk(model, batch, num_recycles=10)               # {'single':[N,384], 'pair':[N,N,128], 'target_feat':[N,447]} fp32
        xyz  = A.run_diffusion(model, batch, emb, seed=1)               # [num_samples, N, 24, 3] fp32 (torch RNG) — or ext=dict(init_pos,noise,rot,trans)
        conf = A.run_confidence(model, batch, emb, xyz[0])              # dict: predicted_lddt [N,24], full_pae [N,N], tmscore_adjusted_pae_global/interface [N,N], ...
        dg   = A.run_distogram(model, batch, emb)
        out  = A.forward(model, batch, seed=1)                          # everything, like alphafold3.model.Model.__call__ (1 sample)
    # DLPack bridge (e.g. to JAX):  jax.dlpack.from_dlpack(emb['pair'].contiguous())  /  torch.from_dlpack(jax_array)

All tensors live on CUDA; token dimension N == bucket size (padded, masks in batch). Shapes follow alphafold3 v3.1.
"""
import os, sys, contextlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "kernels"), os.path.join(HERE, "..", "kernels", "third_party")):
    if p not in sys.path:
        sys.path.insert(0, p)
from xfold import of3
of3.OF3 = True                                   # OF3 weight layout (sokrypton fork global_config.of3_weights=True) — set BEFORE model construction
from xfold.alphafold3 import AlphaFold3
from xfold import params as xparams, feat_batch
from xfold.nn import diffusion_head as dh
from xfold.nn.utils import mask_mean

LEVER_SETS = {
    "eager": (),                 # the port as it is: fp32 weights under bf16 autocast, xfold's own modules, no kernel lever
    # 'fastest' = bf16 weights + the fused pairformer kernels + the whole-denoiser-step CUDA graph with its
    #             hoist of step-invariant conditioning + torch.compile(mode=max-autotune-no-cudagraphs) of the atom-transformer / transition /
    #             single-conditioning glue, captured inside the step graph (compile is the bf16 numerics class, not bitwise to the uncompiled set;
    #             its first call at a token length compiles for ~1-2 min)
    #             resid_fold: the Pairformer block's residual adds folded into the trimul / transition kernel epilogues (bitwise the separate adds);
    #             triattn: the triangle attention on the shared core's triangle-attention provider, the mode's tier word naming the kernel row per
    #             (card, head dim, heads, size) cell; attn_epi: the triangle attention's gate, output projection and the block's residual
    #             add as one core kernel where the core's cell table names a launch cell for the card (bitwise the separate statements); trimul / tmpl_trimul: the
    #             128-channel pair stacks' and the 64-channel template pair stack's TriMul through the shared core's TriMul provider by the mode's tier word
    #             (the provider's measured cell per card / size / direction names the row); pwa_lnl: the MSA module's
    #             pair-weighted averaging takes its pair LayerNorm + logits projection from one fused kernel (same class, other bits)
    #             opm: the MSA module's outer-product mean on the kit's af3t_opm kernels (fused LayerNorm + projections, the outer product one GEMM per chunk of
    #             left tokens with no [N,N,1024] copy, the output contraction + the block's residual add in one kernel; same class, other bits)
    #             + the sample-batched sampler (every diffusion sample advanced together per step, the statics and the step's conditioning shared)
    #             + atom_window: the atom transformers' sequence-local attention on the support library's fused window kernels (TF32 class)
    #             + glu_proj: the transition's gated linear unit + output projection + residual add as one kernel, byte-equal to xfold's fastnn statement
    #             (exact-class: the exact set keeps it; under 'fastest' lever transition's fused LayerNorm kernel serves those modules and glu_proj steps aside by name)
    #             + trimul_exact: the TriMul through the provider's `exact` word in this engine's module form -- served only where the provider's table vouches
    #             the row byte-identical to the module for the running stack and size, the module's own statement by name elsewhere (exact-class: the exact
    #             set keeps it; under 'fastest' lever trimul's tier word serves the class and trimul_exact steps aside by name)
    "fastest": ("bf16w", "trimul", "triattn", "transition", "apb", "resid_fold", "attn_epi", "tmpl_trimul", "pwa_lnl", "opm", "pwa_msa", "stepgraph", "compile", "sbatch", "atom_window", "token_agg", "atom_rows", "prologue", "glu_proj", "trimul_exact"),
}
_KERNEL_LEVERS = ("trimul", "triattn", "transition", "apb", "resid_fold", "attn_epi", "tmpl_trimul", "pwa_lnl", "glu_proj", "pwa_msa", "trimul_exact")
_MSA_LEVERS = ("opm",)          # the MSA module's levers (kernels/af3t_msa.py: class-level patches of OuterProductMean / EvoformerBlock; its census joins af3_kernels')
SAMPLE_BATCH_ALL = 1 << 16      # AlphaFold3.sample_batch under lever 'sbatch': every sample of the run in one denoiser call (an int b > 0 = at most b per call; 0 = the serial sampler)


ATOM_WINDOW_PRECISION = "tf32rn"        # lever 'atom_window': the window kernels' dot precision (opt_core.kernels.atom_window PRECISIONS: tf32rn = cuBLAS-TF32 class)
# lever 'atom_window': the ln_qkvg launch per compute capability (a row tile of the [rows, C] activation + the C x C weight tiles live in shared
# memory): the library default (128 rows x 8 warps, its H100 sweep) needs 192 KB — cc 9.0's 227 KB holds it, cc 8.0's 163 KB does not, where a
# 32-row tile builds and runs fastest of the tiles that fit. Other cards try the default and, when the
# kernels do not build, the lever steps aside by name (warmup:<error>). Same statements, another tile: the numerics class is unchanged.
ATOM_WINDOW_CELLS = {(9, 0): {}, (8, 0): {"BLOCK_R": 32}}


def _atom_window_warmup(AW, C, H, has_bias, precision, ln_kw, device="cuda"):
    """Compile (or load from Triton's cache) both window kernels for this geometry on a dummy 2-block problem with the card's ln_qkvg launch
    (the library's warmup at its default tile, restated so the cell's tile is the one compiled)."""
    S, A_ = 1, 64
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn((S, A_, C), device=device, generator=g)
    cond = [torch.rand((A_, C), device=device, generator=g) for _ in range(5)]
    lin = [torch.nn.Linear(C, C, bias=b).to(device) for b in has_bias[:4]]
    qkvg = AW.ln_qkvg(a, cond[0], cond[1], cond[2], cond[3], *lin, 1e-5, 1.0, precision=precision, **ln_kw)
    NB = A_ // 32
    bias = torch.zeros((NB, H, 32, 128), device=device)
    ks = torch.zeros((NB,), dtype=torch.int32, device=device); n_real = torch.full((1,), A_, dtype=torch.int32, device=device)
    wo = torch.nn.Linear(C, C, bias=has_bias[4]).to(device)
    AW.window_attn(qkvg, a, bias, ks, n_real, torch.ones((1, A_), device=device), cond[4], wo.weight, wo.bias, H, 32, 128, 1e9, precision=precision)
    torch.cuda.synchronize()


def enable_atom_window(model) -> str:
    """Lever 'atom_window': bind the routed support-library module opt_core.kernels.atom_window to xfold's DiffusionCrossAttTransformer, compile
    its two kernels for the atom transformers' geometry (outside any timed step), and switch the diffusion head's hoist to carry the window
    operands. Returns "on", or the word the lever steps aside with (the stock statements serve): needs_hoist (the operands ride the hoist of
    the step-invariant conditioning: levers stepgraph / hoist), kernel_import:<error>, warmup:<error>."""
    from xfold.nn import diffusion_transformer as DT
    dh = model.diffusion_head
    if not getattr(dh, "use_hoist", False):
        return "needs_hoist"
    try:
        import atom_window as AW                                 # the routed name: forward.py routes 'atom_window' to the support library's copy before importing this module
    except Exception as e:                                       # noqa: BLE001 — the shared core absent from this interpreter: named, the stock blocks serve
        return "kernel_import:%s" % type(e).__name__
    tf = dh.atom_cross_att_encoder.atom_transformer_encoder
    C = int(tf.c_query); H = int(tf.num_head); ca = tf.cross_attention[0]
    has_bias = tuple(l.bias is not None for l in (ca.q_projection, ca.k_projection, ca.v_projection, ca.gating_query, ca.adaptive_zero_init.transition2))
    cc = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    ln_kw = dict(ATOM_WINDOW_CELLS.get(cc, ATOM_WINDOW_CELLS[(8, 0)] if cc is not None and cc[0] == 8 else {}))
    try:
        if cc is not None:
            with torch.no_grad():
                _atom_window_warmup(AW, C, H, has_bias, ATOM_WINDOW_PRECISION, ln_kw)   # both kernels compiled for this geometry now, outside any timed or captured step
    except Exception as e:                                       # noqa: BLE001 — the kernels do not build on this card / Triton: named, the stock blocks serve
        return "warmup:%s:%s" % (type(e).__name__, str(e).splitlines()[0][:120] if str(e) else "")
    DT.DiffusionCrossAttTransformer.WINDOW_KERNEL = AW
    DT.DiffusionCrossAttTransformer.WINDOW_PRECISION = ATOM_WINDOW_PRECISION
    DT.DiffusionCrossAttTransformer.WINDOW_LN_KW = ln_kw
    DT.DiffusionCrossAttTransformer.WINDOW_COUNTS["cell"] = "cc%s:%s" % ("%d.%d" % cc if cc else "-", ",".join("%s=%s" % kv for kv in sorted(ln_kw.items())) or "default")
    dh.use_atom_window = True
    return "on"


def enable_token_agg(model) -> str:
    """Lever 'token_agg': bind the kit's aggregation kernel (af3t_token_agg, carried beside the DTK modules) to xfold's AtomCrossAttEncoder and
    compile it for the encoder's channel count (outside any timed step). Returns "on", or the word the lever steps aside with (the stock
    statements serve): kernel_import:<error>, warmup:<error>."""
    from xfold.nn import atom_cross_attention as ACA
    try:
        import af3t_token_agg as TA                              # the kit's dtk directory is on the model process's sys.path (stack.kit_sys_path)
    except Exception as e:                                       # noqa: BLE001 — Triton absent / the module missing: named, the stock statements serve
        return "kernel_import:%s" % type(e).__name__
    enc = model.diffusion_head.atom_cross_att_encoder
    try:
        if torch.cuda.is_available():
            with torch.no_grad():
                TA.warmup(int(enc.project_atom_features_for_aggr.out_features))
    except Exception as e:                                       # noqa: BLE001 — the kernel does not build on this card / Triton: named, the stock statements serve
        return "warmup:%s:%s" % (type(e).__name__, str(e).splitlines()[0][:120] if str(e) else "")
    ACA.AtomCrossAttEncoder.TOKEN_AGG = TA
    return "on"


def enable_atom_rows(model, levers) -> str:
    """Lever 'atom_rows': the atom transformers' three blocks (window kernels AND transition) on one contiguous fp32 copy of the real query
    blocks' rows (DiffusionCrossAttTransformer.forward_windowed_rows). Needs 'atom_window' (the window kernels and their hoisted row count):
    else steps aside with needs_atom_window. Under 'compile' each atom transition block gets a rows variant compiled with a dynamic row count
    (`forward_rows`, guarded like every compiled callable); without it the eager block serves. Returns "on" or the step-aside word."""
    from xfold.nn import diffusion_transformer as DT
    if getattr(model, "_atom_window_state", None) != "on":
        return "needs_atom_window"
    dh = model.diffusion_head
    for tname, tr in (("enc", dh.atom_cross_att_encoder.atom_transformer_encoder), ("dec", dh.atom_cross_att_decoder.atom_transformer_decoder)):
        for i, blk in enumerate(tr.transition_block):
            base = getattr(blk, "_forward_eager", blk.forward)
            blk.forward_rows = _guarded("%s.rows.%d" % (tname, i), base, torch.compile(base, dynamic=True)) if "compile" in levers else base
    DT.DiffusionCrossAttTransformer.WINDOW_ROWS_ONLY = True
    return "on"


def set_provider_tier(word):
    """Name the shared core providers' TIER WORD the kernel levers ask for ("exact" | "fast" | "big"): levers trimul / tmpl_trimul hand it to the TriMul provider
    (opt_core.kernels.trimul), lever transition to the transition provider (opt_core.kernels.transition) and lever apb to the pair-bias attention provider
    (opt_core.kernels.apb), whose measured cells name the row per card, shape and size. The model process calls this BEFORE build_model with its mode's
    word (af3_torch_opt.forward: "big" when the pred composes big's memory levers, "fast" otherwise; exact enables none of these levers -- its trimul_exact asks the
    provider's `exact` word itself); a caller of build_model that never calls it asks for "fast"."""
    import af3_kernels as K
    return K.set_tier(word)


def provider_tier():
    """The tier word this process's kernel levers ask the shared core's providers for (set_provider_tier; "fast" until a mode names "big")."""
    import af3_kernels as K
    return K.provider_tier() if hasattr(K, "provider_tier") else "fast"


def build_model(params_dir, levers="eager", num_recycles=10, num_samples=1, diffusion_steps=200, device="cuda", compile=False):
    """Construct the model, load the OF3-layout haiku params (raises on any missing key), enable levers. Returns the nn.Module.
    levers: name in LEVER_SETS (default "eager": the port, no lever) or an iterable of lever names: bf16w | trimul | trimul_exact | triattn | transition | apb | resid_fold | attn_epi | tmpl_trimul | pwa_lnl |
    opm (the MSA module's outer-product mean on the kit's af3t_opm kernels, the Evoformer block's residual add folded) |
    hoist (step-invariant diffusion conditioning computed once per trajectory) | stepgraph (whole denoiser step CUDA graph + hoist) | compile |
    sbatch (the sample-batched sampler: AlphaFold3._sample_diffusion_batched — all diffusion samples per denoiser call, stock's draws draw for draw) |
    atom_window (the atom transformers' sequence-local attention on the support library's window kernels: DiffusionCrossAttTransformer.forward_windowed) |
    token_agg (the atom-attention encoder's atom -> token aggregation as one kernel: AtomCrossAttEncoder._token_aggregate) |
    atom_rows (the atom transformers' blocks on one contiguous copy of the real query blocks' rows: DiffusionCrossAttTransformer.forward_windowed_rows) |
    prologue (the sample-batched step's augmentation + noise arithmetic over the sample axis, draws unchanged: diffusion_head.augment_and_noise_batched).
    compile=True (or a torch.compile mode string) is the opt-in flag equivalent to adding the 'compile' lever (levers='fastest')."""
    levers = tuple(LEVER_SETS[levers]) if isinstance(levers, str) else tuple(levers)
    model = AlphaFold3(num_recycles=num_recycles, num_samples=num_samples, diffusion_steps=diffusion_steps)
    info = xparams.import_jax_weights_(model, params_dir)
    model = model.to(device).eval()
    model._af3t_load_info = info
    if "bf16w" in levers:
        for mod in model.modules():
            if isinstance(mod, torch.nn.Linear):
                mod.weight.data = mod.weight.data.to(torch.bfloat16)
                if mod.bias is not None:
                    mod.bias.data = mod.bias.data.to(torch.bfloat16)
    model.diffusion_head.use_hoist = bool({"hoist", "stepgraph"} & set(levers))        # the hoist: pair cond + DiT pair logits + all atom-encoder/decoder statics, once per trajectory
    model.diffusion_head.use_step_graph = "stepgraph" in levers
    model.sample_batch = SAMPLE_BATCH_ALL if "sbatch" in levers else 0                    # the sample-batched sampler (xfold/alphafold3.py _sample_diffusion_batched)
    model._atom_window_state = enable_atom_window(model) if "atom_window" in levers else None   # the atom transformers on the window kernels ("on" or the step-aside word)
    model._token_agg_state = enable_token_agg(model) if "token_agg" in levers else None         # the encoder's token aggregation kernel: "on" or the word it stepped aside with
    model._atom_rows_state = enable_atom_rows(model, levers) if "atom_rows" in levers else None   # the rows-only atom transformer: "on" | needs_atom_window
    model.batched_prologue = "prologue" in levers and bool(getattr(model, "sample_batch", 0))    # lever 'prologue': the batched sampler's per-step prologue over the sample axis
    model._prologue_state = ("on" if model.batched_prologue else "needs_sbatch") if "prologue" in levers else None
    model._sbatch_counts = {"batched_calls": 0, "single_calls": 0, "trajectories": 0, "chunk": None, "rng": None} if "sbatch" in levers else None   # its census (forward.json per item)
    if "compile" in levers or compile:
        enable_compile(model, mode=compile if isinstance(compile, str) else "max-autotune-no-cudagraphs")
    kl = [l for l in levers if l in _KERNEL_LEVERS]
    ml = [l for l in levers if l in _MSA_LEVERS]
    if kl or ml:
        import af3_kernels as K
        K.enable(kl)
        if hasattr(K, "set_graph_cells"):                                                    # lever apb: the diffusion transformer's attention asks the provider's graph-replay cells when the sampler
            K.set_graph_cells("stepgraph" in levers)                                           # captures whole steps (stepgraph), its eager cells under the hoist alone
        model._af3t_kernels = K                                                            # the kit's kernel census (af3t_msa's levers join it: one census per process)
    if ml or "af3t_msa" in sys.modules:                                                    # the MSA module's levers; a model built without them in a process that had them restores the stock classes
        import af3t_msa as M
        M.enable(ml)
        model._af3t_msa = M if ml else None
    model._af3t_levers = levers
    return model


COMPILE_FALLBACKS = {}   # name -> "<ExceptionType>: <message>": the compiled callables that raised when called and run EAGERLY for the rest of the process


def _is_oom(e):
    types_ = tuple(t for t in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)) if isinstance(t, type))
    return (bool(types_) and isinstance(e, types_)) or "CUDA out of memory" in str(e)


def _guarded(name, eager, compiled):
    """The compiled callable, falling back BY NAME: a failure raised through the compiled call (the backend cannot compile or launch it on
    this stack) records COMPILE_FALLBACKS[name] and the eager callable serves this and every later call; an out-of-memory propagates."""
    state = [compiled]
    def call(*args, **kwargs):
        f = state[0]
        if f is eager:
            return eager(*args, **kwargs)
        try:
            return f(*args, **kwargs)
        except Exception as e:
            if _is_oom(e):
                raise
            state[0] = eager
            COMPILE_FALLBACKS[name] = "%s: %s" % (type(e).__name__, (str(e).splitlines() or [""])[0][:160])
            print("[af3_torch_api] compile: %s raised %s -> eager for the rest of the process" % (name, COMPILE_FALLBACKS[name]), file=sys.stderr, flush=True)
            return eager(*args, **kwargs)
    return call


def enable_compile(model, mode="max-autotune-no-cudagraphs"):
    """torch.compile the per-step glue of the diffusion module: the 3+3 atom-transformer cross-attention + transition blocks (encoder/decoder),
    the 24 token-transformer transition blocks, and the noise-dependent single conditioning. Kernel levers/graphs are untouched; with
    'stepgraph' the compiled kernels are captured inside the whole-step CUDA graph (compile happens during the graph warm-up calls).
    Every compiled callable is guarded (_guarded): one that cannot compile or launch here runs eagerly, named in COMPILE_FALLBACKS."""
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    DHc = model.diffusion_head; n = 0
    for tname, tr in (("enc", DHc.atom_cross_att_encoder.atom_transformer_encoder), ("dec", DHc.atom_cross_att_decoder.atom_transformer_decoder)):
        for i, blk in enumerate(list(tr.cross_attention) + list(tr.transition_block)):
            blk._forward_eager = blk.forward                     # the uncompiled bound method (lever 'atom_rows' compiles its own rows variant from it)
            blk.forward = _guarded("%s.%d" % (tname, i), blk.forward, torch.compile(blk.forward, mode=mode, dynamic=False)); n += 1
    for i, blk in enumerate(DHc.transformer.transition_block):
        blk.forward = _guarded("dit_transition.%d" % i, blk.forward, torch.compile(blk.forward, mode=mode, dynamic=False)); n += 1
    DHc._single_conditioning = _guarded("single_conditioning", DHc._single_conditioning, torch.compile(DHc._single_conditioning, mode=mode, dynamic=False)); n += 1
    model._af3t_compiled = (mode, n)
    model._af3t_compile_fallbacks = COMPILE_FALLBACKS
    return n


@contextlib.contextmanager
def inference(dtype=torch.bfloat16):
    """inference_mode + CUDA autocast(bf16): the kit's numerics class, matching AlphaFold 3's JAX reference (bf16 activations, fp32 accum/LN/softmax)."""
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        yield


def batch_from_features(feats: dict, device="cuda"):
    """dict of numpy/torch numeric AF3 features (alphafold3.data.featurisation.featurise_input()[i], invalid-typed feats removed)
    -> dict of device tensors (float64 -> float32 like AF3's ModelRunner). Pass the result to every run_* function."""
    out = {}
    for k, v in feats.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind not in "fiub":
                continue
            t = torch.as_tensor(v)
        elif torch.is_tensor(v):
            t = v
        else:
            continue
        if t.dtype == torch.float64:
            t = t.float()
        out[k] = t.to(device)
    return out


def batch_from_npz(path, device="cuda"):
    return batch_from_features(dict(np.load(path)), device)


def _B(model, batch):
    """feature dict -> xfold Batch (applies the OF3 ref_element shift once). Cached on the dict object."""
    if isinstance(batch, feat_batch.Batch):
        return batch
    b = batch.get("_af3t_batch_obj")
    if b is None:
        b = model.prep_batch({k: v for k, v in batch.items() if not k.startswith("_af3t")})
        batch["_af3t_batch_obj"] = b
    return b


def run_target_feat(model, batch):
    return model.create_target_feat_embedding(_B(model, batch)).float()


def run_trunk(model, batch, num_recycles=None, target_feat=None, return_all=False):
    """num_recycles+1 Evoformer passes from zero prev (AF3 semantics). Returns embeddings dict with fp32 'single' [N,384],
    'pair' [N,N,128], 'target_feat' [N,447]."""
    b = _B(model, batch)
    n_iter = (model.num_recycles if num_recycles is None else num_recycles) + 1
    tf = run_target_feat(model, b) if target_feat is None else target_feat
    N = tf.shape[0]
    emb = {"pair": torch.zeros(N, N, model.evoformer.pair_channel, device=tf.device), "single": torch.zeros(N, model.evoformer.seq_channel, device=tf.device), "target_feat": tf}
    traj = []
    for _ in range(n_iter):
        e = model.evoformer(batch=b, prev=emb, target_feat=tf)
        emb = {"pair": e["pair"].float(), "single": e["single"].float(), "target_feat": tf}
        if return_all:
            traj.append(emb)
    return traj if return_all else emb


def run_distogram(model, batch, embeddings):
    return model.distogram_head(_B(model, batch), embeddings)


def denoise(model, batch, embeddings, positions_noisy, noise_level):
    """One denoiser call (DiffusionHead). Uses the whole-step CUDA graph when the model was built with 'stepgraph'
    (call prime_diffusion() once per new embeddings first; run_diffusion does this for you)."""
    DHm = model.diffusion_head; b = _B(model, batch)
    if getattr(DHm, "use_step_graph", False):
        return DHm.forward_graphed(positions_noisy, noise_level, b, embeddings, True)
    return DHm(positions_noisy=positions_noisy, noise_level=noise_level, batch=b, embeddings=embeddings, use_conditioning=True)


def prime_diffusion(model, batch, embeddings):
    """(Re)compute the hoisted step-invariant conditioning for these embeddings (needed once per trajectory when hoist/stepgraph on)."""
    DHm = model.diffusion_head
    if getattr(DHm, "use_hoist", False) or getattr(DHm, "use_step_graph", False):
        DHm.prime_static(_B(model, batch), embeddings)


def run_diffusion(model, batch, embeddings, seed=None, ext=None, steps=None, num_samples=None):
    """AF3 sampler (200 steps, gamma_0=0.8, gamma_min=1.0, noise_scale=1.003, step_scale=1.5, random augmentation per step).
    seed -> torch RNG (torch.manual_seed(seed)); ext -> dict(init_pos [N,24,3], noise [T,N,24,3], rot [T,3,3], trans [T,3]) external
    randomness. Returns positions [num_samples, N, 24, 3] fp32."""
    b = _B(model, batch)
    steps = steps or model.diffusion_steps
    if ext is None:
        if seed is not None:
            torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))
        old = (model.diffusion_steps, model.num_samples)
        model.diffusion_steps = steps; model.num_samples = num_samples or model.num_samples
        try:
            return model._sample_diffusion(b, embeddings)["atom_positions"].float()
        finally:
            model.diffusion_steps, model.num_samples = old
    mask = b.predicted_structure_info.atom_mask
    dev = mask.device
    nl = dh.noise_schedule(torch.linspace(0, 1, steps + 1, device=dev, dtype=torch.float32))
    prime_diffusion(model, b, embeddings)
    pos = ext["init_pos"].to(dev) * nl[0]
    for s in range(steps):
        noise_level_prev, noise_level = nl[s], nl[s + 1]
        center = mask_mean(mask[..., None], pos, dim=(-2, -3), keepdim=True, eps=1e-6)
        with torch.autocast(device_type=pos.device.type, enabled=False):      # the rigid transform of absolute coordinates stays fp32 (dh.random_augmentation's rule)
            pos = (torch.einsum('...i,ij->...j', (pos - center).float(), ext["rot"][s].to(dev, torch.float32)) + ext["trans"][s].to(dev, torch.float32)) * mask[..., None]
        gamma = 0.8 * (noise_level > 1.0)
        t_hat = noise_level_prev * (1 + gamma)
        noise_scale = 1.003 * torch.sqrt(torch.clamp(t_hat ** 2 - noise_level_prev ** 2, min=0.0))
        pos_noisy = pos + noise_scale * ext["noise"][s].to(dev)
        den = denoise(model, b, embeddings, pos_noisy, t_hat)
        pos = pos_noisy + 1.5 * (noise_level - t_hat) * ((pos_noisy - den) / t_hat)
    return pos[None].float()


def run_confidence(model, batch, embeddings, positions):
    """ConfidenceHead on one sample's dense atom positions [N,24,3]. Returns dict of fp32 tensors (predicted_lddt [N,24] 0-100,
    full_pae [N,N], full_pde, average_pde, tmscore_adjusted_pae_global/interface [N,N], predicted_experimentally_resolved ...).
    pTM/ipTM: alphafold3.model.confidences.predicted_tm_score(tmscore_adjusted_pae_*[:n,:n], pair_mask=frames_mask, asym_id)."""
    b = _B(model, batch)
    out = model.confidence_head(dense_atom_positions=positions, embeddings=embeddings, seq_mask=b.token_features.mask,
                                token_atoms_to_pseudo_beta=b.pseudo_beta_info.token_atoms_to_pseudo_beta, asym_id=b.token_features.asym_id)
    return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}


def forward(model, batch, seed=1):
    """Full forward like AF3 Model.__call__ with 1 diffusion sample: trunk -> sampler (torch RNG) -> confidence -> distogram."""
    b = _B(model, batch)
    emb = run_trunk(model, b)
    xyz = run_diffusion(model, b, emb, seed=seed)
    conf = run_confidence(model, b, emb, xyz[0])
    dg = run_distogram(model, b, emb)
    return dict(embeddings=emb, atom_positions=xyz, confidence=conf, distogram=dg)
