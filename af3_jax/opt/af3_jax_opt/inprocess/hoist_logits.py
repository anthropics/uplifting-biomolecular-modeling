"""HOIST_LOGITS — the FlashPairformer hoist's per-block pair logits kept RESIDENT and indexed in place by the 24 diffusion-transformer
blocks, instead of being re-sliced through two nested ``hk.layer_stack`` per-layer inputs at every denoising step (kind=datapath; strategy
id ``LOCAL.af3_jax.hoist_logits``, a refinement of this kit's FPF_HOIST: what is hoisted is kept where the loop reads it cheapest).

Without it (opt/forward/flashpairformer/af3_flashpairformer/diffusion_hoist.py ``HoistTransformer``): mode='precompute' projects the 24
blocks' pair logits ONCE per sample call into one f32 array [n_super_blocks, super_block_size, heads, N, N] (24·16·N²·4 B = 2.3 GB at
1,216 tokens); mode='step' (traced inside the sampler's hk.scan, the samples vmapped) hands it to
``hk.experimental.layer_stack(n_super, with_per_layer_inputs=True)`` whose body hands its [sbs, H, N, N] slice to ``layer_stack(sbs, ...)``,
and XLA materialises both per-layer slices as device copies at every block of every step.

With it: (1) precompute stores the stacked logits in STORE_DTYPE (``float32`` as shipped: the hoist's own values, so the lever is
value-identical to the hoist alone; ``bfloat16`` halves the resident bytes and every later read at the price of one bf16 rounding of the
bias — tolerance class, kept in the code, not the default); (2) step gives the two layer_stacks the LAYER INDEX as their per-layer input
(``jnp.arange``) and each block reads its logits with ONE ``lax.dynamic_index_in_dim(flat, i·sbs+j)`` of the resident [24, H, N, N] view —
the super-block operand copy disappears (XLA passes the loop-invariant array through both while loops by reference); the block-level slice
that feeds the attention kernel remains (a custom-call operand). Everything else in the step is the hoist's own lines (same Haiku scopes and
parameter names, same ``DT.self_attention`` / ``DT.transition_block`` calls — DATTN, TTR, SAMPLER_BF16 compose unchanged). Engages only
where the hoist runs (FPF_HOIST on, of3 weights layout); with the hoist absent or off the lever steps aside BY NAME (``hlog=skipped``),
never silently.

Switch: ``AF3_JAX_HOIST_LOGITS=1`` in the model process's environment (modes.TREE_LEVER_ENV; declared in stack.LEVER_SWITCH_ENV); installed
by fpf_launch AFTER the add-on import (it rebinds ``af3_flashpairformer.diffusion_hoist.HoistTransformer`` to a subclass;
INSTALL_AFTER_ADDON). ``MODEL_OPT_LEVERS_OFF=HOIST_LOGITS`` removes it.
Prints: the launcher's SERVED line carries ``hlog=<step traces>|off|skipped hlog_dtype=<float32|bfloat16|none>`` from ``report()`` ->
{"installed", "dtype", "traced": {"precompute", "step"}, "skipped"}.
"""
import os
import sys

ENV_SWITCH = "AF3_JAX_HOIST_LOGITS"
REBINDS = ("af3_flashpairformer.diffusion_hoist:HoistTransformer",)   # what install() rebinds (module:attribute) — a subclass of the add-on's hoist transformer, hence AFTER the add-on import (INSTALL_AFTER_ADDON); the launcher's INSTALL_ORDER table (fpf_launch.py) and tests/test_install_order.py read it
INSTALL_AFTER_ADDON = True                                                # fpf_launch: install after `import af3_flashpairformer` (the hoist module must exist)
STORE_DTYPE = "float32"                                                   # "float32" (shipped) = index-only: the hoist's own f32 logits indexed in place — value-identical to the hoist without this lever.
                                                                          # "bfloat16" = resident bf16 logits (half the bytes and reads; one bf16 rounding of the bias: tolerance class) — kept in the code, NOT the default.
_STATE = {"installed": False, "dtype": None, "traced": {"precompute": 0, "step": 0}, "skipped": "", "stock": {}}


def wanted(environ=os.environ) -> bool:
    return environ.get(ENV_SWITCH, "") == "1"


def _store_dtype():
    import jax.numpy as jnp
    return {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[STORE_DTYPE]


def _make_class(DHM):
    """``HoistTransformer`` subclassed (a Haiku module method must be bound at class creation to run inside the module's name scope — the
    parameters stay ``.../transformer/__layer_stack_no_per_layer/...`` as stock): precompute stores ``STORE_DTYPE`` logits, step indexes the
    resident array in place; every other line is the hoist's own (diffusion_hoist.py)."""
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from alphafold3.model.components import haiku_modules as hm
    from alphafold3.model.network import diffusion_transformer as DT
    LS = DHM.LS
    Stock = DHM.HoistTransformer
    dt = _store_dtype()
    _STATE["dtype"] = jnp.dtype(dt).name

    class HoistTransformer(Stock):
      """diffusion_hoist.HoistTransformer with resident (STORE_DTYPE, flat-indexed) per-block pair logits (HOIST_LOGITS)."""

      def __call__(self, act, mask, single_cond, pair_cond, mode="stock", pair_logits_pre=None):
        of3 = getattr(self.global_config, "of3_weights", False)
        if mode == "stock" or not of3 or (mode == "step" and pair_logits_pre is None):
            return super().__call__(act, mask, single_cond, pair_cond, mode=mode, pair_logits_pre=pair_logits_pre)   # the hoist's own answer (incl. precompute -> None off the of3 layout)
        nsb = self.config.num_blocks // self.config.super_block_size
        sbs = self.config.super_block_size
        if mode == "precompute":
            def blk(c):
                pair_act = hm.LayerNorm(name="pair_input_layer_norm", use_fast_variance=False, create_offset=False)(pair_cond)
                lg = hm.Linear(self.config.attention.num_head, name="pair_logits_projection")(pair_act)
                return c, jnp.transpose(lg, [2, 0, 1]).astype(dt)                  # LEVER (1): resident dtype — the stock f32 projection, stored as STORE_DTYPE
            def sb(c):
                return hk.experimental.layer_stack(sbs, with_per_layer_inputs=True, name=LS)(blk)(c)
            _, logits = hk.experimental.layer_stack(nsb, with_per_layer_inputs=True, name=LS)(sb)(jnp.zeros((), jnp.float32))
            _STATE["traced"]["precompute"] += 1
            return logits                                                            # [nsb, sbs, H, N, N]
        assert mode == "step"
        flat = jnp.reshape(pair_logits_pre, (nsb * sbs,) + tuple(pair_logits_pre.shape[2:]))   # a view of the resident array: [24, H, N, N]

        def super_block(act, i):
            def block(act, j):
                lg = jax.lax.dynamic_index_in_dim(flat, i * sbs + j, keepdims=False)   # LEVER (2): the block's logits indexed in place — no per-super-block operand copy
                act += DT.self_attention(act, mask, lg, self.config.attention, self.global_config, single_cond, name=self.name)
                act += DT.transition_block(act, self.config.num_intermediate_factor, self.global_config, single_cond, name=self.name)
                return act, None
            act, _ = hk.experimental.layer_stack(sbs, with_per_layer_inputs=True, name=LS)(block)(act, jnp.arange(sbs, dtype=jnp.int32))
            return act, None
        act, _ = hk.experimental.layer_stack(nsb, with_per_layer_inputs=True, name=LS)(super_block)(act, jnp.arange(nsb, dtype=jnp.int32))
        _STATE["traced"]["step"] += 1
        return act

    HoistTransformer.__module__ = Stock.__module__
    return HoistTransformer


def install() -> bool:
    """Rebind the hoist transformer's __call__ (idempotent). Returns whether the lever is installed; when the hoist module is absent (FPF_HOIST
    ablated / the add-on not imported) the lever is NOT installed and says why (report()['skipped'])."""
    if _STATE["installed"]:
        return True
    DHM = sys.modules.get("af3_flashpairformer.diffusion_hoist")
    if DHM is None:
        try:
            from af3_flashpairformer import diffusion_hoist as DHM                   # importable once fpf_launch put the add-on on sys.path
        except Exception as e:                                                       # noqa: BLE001 — named, not raised: the lever steps aside by name
            _STATE["skipped"] = f"hoist_module_absent({type(e).__name__})"
            return False
    if not DHM._STOCK:                                                               # the hoist itself is not installed (AF3_DIFFUSION_HOIST unset = FPF_HOIST ablated): nothing to index
        _STATE["skipped"] = "hoist_off"
        return False
    _STATE["stock"] = {"cls": DHM.HoistTransformer, "module": DHM}
    DHM.HoistTransformer = _make_class(DHM)                                          # HoistDiffusionHead resolves the module-global name at trace time
    _STATE["installed"] = True
    return True


def uninstall() -> None:
    if _STATE["installed"]:
        _STATE["stock"]["module"].HoistTransformer = _STATE["stock"]["cls"]
        _STATE["installed"] = False


def report() -> dict:
    return {"installed": _STATE["installed"], "dtype": _STATE["dtype"], "traced": dict(_STATE["traced"]), "skipped": _STATE["skipped"]}
