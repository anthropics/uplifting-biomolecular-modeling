# The core cells' import hook (the `fast` mode's `trimul_v4` / `triatt_block` / `pair_transition` pair-track levers, its `rollout_bf16`,
# `dit_attn`, `dit_glue`, `token_agg`, `atom_window` and `atom_hoist` sampler levers — the last also the `exact` line's; the fast and exact lines' ENTRY hook:
# first on PYTHONPATH, executed by openfold3_ob0_opt's activation or, on the PYTHONPATH route, by the interpreter at start-up). After
# `openfold3.projects.of3_all_atom.model` executes -> openfold3_ob0_opt.cells.pairfused.install() when OPENFOLD3_OB0_OPT_PAIR names at least one lever and
# openfold3_ob0_opt.cells.rollout.install() when OPENFOLD3_OB0_OPT_ROLLOUT=bf16 (the levers' switches; the resolver exports them on the fast line) and dit_attn.install() when
# OPENFOLD3_OB0_OPT_DIT names its kernel, dit_glue / token_agg / atom_window / atom_hoist / apb_trunk / templ_embed / castcache / apb_hoist / trunk_graph / post_release / tuner_guard .install() on OPENFOLD3_OB0_OPT_DIT_GLUE / _TOKEN_AGG / _ATOM_WINDOW / _ATOM_HOIST / _APB_TRUNK / _TEMPL_EMBED / _CASTCACHE / _APB_HOIST / _TRUNK_GRAPH / _POST_RELEASE (modes.py says which line exports
# which switch; an unset switch installs nothing). The module spec is obtained from the REST of sys.meta_path (the chained trunk-kernels
# hook's finder and the package's instance counter target the same module and do the same), so every finder wraps the module's exec_module
# exactly once; this hook's install runs right after the module body, inside the chained hooks' wrappers — the trunk-kernels add-on's routes
# (installed after) become the named fallbacks of the cells (openfold3_ob0_opt/cells/pairfused.py docstring).
# OPENFOLD3_OB0_OPT_PAIR_CHAIN=<dir> names the next hook directory (the trunk-kernels add-on's of3t_hook) whose sitecustomize.py is executed next —
# the chain pairfused > trunk_kernels > fast_inference, each link exported by the package's resolver (modes.CHAIN_ENV). A failed hook raises.
import importlib.abc
import os
import sys

_TARGET = "openfold3.projects.of3_all_atom.model"
_ON = any(bool((os.environ.get(k) or "").strip()) for k in ("OPENFOLD3_OB0_OPT_PAIR", "OPENFOLD3_OB0_OPT_DIT", "OPENFOLD3_OB0_OPT_ROLLOUT",
                                                            "OPENFOLD3_OB0_OPT_DIT_GLUE", "OPENFOLD3_OB0_OPT_TOKEN_AGG", "OPENFOLD3_OB0_OPT_ATOM_HOIST",
                                                            "OPENFOLD3_OB0_OPT_ATOM_WINDOW", "OPENFOLD3_OB0_OPT_APB_TRUNK", "OPENFOLD3_OB0_OPT_CASTCACHE", "OPENFOLD3_OB0_OPT_EXACTLN", "OPENFOLD3_OB0_OPT_LN_PROVIDER",
                                                            "OPENFOLD3_OB0_OPT_APB_HOIST", "OPENFOLD3_OB0_OPT_TRUNK_GRAPH", "OPENFOLD3_OB0_OPT_POST_RELEASE", "OPENFOLD3_OB0_OPT_TUNER_GUARD", "OPENFOLD3_OB0_OPT_TEMPL_EMBED",
                                                            "OPENFOLD3_OB0_OPT_POSTFWD_MEM", "OPENFOLD3_OB0_OPT_LOADER_WORKERS", "OPENFOLD3_OB0_OPT_TRIATT_EXACT", "OPENFOLD3_OB0_OPT_TRIMUL_EXACT", "OPENFOLD3_OB0_OPT_TRIMUL_EXACT_FORM",
                                                             "OPENFOLD3_OB0_OPT_TRANSITION_EXACT"))


class _PostImportFinder(importlib.abc.MetaPathFinder):
    _target = _TARGET                     # read by openfold3_ob0_opt.hooks.installed()

    def __init__(self):
        self._done = False
        self._busy = False

    def find_spec(self, name, path=None, target=None):
        if name != _TARGET or self._done or self._busy:
            return None
        self._busy = True
        try:
            spec = None
            for finder in sys.meta_path:                 # the rest of the meta path (the chained hooks' finders wrap the same module; re-entrant calls return None above)
                if finder is self:
                    continue
                try:
                    spec = finder.find_spec(name, path, target)
                except Exception:
                    spec = None
                if spec is not None:
                    break
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        self._done = True
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec):
            _orig(module)
            from openfold3_ob0_opt.cells import (apb_hoist, apb_trunk, atom_hoist, atom_window, castcache, dit_attn, dit_glue, exactln, ln_provider, loader_workers, pairfused, post_release, postfwd_mem, rollout, templ_embed, token_agg, transition_exact, triatt_exact, trimul_exact, trunk_graph, tuner_guard)
            pairfused.install()                               # the pair track's cells (OPENFOLD3_OB0_OPT_PAIR)
            apb_trunk.install()                               # the pairformer single track's attention with pair bias (OPENFOLD3_OB0_OPT_APB_TRUNK)
            templ_embed.install()                             # the template embedder's fused feature embedding + mean/relu/linear_t tail (OPENFOLD3_OB0_OPT_TEMPL_EMBED)
            rollout.install()                                 # the diffusion denoiser step under bf16 autocast, its atom-attention modules fp32, above the token gate (OPENFOLD3_OB0_OPT_ROLLOUT)
            dit_attn.install()                                # the diffusion transformer's attention (OPENFOLD3_OB0_OPT_DIT)
            dit_glue.install()                                # the token diffusion transformer block as one fused schedule (OPENFOLD3_OB0_OPT_DIT_GLUE)
            token_agg.install()                               # the atom -> token aggregation kernel (OPENFOLD3_OB0_OPT_TOKEN_AGG)
            atom_window.install()                             # the sampler's atom-attention blocks on the fused window kernels (OPENFOLD3_OB0_OPT_ATOM_WINDOW)
            atom_hoist.install()                              # the atom path's per-rollout memo, exact class (OPENFOLD3_OB0_OPT_ATOM_HOIST)
            castcache.install()                               # memoised bf16 weight copies of the Linear primitive (the engine's LayerNorm upcasts: named skipped), exact class (OPENFOLD3_OB0_OPT_CASTCACHE)
            exactln.install()                                 # the LayerNorm primitive (upcast dialect) on the core's LayerNorm provider row exactln (bitwise ATen; the bf16 branch = the widen form, cast fused), exact class (OPENFOLD3_OB0_OPT_EXACTLN)
            ln_provider.install()                             # the same binding asked with the fast / big line's tier word, tolerance class (OPENFOLD3_OB0_OPT_LN_PROVIDER + OPENFOLD3_OB0_OPT_LN_TIER)
            apb_hoist.install()                               # AttentionPairBias' key-mask bias memoised per stack call, exact class (OPENFOLD3_OB0_OPT_APB_HOIST)
            trunk_graph.install()                             # PairFormerStack captured into a CUDA graph and replayed, exact class (OPENFOLD3_OB0_OPT_TRUNK_GRAPH)
            post_release.install()                            # the item-boundary memory release of the graphed lines at large token counts, exact class (OPENFOLD3_OB0_OPT_POST_RELEASE)
            tuner_guard.install()                             # the engine's chunk-size tuner answers 'changed' for structurally different argument records (one process, mixed token counts), exact class (OPENFOLD3_OB0_OPT_TUNER_GUARD)
            postfwd_mem.install()                             # the runner's confidence scoring after the forward per (sample, row block), exact class (OPENFOLD3_OB0_OPT_POSTFWD_MEM)
            loader_workers.install()                          # the predict DataLoader's worker count capped at the item count, exact class (OPENFOLD3_OB0_OPT_LOADER_WORKERS)
            triatt_exact.install()                            # the exact line's cuEquivariance triangle-attention call on the core's provider, word exact (bit-proven per class), exact class (OPENFOLD3_OB0_OPT_TRIATT_EXACT)
            trimul_exact.install()                            # the exact line's triangle multiplicative updates on the core's trimul provider, word exact (bit-proven per class), exact class (OPENFOLD3_OB0_OPT_TRIMUL_EXACT) (+ OPENFOLD3_OB0_OPT_TRIMUL_EXACT_FORM: its module-statement classes on the exact tier's form of3_module -> row of3_form, cells/trimul_form.py)
            transition_exact.install()                        # the exact line's pair-stack SwiGLU transitions after their own LayerNorm as one kernel with the engine's arithmetic (the core's transition provider row v1), exact class (OPENFOLD3_OB0_OPT_TRANSITION_EXACT)

        spec.loader.exec_module = exec_module
        return spec


if _ON:
    sys.meta_path.insert(0, _PostImportFinder())

_chain = os.environ.get("OPENFOLD3_OB0_OPT_PAIR_CHAIN", "")
if _chain:
    _hook = os.path.join(_chain, "sitecustomize.py")
    if not os.path.isfile(_hook):
        raise RuntimeError(f"[openfold3_ob0-opt/pairfused] OPENFOLD3_OB0_OPT_PAIR_CHAIN={_chain!r}: no sitecustomize.py there")
    if _chain not in sys.path:
        sys.path.insert(1, _chain)
    exec(compile(open(_hook).read(), _hook, "exec"), {"__name__": "openfold3_ob0_opt_hook_trunk_kernels", "__file__": _hook})
