#!/usr/bin/env python
"""forward.py — the model process: the kit's documented entry points under the torch venv (AF3_TORCH_PY), one process per `pred`.

    python forward.py --kit <opt/forward/af3t> --dtk-home <opt/forward/dtk> --params <file> --levers <csv or empty> --dtk 0|1
                      --item NAME=SEED=<batch.npz>=<result.npz> [--item ...] --report <forward.json> [--opt-core <dir> --route <csv kernel names>]
                      [--big <csv levers> --alloc expandable]

Per item, in order: ``build_model`` once (the kit's ``af3_torch_api.build_model(<checkpoint file>, levers=…)``: the module tree, the OF3-layout
parameters, the kit levers), the DTK swap when ``--dtk 1`` (``XfoldDTK`` below — the DTK add-on's in-model adapter over
``opt/forward/dtk`` dtk_modules over the core's routed ``dtk_kernels``, ``swap_in_dtk``), then per item (one per fold input and seed, in order) ``batch_from_npz`` → ``inference()`` → ``forward(model, batch, seed)`` and the result
arrays to ``result.npz`` in the kit's e2e form: ``atom_positions`` [S, N, 24, 3],
``conf_<key>`` (every confidence-head output), the distogram head's keys as they are (``contact_probs``, …), ``seed``. The report
names the levers the model carries (``model._af3t_levers`` — the stock proof of ``off``: an empty tuple and ``dtk`` false), the walls
(build, per-item forward — the first forward of a fast process carries the graph capture and, with ``compile``, the Inductor compile),
the per-item phase walls (``phase_s``: trunk, sampler, confidence head — the wrapper's PHASE line), the peak device memory and every item's outcome (ok or the named exception: total accounting; exit 1 when any item failed). Runs with
no AF3_TORCH_OPT* variable in its environment (stack.model_process_env). ``--route`` names the kernels served from the shared core's carried
copies (registry.KERNEL_ROUTES): routed and gated (``route_kernels``: opt_core.kernels.route + route_check) BEFORE the kit api is imported —
a refused route is exit 2 with the reason on the report, never the kit's own copy served instead; the report's ``kernel_routes`` names per
kernel the gate's verdict, the core copy, and the file the module was imported from in this process. Nothing here writes model parameters anywhere.
"""
import argparse
import json
import os
import sys
import time
import traceback
import types

_HERE = os.path.dirname(os.path.abspath(__file__))            # forward.py runs 3 ways (script, `af3_torch_opt.forward` import, spec_from_file_location);
if _HERE not in sys.path:                                      # only the script form gets this on sys.path for free — put it there ourselves so the bare
    sys.path.insert(0, _HERE)                                  # import below resolves in every mode
from forward_impl import (_load_rowpair_xfold, _reset_item_state, _forward_samples,  # moved: importable so a timing hook can
                          _clock, _stage, _run_trunk_prev_free)                        # wrap _forward_samples / _run_trunk_prev_free
import pipeline_stream as PS                                    # the streamed chain's hand-off markers (package levers prefetch / write_behind; standard library only)


RANK_TIMEOUT_S = 1800.0      # n_gpu > 1: the process group's collective timeout (s) — a rank silent past it fails the run by name (RankFailed)
ROWPAIR_BLOCK = None         # n_gpu > 1: the row grid's block B = the core's Layout.auto choice for (N, P)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kit", required=True, help="the af3t kit dir (opt/forward/af3t)")
    ap.add_argument("--dtk-home", required=True, help="the DTK dir (opt/forward/dtk)")
    ap.add_argument("--params", required=True, help="the pinned OpenFold3-preview2 checkpoint FILE (never a directory: refused by name, rc 2 — the kit's loader would take the first *.bin.zst of a directory)")
    ap.add_argument("--weights-sha256", default=None, help="the checkpoint's sha256 as the wrapper digested it (recorded in forward.json weights; bookkeeping only)")
    ap.add_argument("--weights-pinned", type=int, choices=(0, 1), default=None, help="1 = the digest is stock/PINS.json's pinned checkpoint (recorded; bookkeeping only)")
    ap.add_argument("--levers", default="", help="comma list of kit lever names ('' = the kit's eager set)")
    ap.add_argument("--dtk", type=int, default=0, help="1 = DTK FusedDiT replaces the diffusion token transformer")
    ap.add_argument("--fastnn", type=int, choices=(0, 1), default=0, help="1 = xfold's shipped Triton fastnn kernels (layer norm / dot-product attention / gated linear unit): the stock CLI's own --fastnn switch (run_alphafold.py:223-226), set before the model is built; the wrapper passes it under `--mode off | exact` unless `--nofastnn`")
    ap.add_argument("--num_recycles", type=int, default=None, help="default: the kit's build_model default")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--diffusion_steps", type=int, default=None)
    ap.add_argument("--item", action="append", default=[], help="NAME=SEED=<batch.npz>=<result.npz>; repeatable, run in order in this one process (SEED: the sampler seed of this item)")
    ap.add_argument("--report", required=True, help="forward.json: levers, dtk, walls, memory, per-item outcome")
    ap.add_argument("--big", default="", help="comma list of big memory levers in force (big.LEVER_ORDER): the model process acts on diff_free, and records per-item bucket / N facts; --alloc carries expandable_segments")
    ap.add_argument("--alloc", default="", help="CUDA caching-allocator policy exported before torch initialises CUDA (opt_core.mem.torch_alloc: 'expandable'); refused -> rc 2")
    ap.add_argument("--paircond-rows", type=int, default=0, help="big's paircond_chunk: the sampler's step-invariant pair conditioning evaluated per block of this many rows of the token grid (0 = whole, as stock; big.PAIRCOND_CHUNK_ROWS)")
    ap.add_argument("--graph-drop-tokens", type=int, default=0, help="big's graph_drop size gate: the step graph (lever stepgraph, built) is switched off for an item whose PADDED token count is at or above this (0 = no gate in this process: big.build_levers decided at build time)")
    ap.add_argument("--templates-declared", action="append", default=[], help="NAME=k: the fold input of item NAME declares k templates (the template census: reported per item beside the template slots featurised live)")
    ap.add_argument("--n-gpu", type=int, default=1, help="GPUs the pair stack is row-sharded over (1: this process on one device, the single-GPU path; >1: opt_core.mem.rowpair ranks spawned by this process, mode big — rowpair_xfold.py in every rank)")
    ap.add_argument("--opt-core", default=None, help="the directory holding the opt_core package (the wrapper passes the imported core's): on sys.path so --route can serve the core's carried kernel copies")
    ap.add_argument("--padding", default="none", help="the mode's token-padding policy (modes.MODE_PADDING): none | kernel_tile (recorded; the batch arrives padded, or at its own token count under none)")
    ap.add_argument("--stream-root", default=None, help="the pred's work dir when the wrapper runs the chain STREAMED (package levers prefetch / write_behind): per item this process waits for the featuriser's hand-off marker before it loads the batch (prefetch) and marks each result.npz complete for the writers the moment it is closed (write_behind) — pipeline_stream; absent = the sequential chain (every batch is already there, nobody is waiting)")
    ap.add_argument("--package-levers", default="", help="comma list of package levers composed on the kit's build in this process (registry.PACKAGE_LEVERS: template_dedupe, dev_scalars, tri_layout, ln_rows,attn_layout,gate_fuse, castcache, canonical_noise, prefetch, write_behind, autotune_cache, feat_par); the wrapper passes the mode's (modes.MODE_PACKAGE_LEVERS)")
    ap.add_argument("--route", default="", help="comma list of kernel names ROUTED to the shared core's carried copies (registry.KERNEL_ROUTES); each is gated by opt_core.kernels.route_check before the kit api is imported — refused: rc 2")
    return ap.parse_args(argv)



def _canonical_noise():
    """The package's canonical_noise module (same directory as this script)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import canonical_noise
    return canonical_noise


def _tri_layout():
    """The package's tri_layout module (same directory as this script)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import tri_layout
    return tri_layout


def _ln_rows():
    """The package's ln_rows module (same directory as this script)."""
    import ln_rows
    return ln_rows


def _attn_layout():
    """The package's attn_layout module (same directory as this script)."""
    import attn_layout
    return attn_layout


def _gate_fuse():
    """The package's gate_fuse module (same directory as this script)."""
    import gate_fuse
    return gate_fuse


def _castcache():
    """The package's castcache module (same directory as this script)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import castcache
    return castcache


def _dev_scalars():
    """The package's dev_scalars module (same directory as this script)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import dev_scalars
    return dev_scalars


def _template_dedupe():
    """The package's template_dedupe module (forward.py runs as a script: its own directory is the package's)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import template_dedupe
    return template_dedupe


def templates_declared_map(specs):
    """``--templates-declared NAME=k`` (repeatable) -> {NAME: k}."""
    out = {}
    for spec in specs or []:
        name, _, k = str(spec).rpartition("=")
        out[name] = int(k)
    return out



def template_slots_total(batch) -> int:
    """The number of template slots the featuriser laid (``template_atom_mask`` [T, N, 24] → T; 0 when the feature is absent)."""
    m = batch.get("template_atom_mask") if hasattr(batch, "get") else None
    return int(m.shape[0]) if m is not None and getattr(m, "ndim", 0) >= 1 else 0


def template_slots_live(batch) -> int:
    """The number of template slots whose atom mask has any atom (``template_atom_mask`` [T, N, 24], the fork's feature the model embeds:
    xfold/features.py Templates): 0 = every slot is padding / dummy."""
    m = batch.get("template_atom_mask") if hasattr(batch, "get") else None
    if m is None:
        return 0
    T = int(m.shape[0]) if getattr(m, "ndim", 0) >= 1 else 0
    if T == 0:
        return 0
    flat = m.reshape(T, -1)
    live = (flat != 0).any(1)
    return int(live.sum().item() if hasattr(live.sum(), "item") else live.sum())


def template_entries_live(batch) -> int:
    """The live templates in the units the fold input DECLARES them (cli.templates_declared: per chain): over every chain of the item
    (``asym_id``), the template slots whose atom mask has any atom on that chain's tokens, summed — a homodimer whose four templates all
    featurised counts 8 live of 8 declared (the featuriser's four slots each carry both chains' template atoms). Without ``asym_id`` in the
    batch the count falls back to the slot count (template_slots_live)."""
    m = batch.get("template_atom_mask") if hasattr(batch, "get") else None
    a = batch.get("asym_id") if hasattr(batch, "get") else None
    if m is None or a is None or getattr(m, "ndim", 0) != 3:
        return template_slots_live(batch)
    import numpy as np
    m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
    a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    tok_live = (m != 0).any(-1)                                   # [T, N]: slot t has an atom on token n
    n = 0
    for chain in np.unique(a[a != 0]):                              # asym_id 0 = padding tokens
        n += int(tok_live[:, a == chain].any(-1).sum())
    return n


def kit_sys_path(kit, dtk_home):
    """The kit's own import recipe (af3_torch_api.py docstring) + the DTK dir (dtk_modules.py; `dtk_kernels` is the core's routed module)."""
    return [os.path.join(kit, "af3_torch"), os.path.join(kit, "kernels"), os.path.join(kit, "kernels", "third_party"), dtk_home]


ROUTE_KEYS = ("core_copy", "resolved", "routed", "already_imported", "runtime_imports", "exports", "differing", "missing")


def route_kernels(names, opt_core_dir=None):
    """Route each kernel name to the shared core's carried copy (opt_core.kernels.route) and gate it (route_check: the bytes that resolve
    in THIS process == the core's sums file, every export present, nothing imported before the gate). Returns ({name: record}, error or
    None); a record: ok, reason, version, core_copy, resolved, … . Runs before the kit api is imported — the kit's own third_party copies
    stay on sys.path behind the route (carried bytes, shadowed for exactly these names)."""
    if not names:
        return {}, None
    if opt_core_dir and opt_core_dir not in sys.path:
        sys.path.insert(0, opt_core_dir)
    try:
        from opt_core import kernels as CK
    except Exception as e:                                   # noqa: BLE001 — named on the report, rc 2
        return {n: {"ok": False, "reason": f"opt_core not importable: {e!r}"} for n in names}, f"--route {','.join(names)}: opt_core is not importable ({e!r}); --opt-core names its directory"
    recs, bad = {}, []
    for name in names:
        try:
            CK.route(name)
            g = CK.route_check(name)
            rec = {"ok": bool(g.ok), "reason": g.reason, "version": CK.sums(name).get("version"), **{k: g.details.get(k) for k in ROUTE_KEYS}}
        except Exception as e:                               # noqa: BLE001 — an unknown name, a missing sums file: refused by name
            rec = {"ok": False, "reason": f"{type(e).__name__}: {e}"[:400]}
        recs[name] = rec
        print(f"[forward] kernel_route name={name} ok={int(rec['ok'])} version={rec.get('version')} resolved={rec.get('resolved')}" + ("" if rec["ok"] else f" reason={rec['reason']}"), flush=True)
        if not rec["ok"]:
            bad.append(name)
    err = ("kernel route refused: " + "; ".join(f"{n}: {recs[n]['reason']}" for n in bad)) if bad else None
    return recs, err


def _autotune_cache_on():
    """Package lever autotune_cache: switch Triton's autotune result cache on for this process BEFORE any kernel module is imported (an
    Autotuner reads ``triton.knobs.autotuning.cache`` when it is declared) and count, per autotune, whether the timings came from the disk
    cache (hits) or were timed in this process and stored (benched) — Autotuner.check_disk_cache's own return value. → {"knob": 1, "census": {...}}
    or {"knob": 0, "reason": ...} when this Triton has no such cache (the lever then steps aside by name)."""
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    try:
        import triton.knobs as knobs
        from triton.runtime.autotuner import Autotuner
        knobs.autotuning.cache = True
        check = Autotuner.check_disk_cache
    except Exception as e:  # noqa: BLE001 — an older Triton without the knob / the disk cache: named, not applied
        return {"knob": 0, "reason": f"triton_knob_absent:{type(e).__name__}"}
    census = {"knob": 1, "hits": 0, "benched": 0, "dir": str(getattr(knobs.cache, "dir", None) or os.environ.get("TRITON_CACHE_DIR") or "")}
    if not getattr(Autotuner, "_af3t_counted", False):
        def counted(self, *args, **kwargs):
            hit = check(self, *args, **kwargs)
            census["hits" if hit else "benched"] += 1
            return hit
        Autotuner.check_disk_cache = counted; Autotuner._af3t_counted = True
    return {"knob": 1, "census": census}


def imported_from(names):
    """Where each routed name was imported from in this process (the module's __file__), or None when nothing imported it (the eager set)."""
    return {n: getattr(sys.modules.get(n), "__file__", None) for n in names}


# ---- the DTK in-model adapter: xfold's DiffusionTransformer (24 blocks) served by the fused DTK modules (opt/forward/dtk), weights from the live module
def _xfold_dtk_class(torch, M):
    class XfoldDTK:
        """DTK fused replacement for xfold DiffusionTransformer.forward (24 blocks), weights taken from the live module."""
        def __init__(self, tr):
            self.tr = tr
            W = types.SimpleNamespace(); W.h = tr.num_head; W.c = tr.c_act; W.c_s = tr.c_single_cond; W.d = tr.c_act // tr.num_head; W.blocks = []
            hid = 2 * tr.c_act
            for i in range(tr.num_blocks):
                sa = tr.self_attention[i]; tb = tr.transition_block[i]
                W.blocks.append(dict(
                    ln_s_w1=sa.adaptive_layernorm.single_cond_layer_norm.weight, W_scale1=sa.adaptive_layernorm.single_cond_scale.weight,
                    b_scale1=sa.adaptive_layernorm.single_cond_scale.bias, W_shift1=sa.adaptive_layernorm.single_cond_bias.weight,
                    Wq=sa.q_projection.weight, bq=sa.q_projection.bias, Wk=sa.k_projection.weight, Wv=sa.v_projection.weight,
                    Wg=sa.gating_query.weight, Wo=sa.adaptive_zero_init.transition2.weight,
                    W_zero1=sa.adaptive_zero_init.adaptive_zero_cond.weight, b_zero1=sa.adaptive_zero_init.adaptive_zero_cond.bias,
                    ln_s_w2=tb.adaptive_layernorm.single_cond_layer_norm.weight, W_scale2=tb.adaptive_layernorm.single_cond_scale.weight,
                    b_scale2=tb.adaptive_layernorm.single_cond_scale.bias, W_shift2=tb.adaptive_layernorm.single_cond_bias.weight,
                    W1=tb.transition1.weight[:hid], W2=tb.transition1.weight[hid:], W3=tb.adaptive_zero_init.transition2.weight,
                    W_zero2=tb.adaptive_zero_init.adaptive_zero_cond.weight, b_zero2=tb.adaptive_zero_init.adaptive_zero_cond.bias))
            with torch.no_grad():
                self.fused = M.FusedDiT(W, torch.bfloat16, "xfold")
            self.fused.apb_word = _provider_tier_word()               # the blocks' batched attention through the shared core's pair-bias attention provider by this mode's TIER WORD
            self.fold_key_mask = self.fused.apb_word is not None      # (fast | big): its rows read the key mask from the bias, so the hoisted logits carry it (swap_in_dtk ->
            self.calls = 0                                            # DiffusionHead.fold_key_mask_into_pair_logits) and logits computed here per call are folded here
            self.stock = None                                             # xfold's own DiffusionTransformer.forward (set by swap_in_dtk): the path a dead swap serves
            self.dead = None                                              # "<ExceptionType>: <message>" once the fused transformer raised here: stock serves the rest of the process, by name

        def forward(self, act, mask, single_cond, pair_cond, pair_logits=None):
            if self.dead is not None:
                return self.stock(self.tr, act, mask, single_cond, pair_cond, pair_logits=pair_logits)
            self.calls += 1
            try:
                if pair_logits is None:
                    pair_logits = self.tr.precompute_pair_logits(pair_cond)
                    if self.fold_key_mask:                            # not hoisted (no prime_static this call): the key mask folded into this call's logits, as the hoist does
                        pair_logits = pair_logits.masked_fill((mask == 0).reshape((1,) * (pair_logits.dim() - 1) + (-1,)), -1e9)
                bias = pair_logits.to(torch.bfloat16).contiguous()
                km = mask.to(torch.float32).contiguous()
                with torch.autocast("cuda", enabled=False):
                    out = self.fused.forward(act.float().contiguous(), single_cond.float().contiguous(), bias, key_mask=km)
                return out
            except Exception as e:                                            # the fused kernels cannot compile or launch on this stack: the swap steps aside BY NAME
                if _is_oom(e, torch) or self.stock is None or torch.cuda.is_current_stream_capturing():   # (dead; the census 'dead' record, a FALLBACK line, PARTIAL) and xfold's
                    raise                                                     # transformer serves; an out-of-memory is capacity and propagates as it did; an error under the
                                                                              # whole-step capture is the capture's (DiffusionHead.forward_graphed steps the graph aside), not the swap's
                self.dead = f"{type(e).__name__}: {(str(e).splitlines() or [''])[0][:160]}"
                print(f"[forward] lever_dead name=dtk reason={self.dead!r} -> xfold DiffusionTransformer.forward for the rest of the process", flush=True)
                return self.stock(self.tr, act, mask, single_cond, pair_cond, pair_logits=pair_logits)
    return XfoldDTK


def _provider_tier_word() -> str:
    """The tier word this process's mode asks the shared core's kernel providers for (af3_kernels.provider_tier: fast | big, named by
    set_provider_tier before build_model; fast when the kernels module is not importable here)."""
    try:
        import af3_kernels as K                                               # noqa: N811
        return str(getattr(K, "provider_tier", lambda: "fast")())
    except Exception:                                                         # noqa: BLE001
        return "fast"


def _is_oom(e, torch) -> bool:
    """A CUDA out-of-memory error — the shared core's one recogniser (opt_core.oom.is_oom), torch's own types when the core is absent."""
    try:
        from opt_core.oom import is_oom
        return bool(is_oom(e))
    except Exception:                                                         # noqa: BLE001
        types_ = tuple(t for t in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)) if isinstance(t, type))
        return (bool(types_) and isinstance(e, types_)) or "CUDA out of memory" in str(e)


def swap_in_dtk(model, torch):
    """The DTK swap: the kernel levers stay as built; DiffusionTransformer.forward -> the fused adapter; the whole-step
    graph (when the model carries one) is dropped so it re-captures with the fused transformer."""
    import dtk_modules as M
    from xfold.nn import diffusion_transformer as DT
    XfoldDTK = _xfold_dtk_class(torch, M)
    dtk = XfoldDTK(model.diffusion_head.transformer)
    dtk.fused.apb_graph = bool(getattr(model.diffusion_head, "use_step_graph", True))   # the batched step is replayed from a whole-step graph (stepgraph): the provider's graph-replay
    dtk.stock = DT.DiffusionTransformer.forward                                          # cells serve the warm-up steps too; hoist only (big): its eager cells
    model.diffusion_head.fold_key_mask_into_pair_logits = bool(dtk.fold_key_mask)   # the hoisted DiT pair logits carry the key mask (DiffusionHead.prime_static) when the fused
    DT.DiffusionTransformer.forward = lambda self, act, mask, single_cond, pair_cond, pair_logits=None: dtk.forward(act, mask, single_cond, pair_cond, pair_logits)
    model._af3t_dtk = dtk
    st = getattr(model.diffusion_head, "_static", None)
    if st is not None:
        st.pop("graph", None)
    torch.cuda.synchronize()
    return dtk


def _np_out(result, np, torch):
    """The result dict of _forward_samples -> flat {name: ndarray} (embeddings left out: the trunk state is not an output)."""
    def arr(v):
        return v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    out = {"atom_positions": arr(result["atom_positions"])}
    conf = result.get("confidence") or {}
    for k, v in conf.items():
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            out[f"conf_{k}"] = arr(v)
    dg = result.get("distogram")
    if isinstance(dg, dict):
        for k, v in dg.items():
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                out[k] = arr(v)
    elif dg is not None:
        out["distogram"] = arr(dg)
    return out


FASTNN_SWITCHES = ("layer_norm_implementation", "dot_product_attention_implementation", "gated_linear_unit_implementation")


def apply_fastnn(on) -> dict:
    """`--fastnn 1`: xfold's shipped Triton fastnn kernels — the stock CLI's own statements (run_alphafold.py:224-226, verbatim below; the
    switches are read at call time, so setting them before the model is built serves every layer). `--fastnn 0` leaves the module's defaults
    (`torch`: the eager path). Returns the three switches as this process runs them."""
    from xfold.fastnn import config as fastnn_config
    if on:
        fastnn_config.layer_norm_implementation = 'triton'
        fastnn_config.dot_product_attention_implementation = 'triton'
        fastnn_config.gated_linear_unit_implementation = 'triton'
    return {k: getattr(fastnn_config, k) for k in FASTNN_SWITCHES}


def main(argv=None):
    a = parse_args(argv)
    if int(a.n_gpu) > 1:
        return main_sharded(a, argv)
    return run(a)


# The n_gpu > 1 GATE TABLE (R-TP-3): every lever's state under `--mode big --n_gpu P`, P > 1 — each either COMPOSES with the row shard or
# is REPLACED / turned OFF / SKIPPED by name; no size threshold anywhere switches a statement back to a replicated pair (there is none).
# rowpair_xfold.py is the implementation; tests/test_n_gpu.py holds this table to it and to the README.
PACKAGE_LEVERS_SKIPPED_SHARDED = {"template_dedupe": "template_rows_per_slot", "dev_scalars": "rowpair_trunk", "tri_layout": "rowpair_rows", "ln_rows": "rowpair_ranks", "attn_layout": "rowpair_flash_rows", "gate_fuse": "rowpair_rows", "canonical_noise": "rowpair_sampler", "prefetch": "rowpair_ranks", "write_behind": "rowpair_ranks", "feat_par": "rowpair_ranks"}   # package levers the P > 1 path does not apply (ROWPAIR_GATES words), recorded per run as package_levers_skipped
ROWPAIR_GATES = (
    ("trimul", "replaced:rowpair_rows"),              # the FPF fused kernel assumes a whole square pair -> the core's row schedule: fpf_v4 row-block kernels (K1/K3 per tile) at N >= the core's min_tokens, the eager statements below it (F2.trimul_rows line)
    ("triattn", "composes:flash_rows"),                # flash triangle attention on the row batch with the whole gathered [N, N, 4] bias
    ("transition", "composes:row_local"),
    ("apb", "composes:sdpa_local_query_rows"),
    ("resid_fold", "skipped:rowpair_rows"),            # the row schedule restates the block's pair statements over row shards (pairstack.pair_block_); the kernel-epilogue folds of the single-GPU block forward are not in it
    ("attn_epi", "skipped:rowpair_rows"),              # the row schedule's triangle attention does its own gate / projection / add per row batch (triatt_fns)
    ("trimul_exact", "skipped:rowpair_rows"),          # as resid_fold: the row schedule restates the block's pair statements over row shards; under big lever trimul owns the class anyway (superseded)
    ("tmpl_trimul", "skipped:template_rows"),          # the row-sharded template embedder restates the template pair stack on rows (rowpair.template)
    ("pwa_lnl", "skipped:msa_rows"),                   # the row-sharded MSA module restates the pair-weighted averaging on rows (rowpair.msa)
    ("pwa_msa", "skipped:msa_rows"),          # as pwa_lnl: the MSA module is outside the row schedule
    ("opm", "composes:opm_row_blocks"),                # the row-sharded MSA module runs the lever's kernels per OUTPUT ROW BLOCK of this rank (rowpair_xfold.opm_rows_kernel_fn on msa.opm_rows_budgeted; the module's statements per block where the kernels' cell does not serve, counted by name)
    ("glu_proj", "composes:row_local"),                # as lever transition: the module-level patch serves the row shards' transition statements (with lever transition on, that kernel serves and glu_proj is superseded, as on one GPU)
    ("bf16w", "composes"),
    ("compile", "composes:row_local_modules"),
    ("hoist", "replaced:zcond_rows+bias_cache+band_statics"),
    ("stepgraph", "off:collectives"),
    ("dtk", "skipped:dit_local_query_rows"),
    ("template_dedupe", "skipped:template_rows_per_slot"),   # the row-sharded template embedding evaluates every slot on local rows (rowpair_xfold's template loop)
    ("canonical_noise", "skipped:rowpair_sampler"),       # the row-sharded roll-out restates the sampler; its draws are the model length's
    ("prefetch", "skipped:rowpair_ranks"),            # the streamed chain is the single-process item loop's (cli.cmd_pred strips it from --package-levers under n_gpu > 1, named on its LEVER line)
    ("write_behind", "skipped:rowpair_ranks"),
    ("autotune_cache", "composes:rank_env"),
    ("feat_par", "skipped:rowpair_ranks"),          # TRITON_CACHE_AUTOTUNING rides os.environ into every rank (spawn); each rank's autotuners read / fill the same disk cache
    ("sbatch", "skipped:rowpair_sampler"),                # the row-sharded roll-out restates the sampler sample by sample (rowpair_xfold run_diffusion_sharded)
    ("dev_scalars", "skipped:rowpair_trunk"),             # the row schedule restates the trunk's bond scatter (_bond_pairs) and template loop; the single-GPU rebinding is not applied
    ("tri_layout", "skipped:rowpair_rows"),               # the row schedule replaces the triangle multiplication with the core's row-block kernels; the stock module does not run
    ("atom_window", "skipped:rowpair_hoist"),             # its operands ride the single-GPU hoist, which the row schedule replaces (hoist: replaced) — the stock atom blocks run
    ("ln_rows", "skipped:rowpair_ranks"),
    ("attn_layout", "skipped:rowpair_flash_rows"),          # the row schedule's triangle attention binds flash_triattn on the row batch; the stock module does not run
    ("gate_fuse", "skipped:rowpair_rows"),                   # rides tri_layout / attn_layout, both replaced by the row schedule                 # the rank processes build their model without the package levers (PACKAGE_LEVERS_SKIPPED_SHARDED); the class binding is single-GPU
    ("castcache", "n/a:exact_only"),                      # --n_gpu P is big's, whose bf16w leaves autocast no weight cast to memoise
    ("token_agg", "composes:row_local"),                  # a row-local restatement of the encoder's own aggregation (its operands come with the encoder statics, hoisted or not)
    ("atom_rows", "skipped:rowpair_hoist"),               # rides atom_window, which the row schedule sets aside (hoist: replaced) — the stock atom blocks run
    ("prologue", "skipped:rowpair_sampler"),              # the row-sharded roll-out restates the sampler sample by sample; the batched prologue is the batched sampler's
    ("positions_sync", "bcast:rank0_state_in+out_per_denoiser_call+exit_spread_gate"),   # replicated statements run per rank on per-process-autotuned kernels (compile, fused tiles): equal to rounding, not bitwise
    ("prev_free", "subsumed:rowpair_carry"),
    ("paircond_chunk", "skipped:rowpair_sampler"),        # the row-sharded roll-out computes the pair conditioning on each rank's rows itself (rowpair_xfold pair_cond_rows)
    ("diff_free", "composes"),
    ("expandable_segments", "composes"),
)


def main_sharded(a, argv=None):
    """``--n-gpu P`` (P > 1): this process exports the allocator policy (inherited by every rank), then spawns P ranks through the shared
    core's launcher (opt_core.mem.rowpair.launch.run_sharded: rank r <-> visible device r, one NCCL group, any rank's failure tears all
    down = RankFailed) — each rank runs ``run()`` with the rowpair install; rank 0 writes ``--report`` and the result files, ranks > 0 write
    ``<report>.rank<r>.json``; per-rank transcripts: ``<report dir>/ranks/rank<r>.log``. Returns rank 0's exit code."""
    if a.opt_core and a.opt_core not in sys.path:
        sys.path.insert(0, a.opt_core)
    from opt_core.mem.rowpair import launch, RowpairRefused
    if a.alloc:                                                          # before any rank exists: PYTORCH_CUDA_ALLOC_CONF rides os.environ into every rank (spawn)
        from opt_core.mem import torch_alloc as TA
        try:
            TA.export(a.alloc, lever="expandable_segments", cuda_initialized=False)
        except Exception as e:                                           # noqa: BLE001 — refused by name: rc 2
            _write(a.report, {"error": f"--alloc {a.alloc} refused: {type(e).__name__}: {e}", "items": [], "ok": False, "dtk": bool(a.dtk), "n_gpu": int(a.n_gpu)}); return 2
    log_dir = os.path.join(os.path.dirname(os.path.abspath(a.report)), "ranks")
    os.makedirs(log_dir, exist_ok=True)
    t0 = time.time()
    try:
        rc, records = launch.run_sharded(int(a.n_gpu), _rank_entry, list(sys.argv[1:] if argv is None else argv), mode="big", nccl_timeout_s=RANK_TIMEOUT_S,
                                         log_dir=log_dir, return_records=True)
        rep = _read(a.report)                                            # rank 0's report + the launcher's census: how many ranks RAN (the wrapper's n_gpu assertion reads `world`)
        if rep is not None:
            rep["world"] = len(records); rep["ranks"] = [{k: r.get(k) for k in ("rank", "ok", "rc", "exitcode", "wall_s", "log", "reason") if k in r} for r in records]
            _write(a.report, rep)
    except (launch.RankFailed, RowpairRefused) as e:                     # a rank died / timed out / the launch was refused: the run's named failure (rank logs beside the report)
        rep = _read(a.report) or {"items": [], "dtk": bool(a.dtk)}
        rep.update(ok=False, error=f"rowpair launch: {type(e).__name__}: {str(e)[:2000]}", n_gpu=int(a.n_gpu), rank_logs=log_dir, launch_s=round(time.time() - t0, 3))
        for it in rep.get("items") or []:
            if it.get("ok") is not True:
                it.setdefault("error", f"rank_failed: {type(e).__name__}")
        _write(a.report, rep); print(f"[forward] n_gpu={a.n_gpu} FAILED {rep['error'][:400]}", flush=True); return 1
    return int(rc if rc is not None else 1)


def _rank_entry(argv):
    """Rank r's body under run_sharded (a fresh interpreter; the group is initialised, cuda:r is current). Hang diagnosis: ``kill -USR1``
    dumps every thread's Python stack into this rank's transcript."""
    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    a = parse_args(argv)
    from opt_core.mem.rowpair import launch
    return run(a, tp={"P": int(a.n_gpu), "rank": int(launch.rank()), "B": ROWPAIR_BLOCK})


def _read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def warm_core_imports(opt_core_dir=None):
    """The shared core's start-up warm-up (``opt_core.warm_imports``, opt_core >= 0.5.66.0): the stack's import-heavy model libraries are
    imported once here — torch first, then the others (``WARM_LIBRARIES``) — before the model is built, instead of inside the first item's forward where a kernel row first reaches for them
    (idempotent; free when they are imported already or absent; the core's own variables switch it off). Returns the core's report for
    forward.json (``warm_imports``: per library the import seconds or a word), or the named reason the core is not importable in this
    process — nothing a forward computes changes either way."""
    if opt_core_dir and opt_core_dir not in sys.path:
        sys.path.insert(0, opt_core_dir)
    try:
        import opt_core
    except ImportError as e:
        return {"skipped": f"core_not_importable:{getattr(e, 'name', None) or 'opt_core'}"}
    import torch  # noqa: F401 — torch before the libraries that link against its CUDA runtime (a cuequivariance import ahead of torch cannot load nvrtc on cu13 wheels)
    return {"core": getattr(opt_core, "__version__", "?"), "report": dict(opt_core.warm_imports(libraries=WARM_LIBRARIES, origin="af3_torch_opt.forward"))}


WARM_LIBRARIES = ("torch", "cuequivariance_ops_torch", "cuequivariance_torch")   # torch first, then the core's default import-heavy libraries (opt_core.warm.DEFAULT_LIBRARIES) in its order


def run(a, tp=None):
    """The model process body. ``tp`` None: one process, one device (n_gpu = 1). ``tp`` = {"P", "rank", "B"}: this is rank ``rank`` of P
    (main_sharded): the rowpair install rebinds the xfold pair stack, the trunk runs on this rank's rows, the heads run dense on rank 0."""
    rank = int(tp["rank"]) if tp else 0
    for p in reversed(kit_sys_path(a.kit, a.dtk_home)):
        sys.path.insert(0, p)
    routes = [t.strip() for t in a.route.split(",") if t.strip()]
    if a.stream_root:                                                   # the streamed chain: this process runs beside the wrapper, and ends with it (pipeline_stream)
        PS.exit_with_parent()
    autotune_cache = _autotune_cache_on() if "autotune_cache" in [t.strip() for t in a.package_levers.split(",") if t.strip()] else None   # before any kernel module is imported: Triton's autotuners read the knob when they are declared
    kernel_routes, route_error = route_kernels(routes, a.opt_core)   # before anything of the kit is imported: the routed names resolve to the core's copies, gated
    if route_error:
        _write(a.report, {"kernel_routes": kernel_routes, "error": route_error, "items": [], "ok": False, "dtk": bool(a.dtk)}); return 2
    big_levers = [t.strip() for t in a.big.split(",") if t.strip()]
    big = ({"levers": big_levers, "alloc": None, "diff_free_calls": 0, "prev_free_calls": 0, "items_ok": 0, "items": 0, "graph_drop_items": 0,
              "paircond_rows": (int(a.paircond_rows) if "paircond_chunk" in big_levers else 0), "paircond_items": 0, "paircond_blocks": 0,
              "graph_drop_min_tokens": (a.graph_drop_tokens or None)} if (big_levers or a.alloc) else None)
    if a.alloc:                                                          # the allocator policy: exported BEFORE torch initialises CUDA, under the core's named refusals
        if a.opt_core and a.opt_core not in sys.path:
            sys.path.insert(0, a.opt_core)
        try:
            from opt_core.mem import torch_alloc as TA
            if tp is None:                                               # graphs_on: the step graph stays built beside the policy (big's size-gated graph_drop) — its private pool is
                g_on = "stepgraph" in [t.strip() for t in a.levers.split(",")]   # memory the policy cannot return, composed on purpose below graph_drop's size gate (big.py)
                TA.export(a.alloc, lever="expandable_segments", graphs_on=g_on, allow_with_graphs=g_on)
                big["alloc"] = {"policy": a.alloc, "conf": os.environ.get(TA.ENV), "effective": None, "source": "exported"}
            else:                                                        # a rank: main_sharded exported the policy before spawning; this interpreter inherited it (CUDA is bound already)
                if os.environ.get(TA.ENV) != TA.conf_for(a.alloc):
                    raise RuntimeError(f"{TA.ENV}={os.environ.get(TA.ENV)!r} in rank {rank}: the launcher's export {TA.conf_for(a.alloc)!r} was not inherited")
                big["alloc"] = {"policy": a.alloc, "conf": os.environ.get(TA.ENV), "effective": None, "source": "inherited"}
        except Exception as e:                                           # noqa: BLE001 — MemLeverRefused (CUDA initialised / another conf present) or the core's mem absent: rc 2, named
            _write(_rank_report(a.report, rank), {"kernel_routes": kernel_routes, "big": big, "error": f"--alloc {a.alloc} refused: {type(e).__name__}: {e}", "items": [], "ok": False, "dtk": bool(a.dtk)}); return 2
        print(f"[forward] alloc policy={a.alloc} conf={big['alloc']['conf']}", flush=True)
    import numpy as np
    import torch
    import af3_torch_api as A
    levers = tuple(t.strip() for t in a.levers.split(",") if t.strip())
    warm = warm_core_imports(a.opt_core)                                # the core's heavy model libraries imported here, once, at start-up (after the allocator policy is exported, torch first) — not inside the first item's forward
    rep = {"graph_resets": 0, "graph_captures": 0, "levers_requested": list(levers), "template_dedupe": {"calls": 0, "slots": 0, "evaluated": 0, "scans": 0, "tables": 0}, "dev_scalars": {"norm_calls": 0, "eps_tensors": 0, "bond_calls": 0}, "tri_layout": {"calls": 0, "tiled": 0, "generic": 0, "scope": None}, "ln_rows": {"calls": 0, "served": 0, "wide": 0, "generic": 0, "modules": 0}, "attn_layout": {"calls": 0, "regrouped": 0, "small": 0, "generic": 0, "modules": 0}, "gate_fuse": {"calls": 0, "fused": 0, "generic": 0, "form": None, "hosts": None}, "castcache": {"linears": 0, "mib": 0.0, "served": 0, "stock": 0}, "canonical_noise": {"trajectories": 0, "model_len": None, "canonical_len": None}, "prefetch": {"streamed": int(bool(a.stream_root)), "items": 0, "waited": 0, "wait_s": 0.0, "first_wait_s": None, "withdrawn": 0}, "write_behind": {"streamed": int(bool(a.stream_root)), "published": 0}, "autotune_cache": {"knob": 0, "hits": 0, "benched": 0}, "padding": a.padding, "dtk": bool(a.dtk), "params": a.params, "weights": {"file": a.params, "sha256": a.weights_sha256, "pinned": (None if a.weights_pinned is None else bool(a.weights_pinned))}, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "torch": torch.__version__, "env_prefixes_present": sorted(k for k in os.environ if k.startswith("AF3_TORCH_OPT")), "items": [], "ok": False,
           "kernel_routes": kernel_routes, "warm_imports": warm, "big": big, "n_gpu": int(tp["P"]) if tp else 1, "rank": rank, "world": int(tp["P"]) if tp else 1}   # world: overwritten by main_sharded with the launcher's rank census under n_gpu > 1
    report_path = _rank_report(a.report, rank)                        # rank 0 (and n_gpu = 1): --report itself; rank r > 0: <report>.rank<r>.json
    if not torch.cuda.is_available():
        rep["error"] = "no CUDA device"
        _write(report_path, rep); return 1
    if not os.path.isfile(a.params):                                # the exact pinned checkpoint file, never a directory: the kit's loader
        rep["error"] = f"--params {a.params} is not a file (the pinned checkpoint file is required, never a directory)"   # would take the
        _write(report_path, rep); return 2                                # first *.bin.zst of a directory (xfold/params.py:739-743)
    rep["fastnn"] = apply_fastnn(a.fastnn)                            # xfold's fastnn implementation switches as this process runs them ({..._implementation: torch|triton}; triton = --fastnn 1)
    kw = {k: v for k, v in (("num_recycles", a.num_recycles), ("num_samples", a.num_samples), ("diffusion_steps", a.diffusion_steps)) if v is not None}
    if hasattr(A, "set_provider_tier"):                                   # the shared-core providers' TIER WORD for this pred's mode, where a cells row asks for it (lever trimul at
        A.set_provider_tier("big" if big_levers else "fast")             #   c=128, trimul_row=tier): big = the pred composes big's memory levers (--big), fast otherwise
    t0 = time.time()
    try:
        model = A.build_model(a.params, levers=levers, **kw)
    except Exception as e:                                         # a failed build is the run's named failure (a missing key, a bad params dir)
        rep["error"] = f"build_model failed: {e!r}"; rep["traceback"] = traceback.format_exc()[-4000:]
        _write(report_path, rep); return 1
    rep["build_s"] = round(time.time() - t0, 3)
    rep["levers_applied"] = list(getattr(model, "_af3t_levers", ()))
    rep["package_levers"] = [t.strip() for t in a.package_levers.split(",") if t.strip()]
    rep["package_levers_applied"] = []
    for pl in rep["package_levers"]:                                      # the package levers, in the requested order (registry.PACKAGE_LEVERS)
        try:
            if int(a.n_gpu) > 1 and pl in PACKAGE_LEVERS_SKIPPED_SHARDED:   # ROWPAIR_GATES: the row-sharded path restates these statements — named, not applied
                rep.setdefault("package_levers_skipped", {})[pl] = "n_gpu>1:" + PACKAGE_LEVERS_SKIPPED_SHARDED[pl]
                continue
            if pl == "template_dedupe":                                   # distinct template slots evaluated once, found once per item; the embedder's constant index tables device-resident (template_dedupe.py; bitwise by construction)
                rep["template_dedupe"]["tables"] = len(_template_dedupe().install(model).get("tables") or ())
            elif pl == "dev_scalars":                                     # the trunk pass's constant scalars device-resident: the norm's epsilon clip, the bond contact matrix's ones / zero (dev_scalars.py; bitwise by construction)
                _dev_scalars().install(model)
            elif pl == "tri_layout":                                      # the stock triangle multiplication's layout copies on a tiled transpose kernel (tri_layout.py; bitwise by construction): the class where the stock forward serves, else the instances the trimul kernel lever leaves to stock
                tl_ = _tri_layout().install(model); rep["tri_layout"]["scope"] = tl_.get("scope")
                if tl_.get("reason"): rep["tri_layout"]["reason_scope"] = tl_["reason"]
            elif pl == "ln_rows":                                         # xfold's fastnn Triton LayerNorm over 8 rows per program (ln_rows.py; bitwise by construction)
                lr = _ln_rows().install(model); rep["ln_rows"]["modules"] = int(lr.get("modules") or 0)
                if lr.get("reason"): rep["ln_rows"]["reason_scope"] = lr["reason"]
            elif pl == "attn_layout":                                     # the triangle attention's operand / output regroupings on a row-regroup kernel (attn_layout.py; bitwise by construction)
                alr = _attn_layout().install(model); rep["attn_layout"]["modules"] = int(alr.get("modules") or 0)
                if alr.get("reason"): rep["attn_layout"]["reason_scope"] = alr["reason"]
            elif pl == "gate_fuse":                                       # the two restated forwards' mask / sigmoid-gate statements as one kernel each (gate_fuse.py; the stock rounding points)
                gfr = _gate_fuse().install(model); rep["gate_fuse"]["form"] = gfr.get("form"); rep["gate_fuse"]["hosts"] = "+".join(gfr.get("hosts") or ()) or "none"
                if gfr.get("reason"): rep["gate_fuse"]["reason_scope"] = gfr["reason"]
            elif pl == "castcache":                                       # each fp32 Linear's bf16 autocast weight cast memoised per process (castcache.py; bitwise by construction)
                cc = _castcache().install(model); rep["castcache"]["linears"] = int(cc.get("linears") or 0); rep["castcache"]["mib"] = float(cc.get("mib") or 0.0)
                if cc.get("reason"): rep["castcache"]["reason_scope"] = cc["reason"]
            elif pl == "canonical_noise":                                 # the sampler's shape-dependent draws at the input's own token count, laid into the padded length (canonical_noise.py)
                _canonical_noise().install(model)
            elif pl in ("prefetch", "write_behind", "feat_par"):                      # the streamed chain (pipeline_stream): this process waits for each batch's hand-off (prefetch) / hands each result over as it closes (write_behind) — scheduling only, under --stream-root
                pass
            elif pl == "autotune_cache":                                  # Triton's autotune results cached on disk and reused across processes (the knob was set before any kernel module was imported, above)
                if not autotune_cache or not autotune_cache.get("knob"):
                    rep.setdefault("package_levers_skipped", {})["autotune_cache"] = (autotune_cache or {}).get("reason") or "triton_knob_absent"; continue
                rep["autotune_cache"] = autotune_cache["census"]
            else:
                raise ValueError(f"unknown package lever {pl!r}")
            rep["package_levers_applied"].append(pl)
        except Exception as e:                                            # a lever that cannot attach is the run's named failure, never a silent stock path under the lever's name
            rep["error"] = f"package lever {pl} failed to attach: {e!r}"; rep["traceback"] = traceback.format_exc()[-4000:]
            _write(_rank_report(a.report, rank), rep); return 1
    rep["kernels_enabled"] = bool(getattr(model, "_af3t_kernels", None))
    census0 = _census(model)                                              # the kit's kernel census (None when no kernel lever is enabled: off)
    rep["compiled"] = getattr(model, "_af3t_compiled", None)
    if a.dtk and tp is not None:                                          # GATE (n_gpu > 1): the DTK whole-transformer fusion cannot serve local query rows — skipped BY NAME, never silently
        rep["dtk_state"] = "skipped:n_gpu>1:dit_local_query_rows"
        print(f"[forward] LEVER dtk state=skipped reason=n_gpu>1:dit_local_query_rows rank={rank}", flush=True)
    elif a.dtk:
        t0 = time.time()
        try:
            swap_in_dtk(model, torch)
            rep["dtk_swap_s"] = round(time.time() - t0, 3)
        except Exception as e:                                            # the DTK modules cannot be built or imported on this stack: named (dtk_state, the census
            if _is_oom(e, torch):                                         # 'dead' record -> a FALLBACK line and PARTIAL), and xfold's transformer serves the run;
                raise                                                     # an out-of-memory propagates as it did
            model._af3t_dtk_dead = f"swap_failed: {type(e).__name__}: {(str(e).splitlines() or [''])[0][:160]}"
            rep["dtk_state"] = "dead:swap_failed"; rep["dtk_error"] = traceback.format_exc()[-2000:]
            print(f"[forward] lever_dead name=dtk reason={model._af3t_dtk_dead!r} -> xfold DiffusionTransformer.forward", flush=True)
    RPX = None
    if tp is not None:                                                    # n_gpu > 1: the rowpair adapter (the shared core's row-sharded pair representation; no class is rebound), once per rank
        try:
            RPX = _load_rowpair_xfold()
            st = RPX.install(model, P=tp["P"], rank=rank, align=tp.get("B"))
            rep["rowpair"] = {"P": tp["P"], "rank": rank, "align": st["align"], "impl": "opt_core.mem.rowpair", "core_version": _core_version(),
                              "trimul": "rowpair_rows", "triattn": "flash_rows", "heads": "sharded_rows", "diffusion": "zcond_rows+dit_local_queries",
                              "prev_free": "rowpair_carry" if (big and "prev_free" in big_levers) else False, "dtk": "skipped:n_gpu>1" if a.dtk else "off",
                              "gates": dict(ROWPAIR_GATES)}
        except Exception as e:                                            # noqa: BLE001 — refused / failed by name: rc 2 on this rank (the launcher tears the others down)
            rep["error"] = f"rowpair install failed: {e!r}"; rep["traceback"] = traceback.format_exc()[-4000:]
            _write(report_path, rep); return 2
        print(f"[forward] rowpair P={tp['P']} rank={rank} align={st['align']} " + " ".join(f"{k}={v}" for k, v in ROWPAIR_GATES), flush=True)
    if big is not None:                                                 # the memory line's process-level facts, read back — never the request
        if big["alloc"] is not None:
            eff = TA.effective()
            big["alloc"].update(effective=eff.get("expandable"), source=eff.get("source"))
        print(f"[forward] big levers={','.join(big_levers) or 'none'} alloc={big['alloc']}", flush=True)
    print(f"[forward] levers={'+'.join(rep['levers_applied']) or 'none'} package_levers={','.join(rep['package_levers_applied']) or 'none'} dtk={int(a.dtk)} build_s={rep['build_s']}", flush=True)
    failed = 0
    declared = templates_declared_map(a.templates_declared)
    for spec in a.item:
        try:
            name, seed_s, batch_path, out_path = spec.split("=", 3)
            seed = int(seed_s)
        except ValueError:
            failed += 1
            rep["items"].append({"name": spec, "ok": False, "error": f"bad --item spec {spec!r}: NAME=SEED=<batch.npz>=<result.npz>"})
            continue
        it = {"name": name, "seed": seed, "batch": batch_path, "result": out_path, "ok": False}
        if a.stream_root and "prefetch" in rep["package_levers_applied"]:    # the streamed chain: the featuriser runs beside this process — wait for this seed's hand-off (or learn it will never come: withdrawn, no record, as the sequential chain never lists it)
            w = PS.await_batch(batch_path, a.stream_root)
            pf = rep["prefetch"]
            if not w["ready"]:
                pf["withdrawn"] += 1
                print(f"[forward] {name} seed={seed} withdrawn reason={w.get('reason')} wait_s={w['wait_s']}", flush=True)
                continue
            it["batch_wait_s"] = w["wait_s"]; pf["items"] += 1; pf["waited"] += int(w["wait_s"] >= 0.05); pf["wait_s"] = round(pf["wait_s"] + w["wait_s"], 3)
            if pf["first_wait_s"] is None:
                pf["first_wait_s"] = w["wait_s"]
        rep["items"].append(it)
        met = True                                                            # n_gpu > 1: whether this rank has met the others at the end of this item
        try:
            torch.cuda.reset_peak_memory_stats()
            t_load = time.time(); batch = A.batch_from_npz(batch_path); it["load_s"] = round(time.time() - t_load, 3)   # the host gap before the item: batch.npz read + H2D
            n_tok = int(batch["seq_length"].reshape(-1)[0].item()) if "seq_length" in batch else None
            bucket = int(batch["token_index"].shape[-1]) if "token_index" in batch else None
            it["templates_live"], it["templates_declared"], it["templates_slots"], it["templates_slots_live"] = template_entries_live(batch), declared.get(name, 0), template_slots_total(batch), template_slots_live(batch)   # the template census per item: slots featurised live / templates the fold input declares (the ITEM line's templates=l/d; every item runs, as stock does)
            # The item's randomness is a function of its seed alone: the trunk's MSA row shuffle draws from the global torch RNG
            # (xfold/alphafold3.py:184 shuffle_msa -> nn/featurization.py:227 gumbel_argsort_sample_idx) and the kit seeds only inside its
            # sampler (af3_torch_api.py:183-184) — seeding here makes the row order position-independent; the sampler re-seeds itself.
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); it["rng"] = "seeded_per_item"
            # Every item starts from a model with NO captured CUDA graph and no hoisted state: the diffusion head's whole-step graph
            # (keyed on id(batch), xfold/nn/diffusion_head.py:241) and its per-trajectory conditioning buffers — clear_static (:232) — and
            # the kernel kit's mask-term cache (af3_kernels.clear_caches). The next item re-captures its own graph. Named events: graph_reset
            # (a whole-step graph was dropped), graph_capture (a whole-step graph captured by this item), per item and summed on the report.
            it["graph_reset"] = _reset_item_state(model, torch); rep["graph_resets"] += it["graph_reset"]
            if big and "graph_drop" in big_levers and a.graph_drop_tokens and "stepgraph" in levers:   # big's size-gated graph_drop: the step graph off for THIS item at or
                drop = bucket is not None and int(bucket) >= int(a.graph_drop_tokens)                         # above the gate (the hoist stays: use_hoist rides stepgraph in build_model),
                model.diffusion_head.use_step_graph = not drop                                                 # replayed below it; recorded per item and summed on the big record
                it["graph_drop"] = int(drop); big["graph_drop_items"] += int(drop)
            if big and big.get("paircond_rows") and RPX is None:          # big's paircond_chunk: the pair conditioning per row block (diffusion_head._pair_conditioning -> xfold/nn/paircond_rows.py)
                model.diffusion_head.pair_cond_rows = int(big["paircond_rows"]); model.diffusion_head._paircond_blocks = 0
            if "template_dedupe" in rep["package_levers_applied"]:       # the item's distinct-slot map is found on ITS template tensors: forget the last item's before anything of this one runs
                _template_dedupe().begin_item()
            if RPX is not None:
                lay = RPX.begin_item(bucket); it["layout"] = {"N": lay.N, "P": lay.P, "B": lay.B, "rank": lay.rank, "r0": lay.r0, "r1": lay.r1, "R": lay.R, "Rmax": lay.Rmax}; met = False   # this rank's rows of the item's pair (refused by name when the grid cannot shard this N over P)
                it["guards"] = {"features": RPX.assert_features_replicated(batch)}   # R1: the featurised batch every rank loaded is ONE draw (featurise.py ran once, before any rank): proven replicated by checksum, refused by name otherwise
                if not RPX.STATE.get("params_guard"):                     # once per model process: the weights every rank loaded are bitwise one set (a per-rank desync refuses by name)
                    RPX.STATE["params_guard"] = it["guards"]["params"] = RPX.assert_params_replicated(model)
            torch.cuda.synchronize(); t0 = time.time()
            with A.inference():
                result = _forward_samples(A, model, batch, seed, torch, diff_free=bool(big and "diff_free" in big_levers), item=it,
                                          prev_free=bool(big and "prev_free" in big_levers), rpx=RPX)   # AF3 Model.__call__ over S = model.num_samples (not A.forward: sample 0 only)
            torch.cuda.synchronize(); it["forward_s"] = round(time.time() - t0, 3)
            if RPX is not None:                                           # every rank's peak (collective), then the ranks meet before the next item
                pk = max([torch.cuda.max_memory_allocated() / 2**30] + [v["peak_gb"] for v in (it.get("phases") or {}).values()])
                it["peak_mem_gb_ranks"] = {str(k): round(float(v), 3) for k, v in sorted(RPX.gather_peaks(pk).items())}
                RPX.barrier(); RPX.end_item(); met = True
            it["graph_capture"] = _has_step_graph(model) or int((it.get("diff_freed") or {}).get("step_graph") or 0)   # a whole-step graph captured by this item: still held,
            rep["graph_captures"] += it["graph_capture"]                                                                 # or released by big's diff_free when the sampler returned
            if "template_dedupe" in rep["package_levers_applied"]:       # the lever's census for this item: template embedder calls, slots seen, distinct slots evaluated
                td = _template_dedupe().take(); it["template_calls"], it["template_slots"], it["template_slots_evaluated"], it["template_scans"] = td["calls"], td["slots"], td["evaluated"], td["scans"]
                for k in ("calls", "slots", "evaluated", "scans"):
                    rep["template_dedupe"][k] += td[k]
            if "dev_scalars" in rep["package_levers_applied"]:           # the lever's census for this item: norm calls served, epsilon tensors built (once per process), bond-matrix calls
                ds = _dev_scalars().take(); it["dev_scalars"] = ds
                for k in ("norm_calls", "eps_tensors", "bond_calls"):
                    rep["dev_scalars"][k] += int(ds.get(k) or 0)
            if "tri_layout" in rep["package_levers_applied"]:            # the lever's census for this item: triangle-multiplication calls, those served by the tiled copies, those on the stock strided path
                tly = _tri_layout().take(); it["tri_layout"] = tly
                for k in ("calls", "tiled", "generic"):
                    rep["tri_layout"][k] += int(tly.get(k) or 0)
            if "ln_rows" in rep["package_levers_applied"]:               # the lever's census for this item: LayerNorm calls served row-blocked / wide rows on the stock launch / the stock statement
                lrt = _ln_rows().take(); it["ln_rows"] = lrt
                for k in ("calls", "served", "wide", "generic"):
                    rep["ln_rows"][k] += int(lrt.get(k) or 0)
            if "attn_layout" in rep["package_levers_applied"]:           # the lever's census for this item: grid self-attention calls regrouped / on the stock statements
                alt = _attn_layout().take(); it["attn_layout"] = alt
                for k in ("calls", "regrouped", "small", "generic"):
                    rep["attn_layout"][k] += int(alt.get(k) or 0)
            if "gate_fuse" in rep["package_levers_applied"]:             # the lever's census for this item: gating statements fused / stock
                gft = _gate_fuse().take(); it["gate_fuse"] = gft
                for k in ("calls", "fused", "generic"):
                    rep["gate_fuse"][k] += int(gft.get(k) or 0)
            if "castcache" in rep["package_levers_applied"]:             # the lever's census for this item: Linear calls served from the memo / run on the stock statement
                cct = _castcache().take(); it["castcache"] = cct
                for k in ("served", "stock"):
                    rep["castcache"][k] += int(cct.get(k) or 0)
            if "canonical_noise" in rep["package_levers_applied"]:       # the lever's census for this item: trajectories drawn, the model's padded length, the canonical (bucket) length
                cn = _canonical_noise().take(); it["noise_model_len"], it["noise_canonical_len"] = cn["model_len"], cn["canonical_len"]
                rep["canonical_noise"]["trajectories"] += int(cn["trajectories"] or 0); rep["canonical_noise"]["model_len"] = cn["model_len"]; rep["canonical_noise"]["canonical_len"] = cn["canonical_len"]
            it["graph_retries"] = list(getattr(getattr(model, "diffusion_head", None), "graph_retries", ()) or ())   # whole-step captures that failed once and were retried on the next step (diffusion_head.GRAPH_CAPTURE_RETRIES)
            sb = getattr(model, "_sbatch_counts", None)                     # lever 'sbatch': the batched sampler's census of this item (calls, chunk, RNG plan, the DTK route's attention)
            if sb is not None:
                dtk_ = getattr(getattr(model, "_af3t_dtk", None), "fused", None)
                it["sbatch"] = dict(sb, dtk_route=getattr(dtk_, "batched_route", None), dtk_attn=getattr(dtk_, "batched_attn", None), dtk_attn_event=getattr(dtk_, "batched_attn_event", None),
                                    dtk_apb_word=getattr(dtk_, "apb_word", None), dtk_apb_rows=(dtk_.apb_rows() if hasattr(dtk_, "apb_rows") else None))   # the provider tier word the batched attention asked and the arm served per call class
                sb.update(batched_calls=0, single_calls=0, trajectories=0)
            census1 = _census(model)
            it["fallbacks"], it["dead"] = _fallback_delta(census0, census1); census0 = census1
            if rank == 0:                                                 # rank 0 (and n_gpu = 1) holds the outputs; ranks > 0 end the item at the trunk
                t_d2h = time.time(); arrays = _np_out(result, np, torch); it["d2h_s"] = round(time.time() - t_d2h, 3)   # the result arrays to the host
                arrays["seed"] = np.asarray(seed)
                arrays["model_id"] = _model_id(model, np, torch)        # the params file's __meta__/__identifier__ record the kit's loader keeps on the model
                os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                t_write = time.time(); np.savez(out_path, **arrays); it["write_s"] = round(time.time() - t_write, 3)   # the host gap after the item: result.npz written (the D2H is in _np_out above)
                result = None                                             # this item's device outputs (the trunk embeddings [N, N, 128] fp32, positions, per-sample confidences) are
                #                                                           released HERE, once on the host: held in this name they stayed allocated through the NEXT item's forward
                #                                                           (+[N, N, ~150] x 4 B on every later item's peak; ranks > 0 hold no outputs)
                if a.stream_root and "write_behind" in rep["package_levers_applied"]:   # the streamed chain: the writers run beside this process — the file is closed, hand it over
                    PS.touch(os.path.join(os.path.dirname(os.path.abspath(out_path)), PS.RESULT_READY)); rep["write_behind"]["published"] += 1
            else:
                arrays = {}
            it.update(ok=True, n_tokens=n_tok, bucket=bucket, peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3), peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 3),
                      arrays=sorted(arrays))
            if it.get("phases"):
                it["peak_mem_gb_phases"] = it.pop("phases")
                it["peak_mem_gb"] = max([it["peak_mem_gb"]] + [v["peak_gb"] for v in it["peak_mem_gb_phases"].values()])   # stage peaks are stage-local (reset between stages)
            if big is not None:                                         # the memory line's per-item facts
                big["items"] += 1
                big["diff_free_calls"] += int(it.get("diff_free") or 0)
                big["prev_free_calls"] += int(it.get("prev_free") or 0)
                if big.get("paircond_rows") and RPX is None:                # paircond_chunk's census: blocks evaluated this item (1 = the input ran whole)
                    blocks = int(getattr(model.diffusion_head, "_paircond_blocks", 0) or 0)
                    it["paircond_blocks"] = blocks; big["paircond_blocks"] += blocks; big["paircond_items"] += int(blocks > 1)
                big["items_ok"] = big.get("items_ok", 0) + 1
            print(f"[forward] {name} seed={seed} ok tokens={n_tok} bucket={bucket} forward_s={it['forward_s']} peak_gb={it['peak_mem_gb']} reserved_gb={it['peak_reserved_gb']}", flush=True)
        except Exception as e:
            failed += 1
            it["error"] = repr(e)[:400]; it["traceback"] = traceback.format_exc()[-4000:]
            print(f"[forward] {name} seed={seed} FAILED {it['error']}", flush=True)
            if RPX is not None and not met:                               # this rank left the item early: meet the others (a rank that died inside a collective times the group out by name instead)
                try:
                    it["peak_mem_gb_ranks"] = {str(k): round(float(v), 3) for k, v in sorted(RPX.gather_peaks().items())}
                    RPX.barrier()
                finally:
                    RPX.end_item(gate=False); met = True                    # the item already failed by name: no seam gate on the error path
    rep["ok"] = bool(a.item) and failed == 0
    rep["failed"] = failed
    rep["census"] = _census(model)                                        # the process's final census, verbatim (counts, dead, graphs, hoist, arch)
    rep["fallback_events"] = _events(rep["census"])                       # {lever: {"fallback:<reason>" | "kernel_error": n}} — empty when every call was served
    rep["dead"] = dict((rep["census"] or {}).get("dead") or {})
    for name, src in imported_from(routes).items():                       # what EXECUTED: the routed module's file in this process (None: no lever imported it)
        rep["kernel_routes"][name]["imported_from"] = src
        print(f"[forward] kernel_import name={name} from={src}", flush=True)
    if RPX is not None:
        rep["rowpair"]["stats"] = dict(RPX.STATE["stats"])
        from opt_core.mem.rowpair import launch as _launch                # the core's own log tag for its per-rank lines (SCHEDULE, and this one)
        rep["rowpair"]["trimul_rows"] = RPX.trimul_rows_record()          # the fused TriMul rows' ledger (None: the eager statements ran, as before the core carried them)
        rep["rowpair"]["trimul_rows_line"] = RPX.emit_trimul_rows_line(os.environ.get(_launch.ENV_TAG, "rowpair"))   # printed here once per rank; rank 0's is relayed by the wrapper
    _write(report_path, rep)
    return 0 if rep["ok"] else 1


def _rank_report(report, rank):
    return report if int(rank) == 0 else f"{report}.rank{int(rank)}.json"


def _core_version():
    try:
        from opt_core import __version__ as v
        return v
    except Exception:                                                     # noqa: BLE001
        return None




_EVENT_KEYS = ("fallback:", "kernel_error")


def _model_id(model, np, torch):
    """The loaded parameters' identifier record (uint8[64]; xfold/params.py sets model.__identifier__ from __meta__/__identifier__) as a numpy
    array — postprocess.py hands it to the fork's writers; a model without it is a named failure of the item."""
    ident = getattr(model, "__identifier__", None)
    if ident is None:
        raise RuntimeError("the loaded model carries no __identifier__ record (xfold/params.py import_params_dict_)")
    return np.asarray(ident.detach().cpu().numpy() if torch.is_tensor(ident) else ident)


def _has_step_graph(model) -> int:
    """1 if the diffusion head holds a captured whole-step graph now (captured by the item that just ran), else 0."""
    st = getattr(getattr(model, "diffusion_head", None), "_static", None)
    return int(isinstance(st, dict) and st.get("graph") is not None)


def _census(model):
    """The kit's kernel census (``af3_kernels.census()``: per-lever call counters — served:* / fallback:<reason> / kernel_error — the
    levers on, the dead ones, graph captures / replays, hoist hits / misses), its ``dead`` record extended with the runtime levers that
    stepped aside by name in this process (_runtime_dead); None when the model carries no kernel lever and nothing stepped aside."""
    K = getattr(model, "_af3t_kernels", None)
    census = K.census() if K is not None else None
    if getattr(model, "_atom_window_state", None) is not None:  # lever 'atom_window': served = denoiser calls on the window kernels, fallback:<word> = calls the
        from xfold.nn.diffusion_transformer import DiffusionCrossAttTransformer as _DCAT   # kernels refused by name (stock blocks ran), blocks_real / blocks_total = query
        c = _DCAT.WINDOW_COUNTS                                                            # blocks handed to the kernels / present (padding blocks are not computed)
        census = dict(census or {})
        census["atom_window"] = {"served:calls": int(c["calls"]), **{"fallback:%s" % k: int(v) for k, v in c["refused"].items()},
                                 "kernel_error": 0, "blocks_real": int(c["blocks_real"]), "blocks_total": int(c["blocks_total"]), "cell": c.get("cell")}
    if getattr(model, "_token_agg_state", None) is not None:    # lever 'token_agg': served = encoder calls on the aggregation kernel, fallback:<word> = calls it refused by name
        from xfold.nn.atom_cross_attention import AtomCrossAttEncoder as _ACE              # (the stock gather + masked mean ran)
        c = _ACE.TOKEN_AGG_COUNTS
        census = dict(census or {})
        census["token_agg"] = {"served:calls": int(c["calls"]), **{"fallback:%s" % k: int(v) for k, v in c["refused"].items()}, "kernel_error": 0}
    if getattr(model, "_atom_rows_state", None) is not None:    # lever 'atom_rows': served = atom-transformer calls on the rows-only path, fallback:<word> = calls its layout refused
        from xfold.nn.diffusion_transformer import DiffusionCrossAttTransformer as _DCT2
        c = _DCT2.ROWS_COUNTS
        census = dict(census or {})
        census["atom_rows"] = {"served:calls": int(c["calls"]), **{"fallback:%s" % k: int(v) for k, v in c["refused"].items()}, "kernel_error": 0}
    dead = _runtime_dead(model)
    if dead:
        census = dict(census or {}); census["dead"] = {**dict(census.get("dead") or {}), **dead}
    return census


def _runtime_dead(model) -> dict:
    """{lever: reason} of the runtime levers that could not run on this stack and stepped aside BY NAME (the same statements served without
    them): ``stepgraph`` — the whole-step CUDA graph failed to capture (diffusion_head.graph_failures; the hoisted step runs eagerly);
    ``compile`` — a compiled callable raised when called (af3_torch_api.COMPILE_FALLBACKS; it runs eagerly); ``dtk`` — the fused diffusion
    transformer could not be built or raised when called (xfold's transformer serves). An out-of-memory is never one of these."""
    out = {}
    dh = getattr(model, "diffusion_head", None)
    fails = getattr(dh, "graph_failures", None)
    if fails:
        out["stepgraph"] = f"capture_failed x{len(fails)}: {fails[-1]}"
    cf = getattr(model, "_af3t_compile_fallbacks", None)
    if cf:
        out["compile"] = f"eager x{len(cf)}: " + "; ".join(f"{k}={v}" for k, v in sorted(cf.items())[:3])
    dtk = getattr(model, "_af3t_dtk", None)
    if getattr(dtk, "dead", None):
        out["dtk"] = dtk.dead
    elif getattr(model, "_af3t_dtk_dead", None):
        out["dtk"] = model._af3t_dtk_dead
    aw = getattr(model, "_atom_window_state", None)              # atom_window could not engage at build (needs_hoist / kernel_import / warmup): the stock atom blocks serve
    if aw not in (None, "on"):
        out["atom_window"] = aw
    ta = getattr(model, "_token_agg_state", None)                # token_agg could not engage at build (kernel_import / warmup): the stock aggregation statements serve
    if ta not in (None, "on"):
        out["token_agg"] = ta
    ar = getattr(model, "_atom_rows_state", None)                # atom_rows needs atom_window on: else forward_windowed / the stock blocks serve, by name
    if ar not in (None, "on"):
        out["atom_rows"] = ar
    pl = getattr(model, "_prologue_state", None)                 # prologue needs the sample-batched sampler: else the per-sample prologue serves, by name
    if pl not in (None, "on"):
        out["prologue"] = pl
    return out


def _events(census):
    out = {}
    for lever, counts in (census or {}).items():
        if isinstance(counts, dict) and lever not in ("dead", "graphs", "hoist", "arch"):
            ev = {k: v for k, v in counts.items() if k.startswith(_EVENT_KEYS[0]) or k == _EVENT_KEYS[1]}
            if ev:
                out[lever] = ev
    return out


def _fallback_delta(before, after):
    """The fallback / kernel-error events of one item (the counters' increase over the item) and the levers dead after it."""
    b, a_ = _events(before), _events(after)
    delta = {}
    for lever, counts in a_.items():
        d = {k: v - b.get(lever, {}).get(k, 0) for k, v in counts.items() if v - b.get(lever, {}).get(k, 0) > 0}
        if d:
            delta[lever] = d
    return delta, sorted((after or {}).get("dead") or {})


def _write(path, rep):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, sort_keys=True, default=str)




if __name__ == "__main__":
    import faulthandler, signal                                           # `kill -USR1 <pid>` dumps every thread's Python stack to stderr (the rank transcript):
    faulthandler.register(signal.SIGUSR1, all_threads=True)              # a hung collective is located without a debugger
    sys.exit(main())
