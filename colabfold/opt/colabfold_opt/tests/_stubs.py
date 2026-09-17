"""Stubs for the CPU tests: a stand-in kit directory (the kit's integration module with the kit's own `_STATE` shape and `enable()`
contract, no jax), stub `colabfold.batch` / `alphafold.model.modules` / `alphafold.model.model` modules, a stand-in `jax` for boxes
without one, and the gate monkeypatches (pins, parameters, GPU, jax version) so the activation logic runs on a box without colabfold, jax
or a GPU. Nothing here is imported by the package."""
from __future__ import annotations

import importlib
import os
import sys
import textwrap
import types

STUB_MODULES = ("colabfold", "colabfold.batch", "alphafold", "alphafold.model", "alphafold.model.modules", "alphafold.model.modules_multimer", "alphafold.model.model", "alphafold.model.config",
                "af2_pallas_attn", "af2_flash_pallas", "stub_pallas_attn_serve", "stub_pallas_trimul", "stub_triattn_xla",
                "stub_pallas_provider")
FAKE_HAIKU_MODULES = ("haiku",)                                                # installed only when no real haiku is importable (the levers' parameter-reader sub-modules)
STUB_ATTN_SERVE = "stub_pallas_attn_serve"                                     # stands in for opt_core.kernels.pallas_attn_serve (msa_attn.SERVE is pointed at it by install())
STUB_TRIATTN = "stub_triattn_xla"                                               # stands in for opt_core.kernels.triattn_xla (triattn_xla.SERVE is pointed at it by install(); its require() replaced by a recorder)
STUB_PROVIDER = "stub_pallas_provider"                                          # stands in for opt_core.kernels.pallas (the JAX-family provider's face: select / family / Refusal / parse_arm / setting_config)
STUB_TRIMUL = "stub_pallas_trimul"                                          # stands in for the provider's call face opt_core.kernels.pallas.serve (trimul_pallas.SERVE is pointed at it by install(); its require() is swapped for a recorder):
                                                                                 # resolve() answers the word with the row its ROW_BY_CLASS table names for the class, triangle_multiplication() records the call
STUB_PSERVE = "stub_pallas_serve"                                               # stands in for opt_core.kernels.pallas.serve (transition.py's faces resolve / transition; transition.SERVE is pointed at it by install(), its require() swapped for a recorder)
FAKE_JAX_MODULES = ("jax", "jax.numpy", "jax.tree_util", "jax.core")       # installed only when no real jax is importable (device_resident's calls on the stub RunModel)
KIT_MODULE_SRC = textwrap.dedent('''\
    """Stand-in for the kit's af2_pallas_attn.py (tests only): the same _STATE shape and enable()/disable() contract, no jax."""
    import os
    _STATE = {"enabled": False, "calls": 0, "fallbacks": 0}
    ENABLE_CALLS = []
    FLASH_OP = [("flash_op", {})]      # the kernel op slot (the real adapter: [K.make_flash_attention()]); triattn_xla.bind swaps [0] for the provider call


    def enable(modules_list=None, all_calls=None):
        if all_calls is None:
            all_calls = os.environ.get("AF_PALLAS_ATTN_ALL", "0") == "1"
        ENABLE_CALLS.append(all_calls)
        mods = []
        try:
            import alphafold.model.modules as m
            mods.append(m)
        except Exception:
            pass
        for m in mods:
            A = m.Attention
            if getattr(A, "_pallas_patched", False):
                continue

            class Attention(A):
                _pallas_patched = True
                _stock_cls = A

                def __call__(self, *a, **k):
                    _STATE["calls"] += 1
                    return ("kernel", a, k)
            m.Attention = Attention
        _STATE["enabled"] = True
        return mods


    def disable(modules_list=None):
        _STATE["enabled"] = False


    if os.environ.get("AF_PALLAS_ATTN", "0") == "1":
        enable()
    ''')
GPU_H100 = {"name": "NVIDIA H100 80GB HBM3", "memory_mib": 81559, "compute_cap": 9.0, "count": 1}
VERSIONS = {"colabfold": "1.6.1", "alphafold_colabfold": "2.3.13", "jax": "0.5.3", "jaxlib": "0.5.3", "dm_haiku": "0.0.16"}


def make_kit_dir(root: str) -> str:
    """A stub kit directory <root>/kit with af2_pallas_flash/af2_pallas_attn.py (the file that identifies a kit to stack.kit_home)."""
    kit = os.path.join(root, "kit")
    d = os.path.join(kit, "af2_pallas_flash")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "af2_pallas_attn.py"), "w", encoding="utf-8") as fh:
        fh.write(KIT_MODULE_SRC)
    return kit


class _AttnConfig(dict):
    """ml_collections-like: attribute access plus .get()."""
    __getattr__ = dict.__getitem__


def make_colabfold(run_log=None):
    """Stub `colabfold` + `colabfold.batch` (run(queries=..., result_dir=..., data_dir=...) records its call; main() calls run) and
    `alphafold.model.modules` with a stock Attention class. Installed into sys.modules; returns the modules."""
    colabfold = types.ModuleType("colabfold"); colabfold.__path__ = []
    batch = types.ModuleType("colabfold.batch")
    log = run_log if run_log is not None else []

    def run(queries, result_dir, num_models=5, is_complex=True, data_dir=None, **kw):
        log.append({"queries": queries, "result_dir": result_dir, "data_dir": data_dir, "kw": kw})
        os.makedirs(str(result_dir), exist_ok=True)
        return "stock-ran"

    def safe_filename(file: str) -> str:                                        # colabfold's own (colabfold/input.py:8-9), the name colabfold.batch carries (batch.py:73-80)
        return "".join([c if c.isalnum() or c in ["_", ".", "-"] else "_" for c in file])

    def run_jobs(queries, result_dir, num_models=5, is_complex=True, data_dir=None, keep_existing_results=True, zip_results=False,
                 jobname_prefix=None, **kw):
        """colabfold.batch.run's job loop with its completion facts exactly and a stand-in model: per query its job name (batch.py:1393-1399),
        a finished job skipped (:1405-1412: <job>.result.zip or <job>.done.txt present and keep_existing_results), the MSA written (:1456), the
        input features pickle when num_models == 0 (:1450-1453), else the ranked files of each model (the stand-in model reaches no lever: what
        tells a run that built a model from one that built none is colabfold's own facts, never a lever's counters), then --zip's archive with
        the files removed and no marker (:1643-1651) or the marker when a model ran (:1653-1654)."""
        import zipfile
        log.append({"queries": queries, "result_dir": result_dir, "data_dir": data_dir, "kw": dict(kw, num_models=num_models, keep_existing_results=keep_existing_results, zip_results=zip_results)})
        res = str(result_dir); os.makedirs(res, exist_ok=True)
        fill = len(str(len(queries)))
        for number, q in enumerate(queries):
            job = (safe_filename(str(jobname_prefix)) + "_" + str(number).zfill(fill)) if jobname_prefix is not None else safe_filename(q[0])
            if keep_existing_results and (os.path.isfile(os.path.join(res, job + ".result.zip")) or os.path.isfile(os.path.join(res, job + ".done.txt"))):
                log.append({"skipped": job}); continue
            written = [job + ".a3m"]; open(os.path.join(res, job + ".a3m"), "w").write("".join(q[2] or []))
            if num_models == 0:
                open(os.path.join(res, job + ".pickle"), "wb").write(b""); written.append(job + ".pickle")
            for r in range(1, int(num_models) + 1):                                  # the stand-in model: its ranked files, no lever traffic
                for f in (f"{job}_unrelaxed_rank_{r:03d}_alphafold2_ptm_model_{r}_seed_000.pdb", f"{job}_scores_rank_{r:03d}_alphafold2_ptm_model_{r}_seed_000.json"):
                    open(os.path.join(res, f), "w").write("ATOM\n" if f.endswith(".pdb") else "{}"); written.append(f)
            log.append({"predicted": job, "models": int(num_models)})
            if zip_results:
                with zipfile.ZipFile(os.path.join(res, job + ".result.zip"), "w") as z:
                    for f in written:
                        z.write(os.path.join(res, f), arcname=f)
                for f in written:
                    os.unlink(os.path.join(res, f))
            elif int(num_models) > 0:
                open(os.path.join(res, job + ".done.txt"), "w").write("")
        return "stock-ran"

    def main(queries=None, result_dir="out", data_dir=None, **flags):
        return batch.run(queries=queries, result_dir=result_dir, num_models=flags.pop("num_models", 5), is_complex=True, data_dir=data_dir, **flags)

    def predict_structure(prefix=None, result_dir=None, feature_dict=None, is_complex=True, use_templates=False, **kw):
        log.append({"predict_structure": prefix, "use_templates": use_templates})
        return "stock-predicted"
    batch.run = run; batch.run_jobs = run_jobs; batch.safe_filename = safe_filename; batch.main = main; batch.predict_structure = predict_structure; batch.RUN_LOG = log
    colabfold.batch = batch
    alphafold = types.ModuleType("alphafold"); alphafold.__path__ = []
    model = types.ModuleType("alphafold.model"); model.__path__ = []
    modules = types.ModuleType("alphafold.model.modules")

    class Attention:
        def __init__(self, config=None, global_config=None, output_dim=64, name="attention"):
            self.config = config if config is not None else _AttnConfig(num_head=8, key_dim=256)
            self.global_config, self.output_dim, self.name = global_config, output_dim, name

        def __call__(self, *a, **k):
            return ("stock", a, k)
    modules.Attention = Attention

    class TriangleMultiplication:
        """Stand-in for modules.TriangleMultiplication: (config, global_config, name); the stock body returns ("stock", act, mask)."""
        def __init__(self, config=None, global_config=None, name=None):
            self.config, self.global_config, self.name = config, global_config, name

        def __call__(self, left_act, left_mask, is_training=True):
            return ("stock", left_act, left_mask)
    modules.TriangleMultiplication = TriangleMultiplication

    class Transition:
        """Stand-in for modules.Transition: (config, global_config, name); the stock body returns ("stock", act, mask)."""
        def __init__(self, config=None, global_config=None, name="transition_block"):
            self.config = config if config is not None else types.SimpleNamespace(num_intermediate_factor=4)
            self.global_config, self.name = global_config, name

        def __call__(self, act, mask, is_training=True):
            return ("stock", act, mask)
    modules.Transition = Transition
    multimer = types.ModuleType("alphafold.model.modules_multimer")             # the multimer module: TemplateEmbedding (templ_dedup rebinds it) + the names its subclass body reads at build (prng, common_modules)

    class TemplateEmbedding:
        """Stand-in for modules_multimer.TemplateEmbedding: (config, global_config, name); the stock body returns ("stock", template_batch)."""
        def __init__(self, config=None, global_config=None, name="template_embedding"):
            self.config, self.global_config, self.name = config, global_config, name

        def __call__(self, query_embedding, template_batch, padding_mask_2d, multichain_mask_2d, is_training, safe_key=None):
            return ("stock", template_batch)
    multimer.TemplateEmbedding = TemplateEmbedding
    multimer.SingleTemplateEmbedding = type("SingleTemplateEmbedding", (), {})
    multimer.prng = types.SimpleNamespace(SafeKey=lambda key: types.SimpleNamespace(_key=key, split=lambda: (None, None)))
    multimer.common_modules = types.SimpleNamespace(Linear=None)
    config_mod = types.ModuleType("alphafold.model.config")

    def model_config(name):
        """Stand-in for alphafold.model.config.model_config: a fresh tree with global_config.subbatch_size = 4 (stock's value)."""
        gc = types.SimpleNamespace(subbatch_size=4, bfloat16=True)
        return types.SimpleNamespace(name=name, model=types.SimpleNamespace(global_config=gc))
    config_mod.model_config = model_config
    config_mod.CALLS = []
    tserve = types.ModuleType(STUB_TRIMUL)                                       # the provider's triangle-multiplication call face trimul_pallas.py uses, no kernels: resolve() names the arm the stub's table serves for the
    tserve.EQ_OUT, tserve.EQ_IN = "ikc,jkc->ijc", "kjc,kic->ijc"                  # class (tests set ROW_BY_CLASS / REFUSE), triangle_multiplication() records its call and returns a cast-able stand-in
    tserve.CALLS = []; tserve.RESOLVE_CALLS = []; tserve.REQUIRE_CALLS = []; tserve.COUNTS = {}
    tserve.CC = "9.0"                                                            # tests set "8.0" / "8.6" (engages) or "7.5" (the floor refuses by name)
    tserve.ROW = "cd_trimul"                                                     # the arm a tier word resolves to unless ROW_BY_CLASS names another for "C<cz>E<0|1>" ("xla" = the stock statement by name)
    tserve.ROW_BY_CLASS = {}; tserve.REFUSE = {}                                 # REFUSE: class word -> refusal kind raised by resolve()

    def _resolve(op, fam, dtype, n_tokens, *, word, direction="fwd", jax_line=None, cc=None, prefer=None, **kw):
        prov_ = sys.modules.get(STUB_PROVIDER)
        m = __import__("re").match(r"^af2_(pair|tmpl)_c(\d+)_ch(\d+)_(outgoing|incoming)$", fam)
        cls = "C%sE%d" % (m.group(2), 0 if m.group(4) == "outgoing" else 1) if m else fam
        tserve.RESOLVE_CALLS.append({"op": op, "fam": fam, "dtype": str(getattr(dtype, "name", dtype)), "n": int(n_tokens), "word": word, "cc": cc, "direction": direction})
        if cls in tserve.REFUSE:
            raise prov_.Refusal("%s: %s" % (word, tserve.REFUSE[cls]), kind=tserve.REFUSE[cls], row=tserve.ROW, fallback="xla")
        row = tserve.ROW_BY_CLASS.get(cls, tserve.ROW)
        bucket = min([b for b in (400, 800, 1200) if b >= int(n_tokens)] or [1200])
        key = None if row == "xla" else "0.5|%s|bf16|trimul|%s|N<=%d|%s" % (cc or tserve.CC, fam, bucket, direction)
        return types.SimpleNamespace(row=row.split("@")[0], arm=row, config={}, cell_key=key, cls=("stock" if row == "xla" else "tol"), word=word, candidates=[row, "xla"] if row != "xla" else ["xla"],
                                     note="" if key else "no measured cell: the stock statement, by name")
    tserve.resolve = _resolve

    def triangle_multiplication(act, mask, params, *, equation, form="af2", unit=None, word, direction="fwd", cc=None, selection=None, **kw):
        arm = getattr(selection, "arm", None) or tserve.ROW
        tserve.CALLS.append({"shape": tuple(act.shape), "equation": equation, "form": form, "unit": unit, "word": word, "direction": direction, "params": params,
                             "mask_shape": tuple(mask.shape), "arm": arm})
        k = "served:triangle_multiplication:%s" % arm
        tserve.COUNTS[k] = tserve.COUNTS.get(k, 0) + 1
        return types.SimpleNamespace(kind="pallas", equation=equation, astype=lambda d: ("pallas", equation, d))
    tserve.triangle_multiplication = triangle_multiplication
    txs = types.ModuleType(STUB_TRIATTN)                                         # the core's triangle-attention bridge, no kernels: select() names the row its table would, triangle_attention records the call
    txs.__version__ = "1.1.0"; txs.CALLS = []; txs.REQUIRE_CALLS = []
    txs.CC = "9.0"                                                               # tests set "8.0" (k2b_aot row) or a cc the bridge has no row for

    class Refused(RuntimeError):
        def __init__(self, msg="", *, reasons=None):
            super().__init__(msg); self.reasons = dict(reasons or {}); self.fallback = "stock: xla"
    txs.Refused = Refused

    def select(cc, dtype, head_dim, S, *, N=1, H=4, B=1, has_mask=True, bias_dtype=None, impl="auto"):
        cc = str(cc)
        if cc == "9.0":
            return ("cuda_sm90a" if int(head_dim) == 32 and int(S) >= 384 else "k2b_aot"), {}
        if cc.startswith("8."):
            return "k2b_aot", {}
        raise Refused(f"triattn_xla: no kernel row serves this call — cuda_sm90a: compute capability {cc}; k2b_aot: no cubins for {cc}",
                      reasons={"cuda_sm90a": f"compute capability {cc}", "k2b_aot": f"no cubins for {cc}"})
    txs.select = select
    txs.compute_capability = lambda device=None: txs.CC

    def triangle_attention(q, k, v, bias, mask=None, scale=None, *, impl="auto", layout="BNHSD", vjp=None, cc=None, return_row=False):
        row, _ = select(cc or txs.CC, getattr(q, "dtype", "bf16"), q.shape[-1], q.shape[-3] if layout == "BNSHD" else q.shape[-2], N=q.shape[0], H=(q.shape[-2] if layout == "BNSHD" else q.shape[-3]))
        txs.CALLS.append({"shape": tuple(q.shape), "layout": layout, "row": row, "mask": None if mask is None else tuple(mask.shape), "bias": tuple(bias.shape)})
        return (q, row) if return_row else q
    txs.triangle_attention = triangle_attention
    aserve = types.ModuleType(STUB_ATTN_SERVE)                                   # the core's served attention path, no kernel: a call with >= min_keys keys is "served"
    aserve.MIN_HEAD_DIM = 16; aserve.BELOW_KEYS_RULE = "below_keys_rule"; aserve.KERNEL_FILE = "af2_flash_pallas"; aserve.REQUIRE_CALLS = []
    def _kernel_module(require_gpu=False):                                       # a stand-in kernel module: the adapter's FLASH_OP factory
        km = sys.modules.get(aserve.KERNEL_FILE)
        if km is None:
            km = types.ModuleType(aserve.KERNEL_FILE)
        if not hasattr(km, "make_flash_attention"):
            km.make_flash_attention = lambda **cfg: ("flash_op", dict(cfg))
        return km
    aserve.kernel_module = _kernel_module
    aserve.require = lambda require_gpu=True: aserve.REQUIRE_CALLS.append(require_gpu) or {"ok": True}

    class _Ledger:
        def __init__(self, expected=()):
            self.expected, self.served, self.fallbacks, self.shapes = tuple(expected), 0, {}, {}

        def fallback(self, why):
            self.fallbacks[why] = self.fallbacks.get(why, 0) + 1

        def serve(self, shape=None):
            self.served += 1
    aserve.ledger = lambda expected=(), **kw: _Ledger(expected)

    def served_attention_call(self, q_data, m_data, bias, nonbatched_bias=None, *, all_calls=True, ledger=None, min_keys=0, pad_head_dim_below_min=False, **kw):
        if min_keys and int(m_data.shape[-2]) < int(min_keys):
            ledger.fallback(aserve.BELOW_KEYS_RULE); return None
        ledger.serve(); return ("served", q_data, m_data)
    aserve.served_attention_call = served_attention_call
    aserve.make_flash_attention = lambda **cfg: ("flash_op", dict(cfg))
    prov = types.ModuleType(STUB_PROVIDER)                                       # the provider's face, no kernels: select() records the cell it was asked and refuses BY NAME what REFUSE names
    prov.CALLS = []; prov.REFUSE = {}                                            # REFUSE: (op, cc, dtype word) | op -> refusal kind

    class PRefusal(RuntimeError):
        def __init__(self, msg="", *, kind=None, row=None, fallback=None):
            super().__init__(msg); self.kind, self.row, self.fallback = kind, row, fallback
    prov.Refusal = PRefusal
    prov.TIER_WORDS = ("fast", "exact", "big"); prov.STOCK_ROWS = ("xla", "xla_subbatch4", "xla_sdpa", "cudnn")   # the real provider's words (opt_core.kernels.pallas TIER_WORDS / STOCK_ROWS)

    def _prov_family(op, fam=None, **shape):
        if fam is not None and not shape:
            return fam
        if op == "transition":
            return "%s_%s_c%d_x%d" % (shape.get("form", "af2"), shape.get("activation", "relu"), int(shape["c"]), int(shape["factor"]))
        if op == "trimul":                                                       # the real face's word: the einsum string -> outgoing | incoming
            eq = {"ikc,jkc->ijc": "outgoing", "kjc,kic->ijc": "incoming"}.get(shape.get("equation"), shape.get("equation"))
            return "%s_%s_c%d_ch%d_%s" % (shape.get("form", "af2"), shape.get("unit", "pair"), int(shape["c"]), int(shape["c_hidden"]), eq)
        return "%s_%s" % (op, "_".join("%s%s" % kv for kv in sorted(shape.items())))
    prov.family = _prov_family

    def _prov_select(jax_line, cc, dtype, op, fam, n_tokens, direction="fwd", *, word, prefer=None, **kw):   # its own name: `select` above is the bridge stub's, closed over by its triangle_attention
        dt = str(getattr(dtype, "name", dtype)); dt = {"bfloat16": "bf16", "float32": "f32"}.get(dt, dt)
        prov.CALLS.append({"jax": jax_line, "cc": None if cc is None else str(cc), "dtype": dt, "op": op, "fam": fam, "n": int(n_tokens), "word": word})
        kind = prov.REFUSE.get((op, None if cc is None else str(cc), dt)) or prov.REFUSE.get(op)
        if kind:
            raise PRefusal("%s: %s" % (word, kind), kind=kind, row=word, fallback="xla")
        return types.SimpleNamespace(arm=word, row=word, config={}, setting=None)
    prov.select = _prov_select

    def _prov_parse_arm(arm):
        row, _, st = arm.partition("@")
        return row, None, ("%s@%s" % (row, st) if st else None)
    prov.parse_arm = _prov_parse_arm
    prov.SETTINGS = {"pallas_attn@bq64bk32": {"bq": 64, "bk": 32}, "pallas_attn@s3": {"num_stages": 3}}
    prov.setting_config = lambda setting: dict(prov.SETTINGS[setting]) if setting else {}
    model_mod = types.ModuleType("alphafold.model.model")

    class RunModel:
        """Stand-in for alphafold.model.model.RunModel: `apply(params, key, batch)` returns a result tree of "device" arrays computed from
        its inputs (the sum of the parameter leaves added to `prev`), so a resident wrapper's outputs can be checked against the stock call."""
        def __init__(self, config=None, params=None):
            self.config, self.params = config, params
            self.APPLY_ARGS = []

            def apply(params, key, batch):
                import numpy as np
                jax = sys.modules.get("jax")
                dev = (lambda v: jax.device_put(v)) if jax is not None else (lambda v: v)
                self.APPLY_ARGS.append((params, key, batch))
                total = sum(float(np.asarray(v).sum()) for v in _leaves(params))
                prev = batch.get("prev") or {}
                pair = np.asarray(prev.get("prev_pair", np.zeros((2, 2), np.float32)), np.float32)
                return {"prev": {"prev_pair": dev(pair + np.float32(total)), "prev_pos": dev(np.asarray(prev.get("prev_pos", np.zeros((2, 3))), np.float32) + 1)},
                        "ranking_confidence": dev(np.asarray(0.25 + total, np.float32)), "distogram": {"logits": dev(np.full((2, 2, 3), 1 / 3, np.float32))},
                        "aatype": dev(np.arange(2, dtype=np.int32)), "tol": dev(np.asarray(1.0 / (1.0 + float(pair.sum())), np.float32)),
                        "mean_plddt": dev(np.asarray(80.0 + float(pair.sum()), np.float32)),
                        "structure_module": {"final_atom_positions": dev(pair[:, :, None] + np.arange(3, dtype=np.float32))}}
            self.apply = apply

        STOP_AT_SCORE, EARLY_STOP_TOLERANCE, NUM_RECYCLE = 100.0, 0.0, 3

        def predict(self, feat, random_seed=0, return_representations=False, callback=None):
            """alphafold/model/model.py:123-208 (RunModel.predict), the multimer route, line for line where it touches `result` / `prev`."""
            import numpy as np
            num_iters = self.NUM_RECYCLE + 1
            L = 2
            zeros = lambda shape: np.zeros(shape, dtype=np.float16)  # noqa: E731
            prev = {"prev_pair": zeros([L, L]), "prev_pos": np.zeros([L, 3])}

            def run(key, feat, prev):
                def _jnp_to_np(x):
                    for k, v in x.items():
                        if isinstance(v, dict):
                            x[k] = _jnp_to_np(v)
                        else:
                            x[k] = np.asarray(v, np.float16)
                    return x
                result = _jnp_to_np(self.apply(self.params, key, {**feat, "prev": prev}))
                prev = result.pop("prev")
                return result, prev

            for r in range(num_iters):
                result, prev = run(None, feat, prev)
                if return_representations:
                    result["representations"] = {"pair": prev["prev_pair"], "single": prev["prev_pos"]}
                if callback is not None: callback(result, r)  # noqa: E701
                if result["ranking_confidence"] > self.STOP_AT_SCORE:
                    break
                if r > 0 and result["tol"] < self.EARLY_STOP_TOLERANCE:
                    break
            return result, r
    model_mod.RunModel = RunModel
    alphafold.model = model; model.modules = modules; model.modules_multimer = multimer; model.model = model_mod; model.config = config_mod
    mods = {"colabfold": colabfold, "colabfold.batch": batch, "alphafold": alphafold, "alphafold.model": model, "alphafold.model.modules": modules, "alphafold.model.modules_multimer": multimer,
            "alphafold.model.model": model_mod, "alphafold.model.config": config_mod, STUB_ATTN_SERVE: aserve, STUB_TRIMUL: tserve, STUB_TRIATTN: txs,
            STUB_PROVIDER: prov}
    return mods


def _leaves(tree):
    if isinstance(tree, dict):
        for v in tree.values():
            yield from _leaves(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            yield from _leaves(v)
    else:
        yield tree


def make_fake_jax():
    """A stand-in `jax` for boxes without one (device_resident imports jax inside the rebound RunModel): device arrays are an ndarray
    subclass, device_put / device_get copy, jit is the identity, tree_util walks dicts / lists / tuples."""
    import numpy as np
    jax = types.ModuleType("jax"); jax.__path__ = []
    jnp = types.ModuleType("jax.numpy"); tu = types.ModuleType("jax.tree_util"); core = types.ModuleType("jax.core")

    class Array(np.ndarray):
        pass

    class Tracer:
        pass

    def device_put(v):
        a = np.array(v, copy=True).view(Array); jax.PUTS.append(a); return a

    def tree_map(f, tree):
        if isinstance(tree, dict):
            return {k: tree_map(f, v) for k, v in tree.items()}
        if isinstance(tree, (list, tuple)):
            return type(tree)(tree_map(f, v) for v in tree)
        return f(tree)

    def device_get(tree):
        return tree_map(lambda v: np.asarray(v).copy() if isinstance(v, np.ndarray) else v, tree)
    jax.Array, jax.PUTS, jax.device_put, jax.device_get, jax.jit = Array, [], device_put, device_get, (lambda f: f)
    tu.tree_map, tu.tree_leaves = tree_map, (lambda tree: list(_leaves(tree)))
    core.Tracer = Tracer; jnp.float16 = np.float16; jnp.asarray = np.asarray
    jax.numpy, jax.tree_util, jax.core = jnp, tu, core
    return {"jax": jax, "jax.numpy": jnp, "jax.tree_util": tu, "jax.core": core}


def _real_jax() -> bool:
    """A real jax is importable (the stand-in carries PUTS; a real module does not)."""
    m = sys.modules.get("jax")
    if m is not None:
        return getattr(m, "PUTS", None) is None
    import importlib.util
    try:
        return importlib.util.find_spec("jax") is not None
    except (ImportError, ValueError):
        return False


def make_fake_haiku():
    """A stand-in `haiku` for boxes without one: Module (name-carrying base class) and get_parameter returning a tagged tuple."""
    hk = types.ModuleType("haiku")

    class Module:
        def __init__(self, name=None):
            self.name = name
    hk.Module = Module
    class _Param(tuple):                                                         # a tagged tuple that also answers a slice / index expression with itself tagged (a reader that splits a fused weight: w[:, :C])
        def __getitem__(self, item):
            return _Param(tuple.__add__(self, (("slice", str(item)),))) if isinstance(item, (tuple, slice)) else tuple.__getitem__(self, item)
    hk.get_parameter = lambda name, shape, dtype=None, init=None: _Param(("param", name, tuple(shape), dtype))
    hk.initializers = types.SimpleNamespace(VarianceScaling=lambda *a, **kw: ("init", "variance_scaling"), Constant=lambda v: ("init", v))   # inert under inference (parameters are loaded, never initialised)
    return {"haiku": hk}


def _real_haiku() -> bool:
    try:
        import importlib.util as _u
        return _u.find_spec("haiku") is not None
    except (ImportError, ValueError):
        return False


def make_stub_pallas_serve(prov):
    """A stand-in for the provider's call faces ``opt_core.kernels.pallas.serve`` as transition.py uses them, no kernels: ``resolve()`` answers the
    tier word from ROW_BY_FAMILY (the shape of the shared core's AF2 transition cells: a fused row at pair c128 / template c64, the stock
    statement at MSA c256, no cell at extra-MSA c64 x4; ``exact`` and f32 resolve to the stock statement; a ``pallas:<row>`` word in
    MODEL_OPT_LEVERS_OFF steps that row aside) and records the question; ``transition()`` records the call and returns a cast-able stand-in, or
    raises the provider's Refusal for a family REFUSE_AT_CALL names."""
    m = types.ModuleType(STUB_PSERVE)
    m.CC = "9.0"; m.REQUIRE_CALLS = []; m.RESOLVE_CALLS = []; m.CALLS = []
    m.Refusal = prov.Refusal
    m.ROW_BY_FAMILY = {"af2_relu_c128_x4": ("mlp_transition", "0.5|9.0|bf16|transition|af2_relu_c128_x4|N<=800|fwd"),
                       "af2_relu_c256_x4": ("xla", "0.5|9.0|bf16|transition|af2_relu_c256_x4|N<=800|fwd"),
                       "af2_relu_c64_x2": ("cd_transition", "0.5|9.0|bf16|transition|af2_relu_c64_x2|N<=800|fwd")}   # any other family: no measured cell -> the stock statement
    m.REFUSE_AT_CALL = {}                                                        # family -> the refusal kind transition() raises

    class Served:
        def __init__(self, row, x):
            self.row, self.x = row, x

        def astype(self, dt):
            return ("served", self.row, self.x, dt)

    def resolve(op, fam, dtype, n_tokens, *, word, direction="fwd", jax_line=None, cc=None, prefer=None, key_masked=True):
        dt = str(getattr(dtype, "name", dtype)); dt = {"bfloat16": "bf16", "float32": "f32"}.get(dt, dt)
        m.RESOLVE_CALLS.append({"op": op, "fam": fam, "dtype": dt, "n": int(n_tokens), "word": word, "direction": direction, "cc": None if cc is None else str(cc), "jax": jax_line})
        row, key = m.ROW_BY_FAMILY.get(fam, ("xla", None))
        if word == "exact" or dt != "bf16":
            row = "xla"
        off = [w.strip() for w in (os.environ.get("MODEL_OPT_LEVERS_OFF") or "").split(",")]
        if row != "xla" and ("pallas" in off or "pallas:%s" % row in off):
            row = "xla"                                                          # the provider's own word: that row steps aside, the next candidate (here the stock statement) serves
        cands = [row] + (["xla"] if row != "xla" else [])
        return types.SimpleNamespace(op=op, family=fam, row=row, arm=row, candidates=cands, config={}, input_precision=None, direction=direction, dtype=dt,
                                     cc=None if cc is None else str(cc), jax_line=jax_line, word=word, note="",
                                     cell=({"fast": row, "exact": "xla", "big": row} if key else None), cell_key=key,
                                     cls=(("stock" if row == "xla" else "tol") if key else None), x_stock=((1.0 if row == "xla" else 1.8) if key else None))
    m.resolve = resolve

    def transition(x, params, *, activation, form="af2", word, direction="fwd", n_tokens=None, jax_line=None, cc=None, prefer=None, strict=False,
                   input_precision=None, config=None, selection=None):
        sel = selection if selection is not None else resolve("transition", "af2_%s_c%d_x%d" % (activation, int(x.shape[-1]), int(params["w1"][2][1]) // int(x.shape[-1])), x.dtype, n_tokens or x.shape[-2], word=word, cc=cc)
        kind = m.REFUSE_AT_CALL.get(sel.family)
        if kind:
            raise m.Refusal("%s: %s" % (sel.row, kind), kind=kind, row=sel.row, fallback="xla")
        m.CALLS.append({"row": sel.row, "word": word, "shape": tuple(int(v) for v in x.shape), "n": n_tokens, "activation": activation, "form": form, "strict": strict,
                        "params": {k: (tuple(v[2]) if isinstance(v, tuple) else tuple(getattr(v, "shape", ()))) for k, v in params.items()},
                        "dtypes": {k: (v[3] if isinstance(v, tuple) else getattr(v, "dtype", None)) for k, v in params.items()}})
        return Served(sel.row, x)
    m.transition = transition
    m.report = lambda: {"counts": {}, "last_refusal": {}}
    return m


def install(run_log=None):
    """Install the stub colabfold / alphafold modules and the stub serve layer (and the stand-in jax / haiku when no real one is importable),
    point the package's lever modules at them; returns (modules, saved)."""
    names = STUB_MODULES + (() if _real_jax() else FAKE_JAX_MODULES) + (() if _real_haiku() else FAKE_HAIKU_MODULES)
    saved = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules.pop(n, None)
    mods = make_colabfold(run_log)
    if not _real_jax():
        mods.update(make_fake_jax())
    if not _real_haiku():
        mods.update(make_fake_haiku())
    sys.modules.update(mods)
    from colabfold_opt import msa_attn, subbatch
    saved["__msa_serve__"] = msa_attn.SERVE
    msa_attn.SERVE = STUB_ATTN_SERVE
    from colabfold_opt import stack as _stack
    saved["__kernel_serve__"] = _stack.KERNEL_SERVE; _stack.KERNEL_SERVE = STUB_ATTN_SERVE
    from colabfold_opt import msa_col_cudnn, templ_dedup, trimul_pallas
    saved["__trimul__"] = (trimul_pallas.SERVE, trimul_pallas.require, trimul_pallas.PROVIDER)
    trimul_pallas.SERVE = STUB_TRIMUL; trimul_pallas.PROVIDER = STUB_PROVIDER
    tserve = mods[STUB_TRIMUL]; prov = mods[STUB_PROVIDER]
    tserve.ROW_BY_CLASS.clear(); tserve.REFUSE.clear()

    def _require(word="fast"):                                                   # the floor's recorder: the part class from the stub's CC (trimul_pallas.require's cc_below_8_0 rule kept) and the word asked
        tserve.REQUIRE_CALLS.append(tserve.CC)
        t = trimul_pallas._cc_tuple(tserve.CC)
        if t is not None and t < trimul_pallas.MIN_CC:
            raise trimul_pallas.Refusal("cc_below_8_0", f"compute capability {tserve.CC}: the bf16 Pallas rows need an sm_80-class part or newer")
        if word not in ("fast", "exact", "big"):
            raise trimul_pallas.Refusal("unknown_word", word)
        return {"jax": VERSIONS["jax"], "backend": "gpu", "cc": tserve.CC, "core": "stub"}
    trimul_pallas.require = _require
    saved["__col__"] = (msa_col_cudnn.require,)
    msa_col_cudnn.CC = "9.0"; msa_col_cudnn.REQUIRE_CALLS = []                  # test knobs on the module (removed by remove()): the part the floor reports

    def _require_col():
        msa_col_cudnn.REQUIRE_CALLS.append(msa_col_cudnn.CC)
        t = msa_col_cudnn._cc_tuple(msa_col_cudnn.CC)
        if t is not None and t < msa_col_cudnn.MIN_CC:
            raise msa_col_cudnn.Refusal("cc_below_8_0", f"compute capability {msa_col_cudnn.CC}")
        return {"jax": VERSIONS["jax"], "backend": "gpu", "cc": msa_col_cudnn.CC, "cudnn": "stub"}
    msa_col_cudnn.require = _require_col
    from colabfold_opt import triattn_xla
    saved["__triattn__"] = (triattn_xla.SERVE, triattn_xla.require)
    triattn_xla.SERVE = STUB_TRIATTN
    txs = mods[STUB_TRIATTN]

    def _require_tx():                                                           # the floor's recorder (triattn_xla.require: the bridge importable, its launcher loadable, the GPU backend — none on this box)
        txs.REQUIRE_CALLS.append(txs.CC)
        return {"cc": txs.CC, "core": txs.__version__, "jax": VERSIONS["jax"]}
    triattn_xla.require = _require_tx
    triattn_xla.reset_for_tests()
    # ---- TRANSITION (transition.py over the provider face by tier word): its faces -> the stub serve module, its floor -> a recorder
    from colabfold_opt import transition
    saved[STUB_PSERVE] = sys.modules.get(STUB_PSERVE)
    pss = make_stub_pallas_serve(mods[STUB_PROVIDER]); mods[STUB_PSERVE] = pss; sys.modules[STUB_PSERVE] = pss
    saved["__transition__"] = (transition.SERVE, transition.PROVIDER, transition.require)
    transition.SERVE, transition.PROVIDER = STUB_PSERVE, STUB_PROVIDER

    def _require_tr():                                                           # the floor's recorder (transition.require: jax's GPU backend + the provider face — none on this box)
        pss.REQUIRE_CALLS.append(pss.CC)
        return {"jax": VERSIONS["jax"], "backend": "gpu", "cc": pss.CC, "core": "stub"}
    transition.require = _require_tr
    transition.reset_for_tests()
    # ---- end TRANSITION
    subbatch.reset_for_tests(); msa_attn.reset_for_tests(); trimul_pallas.reset_for_tests(); templ_dedup.reset_for_tests()
    msa_col_cudnn.reset_for_tests()
    return mods, saved


def remove(saved):
    from colabfold_opt import msa_attn, msa_col_cudnn, subbatch, triattn_xla, templ_dedup, trimul_pallas
    subbatch.reset_for_tests(); msa_attn.reset_for_tests(); trimul_pallas.reset_for_tests(); templ_dedup.reset_for_tests(); triattn_xla.reset_for_tests()
    msa_col_cudnn.reset_for_tests()
    saved = dict(saved)
    tr = saved.pop("__transition__", None)                                       # ---- TRANSITION
    if tr is not None:
        from colabfold_opt import transition
        transition.reset_for_tests(); transition.SERVE, transition.PROVIDER, transition.require = tr
    tx = saved.pop("__triattn__", None)
    if tx is not None:
        triattn_xla.SERVE, triattn_xla.require = tx
    tk = saved.pop("__trimul__", None)
    if tk is not None:
        trimul_pallas.SERVE, trimul_pallas.require, trimul_pallas.PROVIDER = tk
    cl = saved.pop("__col__", None)
    if cl is not None:
        (msa_col_cudnn.require,) = cl
    aserve = saved.pop("__msa_serve__", None)
    if aserve is not None:
        msa_attn.SERVE = aserve
    ks = saved.pop("__kernel_serve__", None)
    if ks is not None:
        from colabfold_opt import stack as _stack
        _stack.KERNEL_SERVE = ks
    for n in saved:
        sys.modules.pop(n, None)
        if saved.get(n) is not None:
            sys.modules[n] = saved[n]


def gates_pass(stack, gpu=GPU_H100, versions=VERSIONS):
    """Monkeypatch the gates that need an installed colabfold, real parameters and a GPU; returns what to hand gates_restore."""
    saved = (stack.pins_check, stack.weights_check, stack.gpu_info, stack.versions, stack.cuda_runtime)
    stack.pins_check = lambda: ([], {"colabfold": {"version": "1.6.1", "pinned": "1.6.1", "ok": True},
                                     "alphafold_colabfold": {"version": "2.3.13", "pinned": "2.3.13", "ok": True}, "files": {}})
    stack.weights_check = lambda data_dir, refresh=False: ([], {"data_dir": data_dir, "ok": True, "files": {}, "refresh": refresh})   # refresh: the `check` verb hashes afresh (recorded for the tests)
    stack.gpu_info = lambda index=0: dict(gpu) if gpu else None
    stack.versions = lambda: dict(versions)
    stack.cuda_runtime = lambda: "12.9"
    return saved


def gates_restore(stack, saved):
    stack.pins_check, stack.weights_check, stack.gpu_info, stack.versions, stack.cuda_runtime = saved


def queries(*lengths):
    """colabfold query tuples (jobname, sequence, a3m_lines, templates) of the given total lengths (colabfold/input.py:301)."""
    return [(f"q{i}", "A" * n, [f"#{n}\t1\n>101\n{'A' * n}\n"], None) for i, n in enumerate(lengths)]
