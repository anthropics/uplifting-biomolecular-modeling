"""Shared fixtures for the CPU tests: where the kit is, a stub upstream (esm / transformers / torch) that needs no GPU, and a
stub kit directory whose server carries the REAL server's MODES table (copied at test time from the kit, never typed in)
with a configure() that records its calls instead of patching a model.

The kit is found through the package's own lookup (stack.kit_home(): opt/<class>/*/driver/ef2_server.py, or
$ESMFOLD2_OPT_KIT); tests that need it call `require_kit()` and SKIP with a printed reason when it is absent (the kit files are
carried by the tree, not by this package)."""
import ast
import json
import os
import sys
import textwrap
import unittest

from esmfold2_opt import stack


def kit_or_none():
    try:
        return stack.kit_home()
    except FileNotFoundError:
        return None


def require_kit():
    k = kit_or_none()
    if k is None:
        raise unittest.SkipTest("kit not found (no opt/<class>/*/driver/ef2_server.py and no ESMFOLD2_OPT_KIT): kit-dependent test skipped")
    return k


def pins_or_none():
    p = stack.pins_path()
    return json.load(open(p)) if os.path.isfile(p) else None


def require_pins():
    p = pins_or_none()
    if p is None:
        raise unittest.SkipTest(f"stock/PINS.json not found at {stack.pins_path()} (set MODEL_OPT to the esmfold2/ directory): test skipped")
    return p


STUB_UPSTREAM = {
    "torch/__init__.py": textwrap.dedent('''
        __version__ = "0.0.0+stub"
        class _Cuda:
            @staticmethod
            def is_available(): return False
            @staticmethod
            def is_initialized(): return False
            @staticmethod
            def empty_cache(): pass
        cuda = _Cuda()
        def use_deterministic_algorithms(mode, warn_only=False):
            global _DET
            _DET = (mode, warn_only)
        _DET = None
        def are_deterministic_algorithms_enabled(): return bool(_DET and _DET[0])
        def is_deterministic_algorithms_warn_only_enabled(): return bool(_DET and _DET[1])
        def equal(a, b): return a == b
        class Tensor: pass
        bfloat16 = "bf16"
    '''),
    "numpy_stub_marker.txt": "",
    "esm/__init__.py": "",
    "esm/models/__init__.py": "",
    "esm/models/esmfold2/__init__.py": "",
    "esm/models/esmfold2/processor.py": textwrap.dedent('''
        import json, os
        CALLS = []
        class _T:
            """A stand-in tensor: the two calls the kit's writer and loop make (.numpy(), [0], .cpu(), .sum().item())."""
            def __init__(self, a):
                import numpy as np
                self._a = np.asarray(a)
            def numpy(self): return self._a
            def detach(self): return self
            def cpu(self): return self
            def __getitem__(self, i): return _T(self._a[i])
            def sum(self): return _T(self._a.sum())
            def item(self): return self._a.item()
        def _token_features(spi):
            import numpy as np
            asym, mol = [], []
            for i, s in enumerate(spi.sequences):
                n = len(s.sequence) if getattr(s, "sequence", None) else 1
                asym += [i + 1] * n; mol += [0 if getattr(s, "sequence", None) else 1] * n
            n = len(asym)
            return {"asym_id": _T(np.array([asym])), "mol_type": _T(np.array([mol])), "token_attention_mask": _T(np.ones((1, n), dtype=np.int64))}
        class _Result:
            """A stand-in MolecularComplexResult with deterministic content derived from the input."""
            def __init__(self, spi, seed, n, complex_id):
                import numpy as np
                L = sum(len(s.sequence) for s in spi.sequences if getattr(s, "sequence", None)) + sum(1 for s in spi.sequences if not getattr(s, "sequence", None))
                rng = np.random.default_rng(seed * 1000 + L)
                self.plddt = _T(rng.random(L, dtype=np.float32))
                self.pae = _T(rng.random((L, L), dtype=np.float32) * 30.0)
                self.pde = None
                self.distogram = None
                self.ptm = float(rng.random()); self.iptm = float(rng.random())
                self.pair_chains_iptm = rng.random((len(spi.sequences), len(spi.sequences)), dtype=np.float32)
                self.residue_index = np.arange(L, dtype=np.int64)
                self.entity_id = np.zeros(L, dtype=np.int64)
                self.num_tokens = L
                class _MD:
                    chain_lookup = {i: s.id for i, s in enumerate(spi.sequences)}
                class _C:
                    metadata = _MD()
                    def to_mmcif(_self):
                        return f"data_{complex_id}\\n# stub mmCIF seed={seed} L={L}\\n"
                self.complex = _C()
        class ESMFold2InputBuilder:
            def __init__(self, ccd_cache=None):
                self.ccd_cache = ccd_cache
            def prepare_input(self, input, seed=None, device=None):
                return _token_features(input), []
            def fold(self, model, input, *, num_loops=20, num_sampling_steps=200, num_diffusion_samples=1, seed=None, noise_scale=None,
                     step_scale=None, max_inference_sigma=None, lm_mask_pct=None, early_exit=False, lm_dropout=0.3, msa_max_depth=1024,
                     msa_column_mask_rate=0.1, complex_id="pred"):
                CALLS.append(dict(model=id(model), seed=seed, num_loops=num_loops, num_sampling_steps=num_sampling_steps,
                                  num_diffusion_samples=num_diffusion_samples, lm_dropout=lm_dropout, msa_max_depth=msa_max_depth,
                                  msa_column_mask_rate=msa_column_mask_rate, complex_id=complex_id,
                                  msa_depths=[getattr(getattr(s, "msa", None), "depth", 0) or 0 for s in input.sequences]))
                if os.environ.get("STUB_RECORD"):                                      # the tests' record of what reached the stock API (one JSON line per fold call)
                    rec = dict(CALLS[-1], event="fold", model=None, cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"), scatter_switch=os.environ.get("ESMFOLD2_DETERMINISTIC_SCATTER"),
                               kernel_backend=getattr(model, "kernel_backend", "unread"), chunk=getattr(model, "chunk", "unread"))
                    try:
                        import torch; rec["deterministic_algorithms"] = [torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()]
                    except Exception:  # noqa: BLE001
                        rec["deterministic_algorithms"] = None
                    with open(os.environ["STUB_RECORD"], "a") as fh: fh.write(json.dumps(rec, default=str) + "\\n")
                if complex_id == os.environ.get("STUB_FOLD_FAIL_ON"):                  # a fold that dies (the tests' stand-in for an OOM)
                    raise RuntimeError(f"CUDA out of memory (stub) on {complex_id}")
                if num_diffusion_samples == 1:
                    return _Result(input, seed or 0, 0, complex_id)
                return [_Result(input, seed or 0, i, complex_id) for i in range(num_diffusion_samples)]
        __all__ = ["ESMFold2InputBuilder"]
    '''),
    "esm/utils/__init__.py": "",
    "esm/utils/msa/__init__.py": "from .msa import MSA\n",
    "esm/utils/msa/msa.py": textwrap.dedent('''
        class MSA:
            def __init__(self, sequences):
                self._sequences = list(sequences)
            @property
            def sequences(self): return self._sequences
            @property
            def depth(self): return len(self._sequences)
            @classmethod
            def from_a3m(cls, path, remove_insertions=False, max_sequences=None):
                seqs, cur = [], None
                for ln in open(path):
                    ln = ln.strip()
                    if ln.startswith(">"):
                        cur = ""; seqs.append(cur); continue
                    if seqs:
                        s = ln
                        if remove_insertions:
                            s = "".join(c for c in s if not c.islower())
                        seqs[-1] += s
                if max_sequences is not None:
                    seqs = seqs[:max_sequences]
                return cls(seqs)
            @classmethod
            def from_sequences(cls, sequences, remove_insertions=False):
                return cls(sequences)
    '''),
    "esm/utils/structure/__init__.py": "",
    "esm/utils/structure/input_builder.py": textwrap.dedent('''
        from dataclasses import dataclass
        from esm.utils.msa import MSA
        @dataclass
        class ProteinInput:
            id: str
            sequence: str
            modifications: list = None
            msa: object = None
        @dataclass
        class LigandInput:
            id: str
            ccd: list = None
            smiles: str = None
        @dataclass
        class StructurePredictionInput:
            sequences: list
            pocket: object = None
            distogram_conditioning: object = None
            covalent_bonds: object = None
        def deserialize_structure_prediction_input(data):
            seqs = []
            for ch in data["sequences"]:
                if ch["type"] == "ligand":
                    seqs.append(LigandInput(id=ch["id"], ccd=ch.get("ccd"), smiles=ch.get("smiles")))
                else:
                    msa = ch.get("msa")
                    if isinstance(msa, dict):
                        msa = MSA.from_sequences(msa["sequences"])
                    elif msa is not None:
                        raise ValueError(f"Unsupported MSA string value: {msa!r}")
                    seqs.append(ProteinInput(id=ch["id"], sequence=ch["sequence"], msa=msa))
            return StructurePredictionInput(sequences=seqs, pocket=data.get("pocket"), distogram_conditioning=data.get("distogram_conditioning"),
                                            covalent_bonds=data.get("covalent_bonds"))
    '''),
    "transformers/__init__.py": "__version__ = '0.0.0+stub'\n",
    "transformers/models/__init__.py": "",
    "transformers/models/esmfold2/__init__.py": "",
    "transformers/models/esmfold2/modeling_esmfold2_common.py": "FLASH_ATTN_AVAILABLE = False   # upstream's try-import switch (:25-41) on a box without flash-attn\nclass SWA3DRoPEAttention:\n    def forward(self, x, attention_params):\n        return x\nclass ResIdxAsymIdSymIdEntityIdEncoding:   # the relative-position encoder whose stock forward the XL census probes (big.XL_PROBES relpos)\n    def forward(self, *a, **k):\n        return None\n",
    "transformers/models/esmfold2/configuration_esmfold2.py": textwrap.dedent('''
        class _LM:
            def __init__(self): self.lm_dropout = 0.25; self.per_loop_lm_dropout = True
        class ESMFold2Config:
            """The shipped config's values that matter here (class defaults differ: the checkpoint carries lm_dropout 0.25 per loop)."""
            def __init__(self, repo): self.repo = repo; self.lm_encoder = _LM(); self.num_loops = 3
            @classmethod
            def from_pretrained(cls, repo, *a, **k): return cls(repo)
            def as_dict(self): return {"repo": self.repo, "lm_encoder.lm_dropout": self.lm_encoder.lm_dropout, "lm_encoder.per_loop_lm_dropout": self.lm_encoder.per_loop_lm_dropout}
    '''),
    "transformers/models/esmfold2/modeling_esmfold2.py": textwrap.dedent('''
        import json, os
        LOADS = []
        class _DM:
            _mk_enabled = False
        class _SH:
            def __init__(self): self.diffusion_module = _DM()
        class ESMFold2Model:
            def __init__(self, repo):
                self.repo = repo; self.structure_head = _SH(); self.msa_encoder = (object() if not repo.endswith("-Fast") else None)
                self.kernel_backend = None; self.chunk = 64
            @classmethod
            def from_pretrained(cls, repo, *a, **k):
                LOADS.append((repo, {kk: (vv if kk != "config" else vv.as_dict()) for kk, vv in k.items()}))
                if os.environ.get("STUB_RECORD"):                                          # the tests' record of the load (repo, the config it was given)
                    with open(os.environ["STUB_RECORD"], "a") as fh: fh.write(json.dumps(dict(event="load", repo=repo, config=(k["config"].as_dict() if k.get("config") is not None else None)), default=str) + "\\n")
                m = cls(repo); m.config = k.get("config"); return m
            def to(self, device): self.device = device; return self
            def eval(self): return self
            def set_kernel_backend(self, b): self.kernel_backend = b
            def set_chunk_size(self, c): self.chunk = c
            def _init_pair_state(self, *a, **k): return None                                # the stock pair-state init the XL census probes (big.XL_PROBES init)
    '''),
}


def write_stub_upstream(root: str, pins: dict = None, commits: dict = None) -> str:
    """Write the stub packages under root/site (with dist-info carrying the pinned versions from `pins` and a PEP 610 direct_url.json at
    the pinned commit — or `commits[name]` when given — so the package's version and commit gates pass) and return that directory
    (for sys.path / PYTHONPATH)."""
    site = os.path.join(root, "site")
    for rel, src in STUB_UPSTREAM.items():
        p = os.path.join(site, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(src)
    for name in ("esm", "transformers"):
        pin = ((pins or {}).get("upstream") or {}).get(name) or {}
        ver = pin.get("version") or "0.0.0"
        d = os.path.join(site, f"{name}-{ver}.dist-info")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "METADATA"), "w", encoding="utf-8") as fh:
            fh.write(f"Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\n")
        commit = (commits or {}).get(name, pin.get("commit"))
        if commit:
            with open(os.path.join(d, "direct_url.json"), "w", encoding="utf-8") as fh:
                json.dump({"url": pin.get("repo") or f"https://example.invalid/{name}", "vcs_info": {"vcs": "git", "commit_id": commit}}, fh)
    return site


def write_stub_kit(root: str, real_kit: str, unapplied: tuple = ()) -> str:
    """A kit directory whose driver/ef2_server.py carries the REAL server's MODES table (copied at test time) and a recording configure()
    that leaves the records of a COMPLETE application of the mode's lever set (the base calls in its desc, ef2_opt's install info in its
    desc, an ``ef2_w4`` / ``ef2_msa`` record module and the fourth field's group record modules (ef2_atom / ef2_feats / ef2_msa_v2 / ef2_pair_v2 / ef2_transition_cute / ef2_xte / ef2_hoist / ef2_xln / ef2_dit) in sys.modules, the MK instance flag) — except the levers named in ``unapplied``,
    recorded as not applied with the reason ``stub: cannot launch on this device`` (a kernel that could not run); the real
    driver/run_ef2_om.py (copied verbatim); and beside the kit a stub XL add-on (``EF2_XL_ADDON_v1/ef2_xl.py``) recording a complete
    engagement of the fast / big line's memory levers once per fold. Returns the kit root."""
    from esmfold2_opt import modes
    table = modes.server_table(os.path.join(real_kit, modes.SERVER_RELPATH))
    kit = os.path.join(root, "kit")
    drv = os.path.join(kit, "driver")
    os.makedirs(drv, exist_ok=True)
    real_server_src = open(os.path.join(real_kit, modes.SERVER_RELPATH), encoding="utf-8").read()
    real_fns = {node.name: ast.get_source_segment(real_server_src, node) for node in ast.parse(real_server_src).body
                if isinstance(node, ast.FunctionDef) and node.name in ("utc", "result_array", "write_outputs")}
    real_consts = {node.targets[0].id: ast.get_source_segment(real_server_src, node) for node in ast.parse(real_server_src).body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("NPZ_SUFFIX", "ROW_RESULT_FIELDS", "NPZ_RESULT_FIELDS")}
    with open(os.path.join(drv, "ef2_server.py"), "w", encoding="utf-8") as fh:
        fh.write("import os, sys, json, time, datetime\nimport numpy as np, torch\nimport run_ef2_om as KIT\n")
        fh.write("MODES = " + repr(table) + "\n")
        fh.write(f"UNAPPLIED = {tuple(unapplied)!r}\n")
        fh.write(textwrap.dedent('''
            import types
            CALLS = []
            OPT_KEYS = {"tg": "trunk_graphs", "sg": "sampler_graphs", "eg": "encoder_graphs", "ec": "esmc_cache", "fc": "feature_cache", "pb": "pair_bias_cache_blocks", "msa": "msa_fused_trimul_blocks"}
            W4_KEYS = {"t1": "transition_nolin", "t3": "tiles", "t5": "weight_cache", "t6": "transition_rowblock", "t10": "transition_fused"}
            MSA_KEYS = {"t11": "msa_transition", "t12": "opm", "t13": "pwa", "t14": "ln_bf16"}
            REASON = "stub: cannot launch on this device"
            OPT_KEYS.update({"ls": "loop_static", "rg": "recycle_graph"})
            GROUPS = ("trimul", "atom", "fz", "msa2", "pair", "hoist", "ln", "dit", "mk", "msa")
            TRIMUL_LEVERS = ("tx",)

            def parse_extra(extra):
                out = {}
                for tok in [t.strip() for t in str(extra or "").split(";") if t.strip()]:
                    name, _, flags = tok.partition(":")
                    assert name in GROUPS, name
                    out[name] = [f for f in flags.replace("+", ",").split(",") if f]
                return out

            def _record(name, **attrs):
                m = sys.modules.get(name) or types.ModuleType(name)
                for k, v in attrs.items():
                    setattr(m, k, v)
                sys.modules[name] = m
                return m

            def configure(model, mode, builder=None, off=(), knobs=None):
                """Records a complete application of MODES[mode] minus `off` (the records esmfold2_opt.stack.classify reads), minus UNAPPLIED."""
                assert mode in MODES, mode
                base, opt, w4 = MODES[mode][0], MODES[mode][1], MODES[mode][2]
                extra = MODES[mode][3] if len(MODES[mode]) > 3 else ""
                groups = parse_extra(extra)
                off = set(off or ())
                CALLS.append(dict(model=id(model), mode=mode, builder=id(builder), EF2_MK=os.environ.get("EF2_MK"), EF2_MSA=os.environ.get("EF2_MSA"), off=sorted(off), knobs=dict(knobs or {})))
                desc = []
                has_msa = getattr(model, "msa_encoder", None) is not None
                on = lambda g: [f for f in groups.get(g, []) if f not in off and f not in UNAPPLIED]
                if on("atom") or [f for f in groups.get("atom", []) if f not in off]:
                    want = [f for f in groups.get("atom", []) if f not in off]
                    _record("ef2_atom", groups_on=(lambda w=want: {g: (g in w and g not in UNAPPLIED) for g in ("ax", "af")}), STATS={}, _CFG={"gemm": (knobs or {}).get("af.gemm", "bf16")}, stats=(lambda: {}))
                if "fz" in groups and "fz" not in off:
                    _record("ef2_feats", active=(lambda: "fz" not in UNAPPLIED), STATS={}, stats=(lambda: {}))
                if [f for f in groups.get("msa2", []) if f not in off] and has_msa:
                    _record("ef2_msa_v2", levers_on=(lambda w=on("msa2"): list(w)), STATS={}, stats=(lambda: {}), mh_clear=(lambda: False))
                if [f for f in groups.get("pair", []) if f not in off and f != "t16"]:
                    want = [f for f in groups.get("pair", []) if f not in off and f != "t16"]
                    _record("ef2_pair_v2", levers_on=(lambda w=want: {k: (k in w and k not in UNAPPLIED and (k != "t15msa" or has_msa)) for k in ("t15", "t15msa", "t6s", "t6i")}),
                            _STATE={"t6s_selfcheck": ({"identical": True, "word": "identical", "rows": 4096} if "t6s" in want else None), "row": ({"t15_row": "sm90", "arch": "tested:9.0"} if "t15" in want else None)},
                            STATS={}, stats=(lambda: {}))
                if "t16" in groups.get("pair", []) and "t16" not in off:      # the CuTe transition's record module (levers_on / kernel_evidence) as ef2_transition_cute leaves it
                    KEV = {"cubin": "0123456789ab", "cubin_source": "prebuilt", "regs": 168, "spill_bytes": 0, "smem": 230512, "arch": "sm_90a", "serves": "transition+pair_transition"}
                    _record("ef2_transition_cute", levers_on=(lambda: {"t16": "t16" not in UNAPPLIED}), kernel_evidence=(lambda: dict(KEV)),   # an engaged run: the kernel's call
                            STATS=({} if "t16" in UNAPPLIED else {"transition_calls": 96, "pair_transition_calls": (16 if has_msa else 0)}), stats=(lambda: {}))   # counters are the evidence (stack.guards_after_run)
                if [f for f in groups.get("hoist", []) if f not in off]:
                    want = [f for f in groups.get("hoist", []) if f not in off]
                    _record("ef2_hoist", stats=(lambda w=want: {"levers": {k: (k not in UNAPPLIED and (k == "disto" or has_msa)) for k in w}}), STATS={}, sigmoid_check_word=(lambda: "ok:65280"))
                if [f for f in groups.get("ln", []) if f not in off]:          # ef2_xln's record: levers_on / refusal / evidence / stats as the module leaves them after install
                    want = [f for f in groups.get("ln", []) if f not in off]
                    _record("ef2_xln", levers_on=(lambda w=want: {"xln": ("xln" in w and "xln" not in UNAPPLIED)}), refusal=(lambda: (REASON.replace(" ", "_") if "xln" in UNAPPLIED else None)),
                            evidence=(lambda: {"sites": 57, "floor_rows": 4096, "core": "0.5.32.0", "kernels": 6, "cc": "9.0", "provider": "opt_core.kernels.ln.exactln"}),
                            STATS={}, stats=(lambda: {}))
                if "xte" in groups.get("pair", []) and "xte" not in off:     # ef2_xte's record: levers_on / refusal / evidence / stats as the module leaves them after install
                    _record("ef2_xte", levers_on=(lambda: {"xte": "xte" not in UNAPPLIED}), refusal=(lambda: (REASON.replace(" ", "_") if "xte" in UNAPPLIED else None)),
                            evidence=(lambda: {"word": "esm_fused_exact", "row": "esm_fused_exact", "core": "0.5.62.0", "cc": "9.0", "modules": 22, "bitwise_check": "identical", "check_rows": 300,
                                               "window": "1-3240000", "inplace": "t6i", "launch": "sm90_bm128_bh32_w8_s3", "provider": "opt_core.kernels.transition"}),
                            STATS={}, stats=(lambda: {"xte_calls": 0, "xte_inplace": 0, "xte_fallthrough": 0, "xte_outside_window": 0, "xte_pack": 0, "xte_modules": 22}), describe=(lambda: {}))
                if [f for f in groups.get("dit", []) if f not in off]:
                    want = [f for f in groups.get("dit", []) if f not in off]
                    _record("ef2_dit", levers_on=(lambda w=want: {k: (k in w and k not in UNAPPLIED) for k in ("ro", "kd", "dit")}), SCOPE={"num_diffusion_samples": 1, "batch": 1},
                            _CFG={"kabsch": ("device" if "kd" in want else "torch")}, knobs=(lambda: {"gemm": "bf16", "cond": "bf16", "attn": "flash", "attn_precision": "bf16"}),
                            STATS={}, stats=(lambda: {}))
                if base == "fused":
                    model.set_kernel_backend("fused"); model.set_chunk_size(None); desc.append("set_kernel_backend('fused') set_chunk_size(None); stub configure " + mode)
                env_mk = os.environ.get("EF2_MK")
                if ((env_mk == "1") if env_mk is not None else ("mk" in groups)) and "mk" not in UNAPPLIED and "mk" not in off:
                    model.structure_head.diffusion_module._mk_enabled = True
                msa_flags = os.environ.get("EF2_MSA") if os.environ.get("EF2_MSA") is not None else ",".join(f for f in groups.get("msa", []) if f not in off)
                if msa_flags and getattr(model, "msa_encoder", None) is not None:
                    m = sys.modules.get("ef2_msa") or types.ModuleType("ef2_msa")
                    m._STATE = dict({key: (lever not in UNAPPLIED) for lever, key in MSA_KEYS.items()}, enabled=True, models=[id(model)]); m.STATS = {}
                    sys.modules["ef2_msa"] = m; desc.append("ef2_msa.enable(model, %s) -> %s" % (msa_flags, {k: v for k, v in m._STATE.items() if k != "models"}))
                w4 = ",".join(f for f in (w4 or "").split(",") if f and f not in off)
                tflags = [f for f in groups.get("trimul", []) if f not in off]
                if w4 or tflags:
                    fl4 = set(w4.replace("+", ",").split(",")) if w4 else set()
                    st = {key: (flag in fl4 and flag not in UNAPPLIED) for flag, key in W4_KEYS.items()}
                    disabled = {flag.upper(): REASON for flag in sorted(fl4) if flag in UNAPPLIED}
                    m = sys.modules.get("ef2_w4") or types.ModuleType("ef2_w4")
                    m._STATE = dict(st, enabled=True, tx=("tx" in tflags)); m.describe = (lambda st=st, disabled=disabled: dict(st, device={"disabled": dict(disabled)}, tx=False)); m.stats = (lambda: {}); m.STATS = {}
                    sys.modules["ef2_w4"] = m
                    if fl4:
                        desc.append("ef2_w4.enable(%s) -> %s" % (sorted(fl4), m.describe()))
                    if "tx" in tflags:                                        # ef2_w4's tx record (describe()['tx'] + tx_state), as enable_tx(word) leaves it when a provider kernel row serves the canary
                        word = "big" if os.environ.get("ESMFOLD2_OPT", "") == "big" else {"opt7x": "exact"}.get(mode, "fast")
                        txs = {"on": ("tx" not in UNAPPLIED), "bound": True, "installed": True, "word": word, "why": (None if "tx" not in UNAPPLIED else REASON), "abi": "torch2.13.0+cu130-cpython-312-x86_64-linux-gnu-sm90",
                               "stack": "H100:2.13.0+cu130/3.7.1/nocueq", "cc": "9.0", "row": "native", "cell": "9.0|H100:stub|bf16|C256H256|N<=128|in.fwd", "selection": None, "rows": {}, "classes": [],
                               "calls": 0, "refused": {}, "class_stock_calls": 0, "stock_classes": 0, "weight_packs": 0}
                        m.describe = (lambda st=st, disabled=disabled, txs=txs: dict(st, device={"disabled": dict(disabled)}, tx=bool(txs["on"]), tx_state=dict(txs)))
                        m.tx_state = (lambda txs=txs: dict(txs))
                        desc.append("ef2_w4.enable_tx(%r) -> %s" % (word, txs))
                if opt:
                    fl = set(f for f in opt.split("+") if f not in off)
                    desc.append("ef2_opt.install(%s) -> %s" % (sorted(fl), {OPT_KEYS[f]: (f not in UNAPPLIED) for f in sorted(fl) if f in OPT_KEYS}))
                if off:
                    desc.append("off=" + ",".join(sorted(off)))
                return "; ".join(desc)
        '''))
        fh.write("\n\n" + "\n".join(real_consts[k] for k in ("NPZ_SUFFIX", "ROW_RESULT_FIELDS", "NPZ_RESULT_FIELDS")) + "\n\n\n" + real_fns["utc"] + "\n\n\n"
                 + real_fns["result_array"] + "\n\n\n" + real_fns["write_outputs"] + "\n")
    from esmfold2_opt import big as _big
    xl_dir = os.path.join(root, _big.XL_RELPATH)                             # the XL add-on beside the kit (big.xl_home): a recording apply() whose stats() report every memory
    os.makedirs(xl_dir, exist_ok=True)                                        # lever of the line engaged once per fold (the counters esmfold2_opt.big.xl_gate reads), x4 off by size
    with open(os.path.join(xl_dir, "ef2_xl.py"), "w", encoding="utf-8") as fh:
        fh.write(textwrap.dedent('''
            """Stub XL add-on: apply() records its knobs and counts the builder's folds; stats() reports one engagement of every lever per fold."""
            __version__ = "stub"
            STATE = {"applied": 0, "knobs": {}, "folds": 0}
            def apply(model, **knobs):
                import esm.models.esmfold2.processor as P
                STATE["applied"] += 1; STATE["knobs"] = dict(knobs)
                B = P.ESMFold2InputBuilder
                if not getattr(B.fold, "_xl_stub_counter", False):
                    inner = B.fold
                    def fold(self, *a, **k):
                        STATE["folds"] += 1
                        return inner(self, *a, **k)
                    fold._xl_stub_counter = True; fold.__wrapped__ = inner; B.fold = fold
                return {"patch_shas": {}, "knobs": dict(knobs), "stub": True}
            def stats():
                from esmfold2_opt.registry import XL_LEVERS
                n = STATE["folds"]; k = STATE["knobs"]
                st = {lv.probe[1]: n for lv in XL_LEVERS.values()}
                st.update({"loop_calls": n, "owned_blocks": (n if k.get("own") else 0), "init_xl_calls": n, "disto_cpu_events": n,
                           "free_events": (3 * n if k.get("free") else 0), "cond_calls": n, "cond_xl_calls": n, "s2p_calls": n, "s2p_xl_calls": n,
                           "relpos_calls": n, "relpos_xl_calls": n, "esmc_offload_events": 0})
                return st
        '''))
    import shutil
    shutil.copy(os.path.join(real_kit, modes.DRIVER_RELPATH), os.path.join(drv, "run_ef2_om.py"))
    for rel in ("tests/w4_public_slice.json", "README.md"):
        src = os.path.join(real_kit, rel)
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(os.path.join(kit, rel)), exist_ok=True)
            shutil.copy(src, os.path.join(kit, rel))
    return kit


def write_stub_tree(root: str, pins: dict) -> str:
    """esmfold2/ with stock/PINS.json (the given pins) and the tree's own stock/check_pins.py (copied: the one place the commit-level pin
    check lives) — MODEL_OPT for the package."""
    import shutil
    from esmfold2_opt import stack
    tree = os.path.join(root, "esmfold2")
    os.makedirs(os.path.join(tree, "stock"), exist_ok=True)
    with open(os.path.join(tree, "stock", "PINS.json"), "w", encoding="utf-8") as fh:
        json.dump(pins, fh, indent=1)
    shutil.copy(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", stack.CHECK_PINS_RELPATH),
                os.path.join(tree, "stock", "check_pins.py"))
    write_fake_weights_root(os.path.join(root, "hf"), pins)
    return tree


def write_fake_weights_root(hf: str, pins: dict) -> str:
    """A weights root (HF_HOME) in the HuggingFace cache layout holding every pinned file of stock/PINS.json ``weights`` as a ONE-BYTE file
    (present, so the data-path gate does not refuse; an UNKNOWN checkpoint by its sha256, so the gate WARNS by name and proceeds — the stubs
    never read the weights, and the gate digests these bytes in milliseconds)."""
    from esmfold2_opt import stack
    for _repo, rel, _size in stack.pinned_weight_files(pins, None):
        p = os.path.join(hf, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(b"\0")
    return hf


def env_for_stub(site: str, kit: str, tree: str, **extra) -> dict:
    """A subprocess environment that sees the stub upstream, the stub kit and the stub tree (no kit switches); HF_HOME = the stub tree's
    fake weights root (write_stub_tree) unless the caller names one."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EF2_", "ESMFOLD2_"))}
    env["PYTHONPATH"] = site + os.pathsep + env.get("PYTHONPATH", "")
    env["ESMFOLD2_OPT_KIT"] = kit
    env["MODEL_OPT"] = tree
    env["HF_HOME"] = os.path.join(os.path.dirname(tree), "hf")
    env["ESMFOLD2_OPT_FORCE"] = "1"                                   # no GPU on the test box
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env


def _is_upstream_module(name: str) -> bool:
    """The upstream names exactly (`esm`, never this package, esmfold2_opt…): torch, esm, transformers, the kit server and its runner."""
    return name in ("torch", "esm", "transformers") or name.startswith(("torch.", "esm.", "transformers.", "ef2_server", "run_ef2_om"))


_REAL_UPSTREAM: dict = {}                                                     # real upstream modules set aside by install_stub_modules, put back by forget_stub_modules


def install_stub_modules(site: str) -> None:
    """Put the stub site on sys.path (front) and forget any upstream modules loaded so far. A REAL upstream module (one not loaded from the
    stub site) is set aside, not dropped: forget_stub_modules() puts the same module objects back, because a real torch initialises its
    C extension once per interpreter — re-importing it after a `del sys.modules["torch"]` raises (or, on newer torch, kills the process)."""
    if site not in sys.path:
        sys.path.insert(0, site)
    for m in list(sys.modules):
        if _is_upstream_module(m):
            mod = sys.modules.pop(m)
            origin = getattr(mod, "__file__", None) or ""
            if not origin.startswith(site) and m not in _REAL_UPSTREAM:
                _REAL_UPSTREAM[m] = mod


def forget_stub_modules(site: str) -> None:
    """End an in-process stub session: the stub site leaves sys.path, the stub upstream modules leave sys.modules, and every real upstream
    module set aside at install time returns as the same object (a later `import torch` finds it; nothing re-initialises)."""
    while site in sys.path:
        sys.path.remove(site)
    for m in list(sys.modules):
        if _is_upstream_module(m):
            del sys.modules[m]
    sys.modules.update(_REAL_UPSTREAM)


def read_stub_record(path: str) -> list:
    """The STUB_RECORD lines (what reached the stub upstream: fold kwargs + the det state at the call, from_pretrained repo + config)."""
    import json as _json
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [_json.loads(l) for l in fh if l.strip()]
