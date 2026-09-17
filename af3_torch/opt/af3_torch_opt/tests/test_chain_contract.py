"""The three model processes agree with each other and with the kit, statically (the scripts run only under the image's venvs):
every flag cli.py passes to featurise.py / forward.py / postprocess.py is an argument of that script; the result.npz keys forward.py
writes (atom_positions, conf_<key> for every key of the kit's run_confidence, the distogram head's own keys, seed) are the keys
postprocess.py requires; the fork-side scripts import nothing of the fork at module level (importable on any interpreter)."""
import ast
import os
import sys
import subprocess
import re

from af3_torch_opt import stack

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = open(os.path.join(PKG, "cli.py"), encoding="utf-8").read()


def _flags(script):
    return set(re.findall(r'add_argument\("(--[a-z0-9_-]+)"', open(os.path.join(PKG, script), encoding="utf-8").read()))


def _passed(start, end):
    seg = CLI[CLI.index(start):CLI.index(end)]
    return set(re.findall(r'"(--[a-z0-9_-]+)"', seg))


def test_cli_flags_are_the_scripts():
    feat = _passed("# 1. featurise", "# 2. forward")
    assert {"--item", "--report", "--buckets", "--repo_dir"} <= feat <= _flags("featurise.py"), feat
    from af3_torch_opt import cli as _cli                                       # the stock CLI's pipeline flags pred passes through are the featuriser's flags, name for name
    from opt_core.modes import literal_assignment                              # featurise.py is a script of the JAX interpreter: read its flag tuple from the file
    assert tuple(literal_assignment(os.path.join(PKG, "featurise.py"), "PIPELINE_FLAGS")) == _cli.STOCK_PIPELINE_STRINGS + _cli.STOCK_PIPELINE_INTS
    assert {"--run_data_pipeline", "--run_inference", "--repo_dir", "--db_dir"} <= _flags("featurise.py")
    assert {f"--{n}" for n in _cli.STOCK_PIPELINE_STRINGS + _cli.STOCK_PIPELINE_INTS} | {"--run_data_pipeline", "--run_inference", "--model_dir", "--norun_data_pipeline", "--norun_inference"} <= set(re.findall(r'(--[a-z0-9_]+)', " ".join(a for act in _cli.build_parser()._subparsers._group_actions[0].choices["pred"]._actions for a in act.option_strings)))
    fwd = _passed("# 2. forward", "# 3. postprocess")
    assert {"--kit", "--dtk-home", "--params", "--levers", "--dtk", "--report", "--item", "--opt-core", "--route"} <= fwd <= _flags("forward.py"), fwd
    post = _passed("# 3. postprocess", "    n_ok = sum(")
    assert {"--item", "--report", "--repo_dir"} <= post <= _flags("postprocess.py") and "--params_dir" not in _flags("postprocess.py"), post


def _constants(path, *names):
    """The named top-level literal assignments of a script, by AST (a script that imports torch, jax or numpy at module level is read, never
    executed, on the tests' interpreter); a name without a literal assignment fails loud."""
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    found = {t.id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Name) and t.id in names}
    missing = [n for n in names if n not in found]
    assert not missing, f"{path}: no literal top-level assignment of {missing}"
    return found


def test_result_keys_contract():
    fwd = open(os.path.join(PKG, "forward.py"), encoding="utf-8").read()
    post = _constants(os.path.join(PKG, "postprocess.py"), "REQUIRED_CONF", "DISTOGRAM_KEYS")   # read, not executed: the script imports numpy at module level
    assert 'out[f"conf_{k}"] = arr(v)' in fwd and '"atom_positions": arr(result["atom_positions"])' in fwd and 'arrays["seed"] = np.asarray(seed)' in fwd
    head = open(os.path.join(stack.kit_home(), "af3_torch", "xfold", "nn", "head.py"), encoding="utf-8").read()   # the kit's confidence + distogram heads
    for key in post["REQUIRED_CONF"]:                       # every key the writers require is a key the kit's confidence head returns
        assert f'"{key}"' in head or f"'{key}'" in head, key
    assert "contact_probs" in post["DISTOGRAM_KEYS"] and "contact_probs" in head


def test_fork_side_scripts_import_nothing_of_the_fork_at_module_level():
    for script in ("featurise.py", "postprocess.py"):
        tree = ast.parse(open(os.path.join(PKG, script), encoding="utf-8").read())
        top = {n.names[0].name.split(".")[0] for n in tree.body if isinstance(n, ast.Import)} | {(n.module or "").split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)}
        assert not top & {"alphafold3", "jax", "haiku", "torch", "absl"}, (script, top)


def test_forward_seeds_per_item_before_the_forward_and_guards_the_graph_key():
    """The item's randomness is a function of its seed alone (the kit's MSA shuffle draws from the global RNG and the kit seeds only inside
    its sampler): forward.py calls torch.manual_seed(seed) + torch.cuda.manual_seed_all(seed) and the graph-key guard BEFORE the forward,
    inside the per-item loop, and records rng / graph_reset on the item."""
    src = open(os.path.join(PKG, "forward.py"), encoding="utf-8").read()
    src_impl = open(os.path.join(PKG, "forward_impl.py"), encoding="utf-8").read()   # _reset_item_state / _forward_samples live here, imported by forward.py
    i_seed, i_cuda, i_guard, i_fwd = (src.index(s_) for s_ in ("torch.manual_seed(seed)", "torch.cuda.manual_seed_all(seed)", "_reset_item_state(model, torch)", "_forward_samples(A, model, batch, seed, torch, "))
    assert i_seed < i_cuda < i_guard < i_fwd
    loop = src.index("for spec in a.item"); assert loop < i_seed, "the seeding must be inside the per-item loop"
    assert 'it["rng"] = "seeded_per_item"' in src and 'it["graph_reset"] = _reset_item_state' in src and 'rep["graph_resets"]' in src and "pool_reset" not in src
    assert "dh.clear_static()" in src_impl and 'it["graph_capture"] = _has_step_graph(model)' in src   # every item starts from a cleared head; captures named
    assert 'K.clear_caches()' in src_impl and "GRAPHS" not in src_impl                                          # and from an empty kernel-kit mask-term cache (the kit keeps no graph pool)


def test_forward_refuses_a_params_directory_by_name():
    """forward.py takes the pinned checkpoint FILE only: a directory is refused at parse (rc 2, the reason in the report) — the kit's loader
    would silently take the first *.bin.zst of a directory (xfold/params.py:739-743)."""
    src = open(os.path.join(PKG, "forward.py"), encoding="utf-8").read()
    i = src.index("if not os.path.isfile(a.params):"); assert "return 2" in src[i:i + 400] and "is not a file" in src[i:i + 400]
    assert i < src.index("A.build_model("), "the refusal precedes the model build"


def test_reset_item_state_runs_and_counts():
    """_reset_item_state executes against a model shaped like the kit's (diffusion_head.clear_static, _af3t_kernels.clear_caches):
    it drops the whole-step graph, clears the kernel kit's mask-term cache, and returns whether a graph was dropped."""
    import importlib.util, types
    import af3_torch_opt
    spec = importlib.util.spec_from_file_location("af3_torch_opt_forward", os.path.join(os.path.dirname(af3_torch_opt.__file__), "forward.py"))
    fwd = importlib.util.module_from_spec(spec); spec.loader.exec_module(fwd)
    calls = []
    class DH:
        _static = {"graph": object()}
        def clear_static(self):
            calls.append("static"); self._static = None
    K = types.SimpleNamespace(clear_caches=lambda: calls.append("caches"))
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: calls.append("sync"), empty_cache=lambda: calls.append("empty")))
    model = types.SimpleNamespace(diffusion_head=DH(), _af3t_kernels=K)
    assert fwd._reset_item_state(model, torch) == 1
    assert model.diffusion_head._static is None and calls == ["static", "caches"]            # no allocator call: freeing is the caller's (diff_free) or the next capture's
    assert fwd._reset_item_state(model, torch) == 0 and calls == ["static", "caches", "static", "caches"]


def test_postprocess_resolves_the_hasher_through_opt_core(tmp_path):
    """postprocess.py runs standalone (the JAX venv: no package, no core installed): its file hasher is the core's, reached through the
    --opt_core directory the wrapper passes — proven on a bare interpreter (-I: no site, empty path)."""
    from opt_core import gates as cg
    core_dir = os.path.dirname(cg.imported_core()["package_dir"])
    (tmp_path / "numpy.py").write_text("", encoding="utf-8")            # the script imports numpy at module level (the JAX venv's); a stand-in here — the tests' interpreter has none
    code = ("import sys, importlib.util, hashlib; sys.path.insert(0, %r); sys.argv = ['postprocess.py']; spec = importlib.util.spec_from_file_location('pp', %r); m = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(m); h = m.file_hasher(%r); p = %r; open(p, 'wb').write(b'abc'); print(h(p) == hashlib.sha256(b'abc').hexdigest())"
            % (str(tmp_path), os.path.join(PKG, "postprocess.py"), core_dir, str(tmp_path / "f")))
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "True", r


def test_forward_composes_the_confidence_head_per_sample():
    """AF3 Model.__call__'s composition from the kit's public steps: the trunk once, the sampler once (S samples), the confidence head once
    PER SAMPLE on that sample's positions (stacked to [S, ...]), the distogram once — never the kit's own forward (confidence on sample 0
    only, af3_torch_api.py:219-226). At S = 1 the same calls in the same order on the same tensors (the 1-sample line's bytes unchanged).
    A minimal tensor stand-in (no numpy / torch on the tests' interpreter)."""
    import importlib.util, types
    import af3_torch_opt

    class T:                                                   # a tensor stand-in: nested lists with a shape, indexable on the first axis
        def __init__(self, data): self.data = data
        @property
        def shape(self):
            s, d = [], self.data
            while isinstance(d, list): s.append(len(d)); d = d[0]
            return tuple(s)
        def __getitem__(self, i): return T(self.data[i])
        def __eq__(self, o): return isinstance(o, T) and o.data == self.data
    spec = importlib.util.spec_from_file_location("af3_torch_opt_forward", os.path.join(os.path.dirname(af3_torch_opt.__file__), "forward.py"))
    fwd = importlib.util.module_from_spec(spec); spec.loader.exec_module(fwd)
    src_ = open(spec.origin, encoding="utf-8").read()
    src_impl_ = open(os.path.join(os.path.dirname(spec.origin), "forward_impl.py"), encoding="utf-8").read()   # _forward_samples lives here, importable so a timing hook can wrap it
    assert "A.forward(" not in src_impl_ and "def _forward_samples(" in src_impl_
    assert src_.index("from forward_impl import") < src_.index('if __name__ == "__main__":')   # imported before the script form runs main()
    for S in (1, 5):
        calls = []; xyz = T([[[[float(s * 10 + n)] * 3] for n in range(2)] for s in range(S)])          # [S, N=2, 1, 3]
        A = types.SimpleNamespace(run_trunk=lambda m, b: (calls.append("trunk"), {"pair": 1})[1],
                                  run_diffusion=lambda m, b, e, seed=None: (calls.append(("diffusion", seed)), xyz)[1],
                                  run_confidence=lambda m, b, e, pos: (calls.append(("confidence", pos.data[0][0][0])), {"predicted_lddt": T([row[0][0] for row in pos.data]), "n": 7})[1],
                                  run_distogram=lambda m, b, e: (calls.append("distogram"), {"contact_probs": T([[0.0]])})[1])
        torch = types.SimpleNamespace(is_tensor=lambda v: isinstance(v, T), stack=lambda vs: T([v.data for v in vs]), cuda=types.SimpleNamespace(synchronize=lambda: calls.append("sync")))
        item = {}
        r = fwd._forward_samples(A, "model", "batch", 101, torch, item=item)
        assert [c for c in calls if c == "sync"] == ["sync"] * 5 and calls[0] == "sync" and calls[-2:] == ["sync", "distogram"]   # a phase boundary (synchronize + clock) before the trunk, between the phases, after the confidence head; the distogram is in no phase
        assert list(item["phase_s"]) == ["lm", "trunk", "sampler", "conf"] and item["phase_s"]["lm"] is None and all(isinstance(item["phase_s"][k], float) for k in ("trunk", "sampler", "conf"))
        calls = [c for c in calls if c != "sync"]
        assert calls == ["trunk", ("diffusion", 101)] + [("confidence", float(s * 10)) for s in range(S)] + ["distogram"], calls
        assert r["atom_positions"] is xyz and r["embeddings"] == {"pair": 1} and r["distogram"]["contact_probs"] == T([[0.0]])
        assert r["confidence"]["predicted_lddt"].shape == (S, 2) and r["confidence"]["predicted_lddt"] == T([[float(s * 10), float(s * 10 + 1)] for s in range(S)])
        assert r["confidence"]["n"] == [7] * S


def test_pred_passes_upstreams_sample_count_unless_named(box, tmp_path):
    """The model process is built at the stock CLI's sample count (--num_diffusion_samples, default 5 = modes.UPSTREAM_SAMPLES) unless the
    flag names one: the single-sample form is an opt-in, never a silent default of the kit's own build_model (num_samples=1)."""
    from .test_cli_manifest import _inputs
    from af3_torch_opt import cli, modes
    from .conftest import stub_calls
    assert modes.UPSTREAM_SAMPLES == 5
    assert cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(tmp_path / "o5")]) == 0
    fwd = [c for c in stub_calls(box) if os.path.basename(c[1]) == "forward.py"][-1]
    assert fwd[fwd.index("--num_samples") + 1] == "5"
    assert cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(tmp_path / "o1"), "--num_diffusion_samples", "1"]) == 0
    fwd = [c for c in stub_calls(box) if os.path.basename(c[1]) == "forward.py"][-1]
    assert fwd[fwd.index("--num_samples") + 1] == "1"
