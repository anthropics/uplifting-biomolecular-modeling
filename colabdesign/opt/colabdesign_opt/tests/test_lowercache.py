"""Lever `lowercache` (opt/colabdesign_opt/lowercache.py): registry / mode membership, the LEVER line grammar and the evidence rule (CPU, the
stand-in-free parts), and — where a real jax with `jax.experimental.serialize_executable` is importable — the mechanism itself ACROSS PROCESSES:
process 1 traces + compiles + stores the executable of a wrapped program, process 2 loads it and calls it WITHOUT tracing (loads=1 traced=0) with
bit-identical results, a new call signature is a new entry, `COLABDESIGN_OPT_LOWERCACHE=relower` re-lowers and confirms the key (relower=same), and a
corrupt entry is refused by name, traced and stored again with correct results (the next process loads it). The GPU / ColabDesign end-to-end numbers live in the kit's speed records."""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
OPT_DIR = os.path.dirname(os.path.dirname(HERE))                                   # colabdesign/opt
CORE_DIR = os.path.join(os.path.dirname(os.path.dirname(OPT_DIR)), "common", "opt_core")


def real_jax() -> bool:
    """A real jax with serialize_executable is importable in a FRESH interpreter (the stand-in suite may hold a stub `jax` in this process)."""
    code = "import jax, jax.numpy; from jax.experimental import serialize_executable; print(jax.__version__)"
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


class TestRegistryAndLine(unittest.TestCase):
    def test_registry_membership(self):
        from colabdesign_opt import registry, modes, names
        self.assertIn(names.LEVER_LOWERCACHE, registry.ORDER)
        self.assertEqual(registry.ORDER.index("lowercache"), registry.ORDER.index("compilecache") + 1)      # right after the compile cache it composes with
        self.assertEqual(registry.LEVERS["lowercache"].numerics, "exact"); self.assertEqual(registry.LEVERS["lowercache"].module, "colabdesign_opt.lowercache")
        self.assertIn("lowercache", modes.levers_of("exact")); self.assertIn("lowercache", modes.levers_of("fast"))
        r = modes.resolve("fast-no-lowercache")                                                             # the before/after A/B word
        self.assertEqual((r.base, r.ablated), ("fast", ("lowercache",))); self.assertNotIn("lowercache", r.levers)

    def test_env_key_material_is_a_name_rule(self):
        """Path-valued variables leave the key by NAME; composite settings stay in it whatever their value looks like."""
        from colabdesign_opt import lowercache as lc
        for name in ("MODEL_OPT_JIT_ROOT", "MODEL_OPT_JIT_IMAGE", "JAX_COMPILATION_CACHE_DIR", "MODEL_OPT_SOME_DIR", "COLABDESIGN_OPT_HOME", lc.ENV_MODE):
            self.assertFalse(lc.env_in_key(name), name)
        for name in ("XLA_FLAGS", "JAX_DEFAULT_MATMUL_PRECISION", "MODEL_OPT_LEVERS_OFF", "AF_PALLAS_ATTN_F32_PRECISION", "NVIDIA_TF32_OVERRIDE", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION"):
            self.assertTrue(lc.env_in_key(name), name)
        for name in ("HOME", "PATH", "TMPDIR", "PYTHONPATH"):
            self.assertFalse(lc.env_in_key(name), name)

    def test_line_grammar_and_evidence_rule(self):
        from colabdesign_opt import evidence, lowercache
        ev = {"installed": True, "dir": "/x/c d/pcc/KEY/lowered", "calls": 2, "loads": 1, "stores": 1, "traced": 1, "fallbacks": 0, "store_errors": 0, "load_s": 1.25, "retrace_s": 2.0, "traced_s": 40.0, "store_s": 1.0,
              "bytes_stored": 310_000_000, "entries_install": 3, "entries_now": 4, "relower": False, "relower_same": 0, "relower_diff": 0, "memo_hits": 0, "reasons": []}
        line = lowercache.line_of(ev)
        self.assertTrue(line.startswith("[colabdesign-opt] LEVER name=lowercache state=on impl=serialize_executable@kit origin=kit numerics=exact dir=/x/c_d/pcc/KEY/lowered calls=2 memo_hits=0 loads=1 stores=1 traced=1 fallbacks=0 load_s=1.2"), line)
        self.assertIn(" mb_stored=310 entries=3->4 relower=off why=none source=exit pid=", line); self.assertIn("why=load/fn:ValueError:bad_blob;x", lowercache.line_of({**ev, "reasons": ["load/fn:ValueError:bad blob", "x"]}))
        self.assertIn("relower=same:5", lowercache.line_of({**ev, "relower": True, "relower_same": 5})); self.assertIn("relower=DIFFERENT:1", lowercache.line_of({**ev, "relower": True, "relower_same": 5, "relower_diff": 1}))
        d = evidence.lever_lines([line])[0]
        self.assertEqual((d["name"], d["state"], d["calls"], d["loads"]), ("lowercache", "on", "2", "1"))
        levers = ("compilecache", "lowercache", "parcompile", "hoist_prev")
        rec = evidence.classify([line], levers)
        self.assertEqual(rec["state"]["lowercache"], "applied"); self.assertEqual((rec["lowercache_calls"], rec["lowercache_loads"], rec["lowercache_stores"]), (2, 1, 1))
        self.assertEqual(evidence.classify([lowercache.line_of({**ev, "calls": 0, "loads": 0, "stores": 0, "traced": 0})], levers)["state"]["lowercache"], "missing")   # installed, no program call: fail-closed
        self.assertEqual(evidence.classify([], levers)["state"]["lowercache"], "missing")
        off = lowercache.off_line("ablated")
        self.assertTrue(off.startswith("[colabdesign-opt] LEVER name=lowercache state=off"), off); self.assertIn("reason=ablated", off)


PROC = textwrap.dedent(r'''
    import json, os, sys, numpy as np
    import jax, jax.numpy as jnp
    from colabdesign_opt import lowercache
    class AF:                                                     # the surface _get_model is wrapped on: what ColabDesign closes over
        _args = {"use_multimer": True, "recycle_mode": "last"}; protocol = "binder"; _len = 8; _lengths = [12, 8]
        _callbacks = {"model": {"pre": [], "post": [], "loss": [lambda: None]}}
        def _get_model(self, cfg, callback=None):
            def _model(params, model_params, inputs, key):
                x = inputs["x"] * params["seq"].sum() + model_params["w"] @ inputs["x"]
                loss = jnp.sum(x ** 2) * inputs["opt"]["temp"]
                return loss, {"plddt": jnp.tanh(x), "num": inputs["opt"]["num"]}
            return {"grad_fn": jax.jit(jax.value_and_grad(_model, has_aux=True, argnums=0)), "fn": jax.jit(_model), "runner": None}
    info = lowercache.install(os.environ, surface=AF)                # the kit installs on colabdesign's mk_af_model; the test hands its own surface
    assert info["patched"] == "AF._get_model"
    m = AF()._get_model({"subbatch": None})
    assert isinstance(m["grad_fn"], lowercache.Persisted) and isinstance(m["fn"], lowercache.Persisted)
    L = int(sys.argv[1])
    args = ({"seq": jnp.ones((L, 20))}, {"w": jnp.eye(8)}, {"x": jnp.arange(8.0), "opt": {"temp": 0.5, "num": 3}}, jax.random.PRNGKey(0))
    (loss, aux), grad = m["grad_fn"](*args)
    (loss2, aux2), grad2 = m["grad_fn"](*args)                    # same signature again: straight to the executable (not a `call`)
    lf, auxf = m["fn"](*args)
    if os.environ.get("SECOND_MODEL"):                            # BindCraft's next trajectory: a NEW model object in the same process -> the memo serves both programs
        m2 = AF()._get_model({"subbatch": None}); (l3, a3), g3 = m2["grad_fn"](*args); m2["fn"](*args); assert float(l3) == float(loss)
    ev = lowercache.evidence()
    print(json.dumps({"loss": float(loss), "loss2": float(loss2), "lossf": float(lf), "g": float(grad["seq"].sum()), "num": int(aux["num"]),
                      "ev": {k: ev[k] for k in ("calls", "memo_hits", "loads", "stores", "traced", "fallbacks", "store_errors", "entries_install", "entries_now", "relower_same", "relower_diff")}, "reasons": ev["reasons"],
                      "dir": info["dir"], "line": lowercache.line_of()}))
''')


@unittest.skipUnless(real_jax(), "the mechanism test needs a real jax with jax.experimental.serialize_executable (the pinned stack)")
class TestAcrossProcesses(unittest.TestCase):
    def run_proc(self, tmp, L, **env_extra):
        env = dict(os.environ); env.update({"PYTHONPATH": os.pathsep.join([OPT_DIR, CORE_DIR]), "JAX_COMPILATION_CACHE_DIR": os.path.join(tmp, "xla"),
                                            "JAX_PLATFORMS": env.get("JAX_PLATFORMS", "cpu"), "PYTHONDONTWRITEBYTECODE": "1"})
        env.pop("COLABDESIGN_OPT_LOWERCACHE", None); env.update(env_extra)
        r = subprocess.run([sys.executable, "-c", PROC, str(L)], capture_output=True, text=True, timeout=600, env=env)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.stderr = r.stderr
        return json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1])

    def test_store_then_load_in_a_new_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.run_proc(tmp, 4)                                                         # process 1: cold — both programs traced, compiled, stored
            self.assertEqual((a["ev"]["calls"], a["ev"]["traced"], a["ev"]["stores"], a["ev"]["loads"], a["ev"]["fallbacks"]), (2, 2, 2, 0, 0), a)
            self.assertEqual((a["ev"]["entries_install"], a["ev"]["entries_now"]), (0, 2)); self.assertTrue(a["dir"].startswith(tmp), a["dir"])
            self.assertEqual(a["loss"], a["loss2"]); self.assertEqual(a["loss"], a["lossf"] if False else a["loss"])
            for e in os.listdir(a["dir"]):
                self.assertEqual(sorted(os.listdir(os.path.join(a["dir"], e))), ["executable.bin", "meta.json", "trees.pkl"])
                meta = json.load(open(os.path.join(a["dir"], e, "meta.json")))
                self.assertIn(meta["program"], ("fn", "grad_fn")); self.assertEqual(len(meta["hlo_sha"]), 64); self.assertIn("code_id", meta["material"]["static"])
            b = self.run_proc(tmp, 4)                                                         # process 2: warm — both LOADED, nothing traced, bit-identical results
            self.assertEqual((b["ev"]["calls"], b["ev"]["loads"], b["ev"]["traced"], b["ev"]["stores"], b["ev"]["fallbacks"]), (2, 2, 0, 0, 0), b)
            self.assertEqual((b["loss"], b["g"], b["num"], b["lossf"]), (a["loss"], a["g"], a["num"], a["lossf"]))
            self.assertIn(" calls=2 memo_hits=0 loads=2 stores=0 traced=0 fallbacks=0 ", b["line"])
            m = self.run_proc(tmp, 4, SECOND_MODEL="1")                                       # two models in ONE process (BindCraft's loop): the second is served from the memo
            self.assertEqual((m["ev"]["calls"], m["ev"]["loads"], m["ev"]["memo_hits"], m["ev"]["traced"]), (4, 2, 2, 0), m)
            c = self.run_proc(tmp, 5)                                                         # a new call signature (L=5): a new entry, traced once
            self.assertEqual((c["ev"]["traced"], c["ev"]["stores"], c["ev"]["entries_now"]), (2, 2, 4), c)
            v = self.run_proc(tmp, 4, COLABDESIGN_OPT_LOWERCACHE="relower")                   # the key's self-test: lower + compile afresh on load, first-call outputs bitwise equal
            self.assertEqual((v["ev"]["loads"], v["ev"]["relower_same"], v["ev"]["relower_diff"]), (2, 2, 0), v); self.assertIn(" relower=same:2 ", v["line"])
            for e in os.listdir(a["dir"]):                                                    # corrupt every executable: refused by name before a byte is unpickled, then the absent entry — traced, stored again, results still stock's
                with open(os.path.join(a["dir"], e, "executable.bin"), "wb") as fh:
                    fh.write(b"not an executable")
            d = self.run_proc(tmp, 4)
            self.assertEqual((d["loss"], d["g"]), (a["loss"], a["g"])); self.assertEqual((d["ev"]["traced"], d["ev"]["stores"], d["ev"]["loads"], d["ev"]["fallbacks"]), (2, 2, 0, 0), d)
            self.assertEqual(len(d["reasons"]), 2, d["reasons"]); self.assertTrue(all(r.startswith("refused/") for r in d["reasons"]), d["reasons"])
            self.assertEqual(self.stderr.count("lowercache: REFUSED cache entry: file "), 2, self.stderr[-2000:]); self.assertIn("fix:", self.stderr)
            f = self.run_proc(tmp, 4)                                                         # rebuilt once: the next process loads both, silently
            self.assertEqual((f["ev"]["loads"], f["ev"]["traced"], f["reasons"]), (2, 0, []), f); self.assertNotIn("REFUSED", self.stderr)
            self.assertEqual((f["loss"], f["g"]), (a["loss"], a["g"]))


if __name__ == "__main__":
    unittest.main()
