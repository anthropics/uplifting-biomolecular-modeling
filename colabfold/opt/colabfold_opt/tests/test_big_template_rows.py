"""`big --n_gpu P`: the template pair features are born on ROWS. What `big` asks of the shared recipe (``big.RECIPE_WORDS``: the trunk,
model and heads site groups) installs the template embedding's row bodies on the pinned alphafold-colabfold tree, so a templated query is
consumed with each device's own rows of the template features — computed from the per-residue template fields (aatype, atom positions, atom
masks: O(T·N)) — never as a dense ``[T, N, N, F]`` per device and never substituted. Asserted here on jax's CPU backend with forced host devices,
on the multimer tree colabfold runs for complexes, with synthetic non-zero template atoms and one all-dummy slot:
  (1) the distogram rows the row body forms == stock ``dgram_from_positions`` rows, BITWISE, at the multimer template distogram settings;
  (2) the install `big` requests yields ``template=rows`` (the word its LEVER line prints);
  (3) stock ``modules_multimer.TemplateEmbedding`` on the whole pair vs the installed rows region: bit-exact on a 1-device mesh; at P = 2 the
      output leaves row-sharded, the region ran (``template_regions_built``), and the values equal stock within 1e-5 relative (the nested template
      pair stack's cross-device sums re-associate in float32; the features themselves are the bitwise case (1));
  (4) the per-query census line ``TEMPLATES job=… real=<r>/<T> form=…`` (``big.template_census`` / ``templates_line``): real slots counted from
      the template atom mask of the dict colabfold hands to ``predict_structure`` (mock and padded slots are mask 0), both key spellings; numpy only.
float32 with ``jax.default_matmul_precision("float32")`` (on a GPU the default TF32 rounds the row-shaped and the whole-pair matmuls differently — a
tiling fact, not a decomposition fact; the tree's bfloat16 casts are transcribed alike in both bodies). Skipped by name where jax / dm-haiku /
alphafold are not importable in this interpreter or it shows fewer devices than P."""
import os
import sys
import unittest

_XLA_BEFORE = os.environ.get("XLA_FLAGS")
if "jax" not in sys.modules and "xla_force_host_platform_device_count" not in (_XLA_BEFORE or ""):
    os.environ["XLA_FLAGS"] = ((_XLA_BEFORE or "") + " --xla_force_host_platform_device_count=2").strip()   # two host devices when the backend is the CPU
import numpy as np
from colabfold_opt import big

try:
    import jax
    import jax.numpy as jnp
    import haiku as hk
    from alphafold.model import config as af_config, modules as M, modules_multimer as MM, prng
    from opt_core.mem.rowpair_jax import alphafold as recipe, alphafold_template as templ, mesh as rmesh_mod, shard
    jax.devices()                                                        # backend initialised while the flag is set
    HAVE, WHY = hasattr(jax, "__version__"), ""
except Exception as _e:  # noqa: BLE001
    HAVE, WHY = False, repr(_e)
finally:
    if _XLA_BEFORE is None:
        os.environ.pop("XLA_FLAGS", None)
    else:
        os.environ["XLA_FLAGS"] = _XLA_BEFORE

N, T, CQ, SEED, TOL = 24, 4, 64, 7, 1e-5        # residues (a multiple of 2: big pads N to a multiple of P before the model), template slots (multimer max_templates), query pair channels


def _multimer_template_config():
    """The template config of the multimer tree colabfold runs (model_*_multimer_v3), float32 and no remat for the CPU comparison."""
    mc = af_config.model_config("model_1_multimer_v3")
    c, gc = mc.model.embeddings_and_evoformer.template, mc.model.global_config
    gc.use_remat = False; gc.deterministic = True; gc.subbatch_size = 4; gc.bfloat16 = False
    for k, v in (("bfloat16_output", False), ("eval_dropout", False), ("use_flash_attention", False)):
        try:
            setattr(gc, k, v)
        except Exception:  # noqa: BLE001 — a key this tree's global_config does not carry
            pass
    return c, gc


def _template_fields(rs):
    """T slots of an N-residue two-chain target: random atoms around an extended backbone, ~10 % missing atoms; slot 1 is all-dummy (zeros, mask 0)."""
    aatype = rs.randint(0, 21, size=(T, N)).astype("int32")
    pos = (rs.standard_normal((T, N, 37, 3)) * 3.0 + np.arange(N)[None, :, None, None] * np.array([3.8, 0., 0.])[None, None, None, :]).astype("float32")
    amask = (rs.rand(T, N, 37) > 0.1).astype("float32")
    pos[1], amask[1], aatype[1] = 0., 0., 0
    return {"template_aatype": jnp.asarray(aatype), "template_all_atom_positions": jnp.asarray(pos), "template_all_atom_mask": jnp.asarray(amask)}


@unittest.skipUnless(HAVE, "needs jax + dm-haiku + alphafold.model importable in this interpreter: %s" % WHY)
class TemplateFeaturesOnRows(unittest.TestCase):
    def setUp(self):
        self.rs = np.random.RandomState(SEED)
        self.c, self.gc = _multimer_template_config()
        self.fields = _template_fields(self.rs)

    def tearDown(self):
        recipe.uninstall()                                                # every stock attribute restored whatever the outcome

    def _mesh(self, p):
        if p > jax.device_count():
            self.skipTest("needs %d jax devices on the %s backend (it shows %d)" % (p, jax.devices()[0].platform, jax.device_count()))
        return rmesh_mod.build(p, platform=jax.devices()[0].platform, lever=big.NAME, _allow_single_device_mesh=(p == 1))

    def test_distogram_rows_equal_stock_rows_bitwise(self):
        """(1) per template slot and per row block: the row body's distogram == stock dgram_from_positions sliced to the rows, bit for bit."""
        cfg = dict(self.c.dgram_features)                                 # min_bin 3.25, max_bin 50.75, num_bins 39 on the multimer tree
        self.assertEqual(int(cfg["num_bins"]), 39)
        f = self.fields
        for t in range(T):
            pb, _ = M.pseudo_beta_fn(f["template_aatype"][t], f["template_all_atom_positions"][t], f["template_all_atom_mask"][t])
            full = np.asarray(M.dgram_from_positions(pb, **cfg))
            for p in (2, 4):
                r = N // p
                for k in range(p):
                    got = np.asarray(templ.dgram_rows(pb[k * r:(k + 1) * r], pb, lever=big.NAME, **cfg))
                    self.assertEqual(got.shape, (r, N, 39))
                    self.assertTrue(np.array_equal(got, full[k * r:(k + 1) * r]), "slot %d P=%d block %d: max|d|=%g" % (t, p, k, float(np.max(np.abs(got - full[k * r:(k + 1) * r])))))

    def test_template_embedding_rows_equal_stock(self):
        """(2)+(3): the install big requests puts the template term on rows; stock TemplateEmbedding (dense) vs the rows region per P."""
        c, gc, f, rs = self.c, self.gc, self.fields, self.rs
        query = jnp.asarray(rs.standard_normal((N, N, CQ)).astype("float32") + 40.0)
        seq_mask = (rs.rand(N) > 0.1).astype("float32")
        pad = jnp.asarray(seq_mask[:, None] * seq_mask[None, :])
        asym = np.concatenate([np.zeros(N // 2 + 1), np.ones(N - N // 2 - 1)]).astype("int32")
        multi = jnp.asarray(asym[:, None] == asym[None, :])

        def fwd(q, pm, mm, b):
            return MM.TemplateEmbedding(c, gc)(q, b, pm, mm, is_training=False, safe_key=prng.SafeKey(jax.random.PRNGKey(SEED + 1)))
        tf = hk.transform(fwd); key = jax.random.PRNGKey(SEED)
        prec = jax.default_matmul_precision("float32"); prec.__enter__(); self.addCleanup(prec.__exit__, None, None, None)
        params = tf.init(key, query, pad, multi, f)
        stock = np.asarray(jax.jit(tf.apply)(params, key, query, pad, multi, f))          # the dense stock program, nothing installed
        self.assertEqual(stock.shape, (N, N, CQ))
        for p in (1, 2):
            rm = self._mesh(p)
            recipe.install(rm, M, MM, lever=big.NAME, _test_single_device_mesh=(p == 1), **big.RECIPE_WORDS)   # exactly big's request (big.py apply)
            try:
                self.assertEqual(recipe.describe()["template"], "rows", recipe.describe())
                rep, rows = shard.named(rm, shard.replicated_spec()), shard.named(rm, shard.rows_spec(rm.axis, 3, 0))
                jitted = jax.jit(tf.apply, in_shardings=(rep, rep, rows, rep, rep, rep))    # the query pair ROW-sharded in; the output placement left to the program
                args = (shard.put(params, rm), shard.put(key, rm), jax.device_put(query, rows), shard.put(pad, rm), shard.put(multi, rm), shard.put(f, rm))
                built_before = int(templ.describe().get("template_regions_built", 0))
                got = jitted(*args)
                spec = tuple(getattr(got.sharding, "spec", ()))
                built = int(templ.describe().get("template_regions_built", 0)) - built_before
                got = np.asarray(got)
            finally:
                recipe.uninstall()
            self.assertGreaterEqual(built, 1, "P=%d: the template rows region was not traced — the stock body ran" % p)
            mad, scale = float(np.max(np.abs(got - stock))), float(np.max(np.abs(stock)))
            if p == 1:
                self.assertTrue(np.array_equal(got, stock), "1-device mesh: the rows region is not bitwise vs stock (max|d|=%g)" % mad)
            else:
                self.assertTrue(spec and spec[0] == rm.axis and all(s is None for s in spec[1:]), ("the template embedding must leave the program row-sharded", spec))
                self.assertLessEqual(mad, TOL * max(scale, 1.0), "P=%d: max|d|=%g vs stock (scale %g)" % (p, mad, scale))


class TemplateCensus(unittest.TestCase):
    """(4) the per-query TEMPLATES line at P > 1: counted on the host from the query's feature dict."""
    def setUp(self):
        big.reset_for_tests()
        rs = np.random.RandomState(SEED)
        self.mask = (rs.rand(T, N, 37) > 0.1).astype("float32"); self.mask[1] = 0.          # four slots, slot 1 a dummy
        self.multimer = {"template_aatype": np.zeros((T, N), "int32"), "template_all_atom_mask": self.mask, "aatype": np.zeros((N,), "int32")}
        self.monomer = {"template_all_atom_masks": np.stack([self.mask] * 2)}               # the monomer key spelling; a leading ensemble axis is looked through
        self.mock = {"template_aatype": np.zeros((1, N), "int32"), "template_all_atom_mask": np.zeros((1, N, 37), "float32")}   # colabfold's mk_mock_template slot

    def tearDown(self):
        big.reset_for_tests()

    def test_census_counts_real_slots_from_the_atom_mask(self):
        self.assertEqual(big.template_census(self.multimer), (3, 4))
        self.assertEqual(big.template_census(self.monomer), (3, 4))
        self.assertEqual(big.template_census(self.mock), (0, 1))
        self.assertEqual(big.template_census({"aatype": np.zeros((N,), "int32")}), (0, 0))

    def test_line_names_job_real_and_form(self):
        self.assertEqual(big.TEMPLATES_FMT, "{prefix} TEMPLATES job={job} real={real}/{slots} form={form}")
        self.assertEqual(big.templates_line("1BRS_AD", self.mock), "[colabfold-opt] TEMPLATES job=1BRS_AD real=0/1 form=none")      # nothing installed in this process: the recipe's absence is named, never row_born
        self.assertTrue(big.templates_line("9m0h four", self.multimer).startswith("[colabfold-opt] TEMPLATES job=9m0h_four real=3/4 form="))


if __name__ == "__main__":
    unittest.main()
