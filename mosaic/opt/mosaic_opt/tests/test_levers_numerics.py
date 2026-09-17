"""The kit's other per-step numeric levers, distinct from the trimul/attention-kernel family: E1 (dead_template) and E10 (layer_unroll) are
graph-rewrite levers (eliding a real computation under a licensed all-zero input; unrolling a checkpointed scan into a static loop), P6
(precision) is the matmul-precision lever, and P5 (memlevers) is the big-tier memory lever (chunked triangle attention, grouped pairformer
scan, sub-block rematerialisation). Covers, per lever: registry wiring and row-of-record: spec grammar and refusals by name; install/configure/
uninstall round-trips (rebound calls delegate to the stock body off, restore by identity on uninstall); and each lever's own numerics —
E1's skip-mode bit-identity to the real body on an all-zero mask (plus the documented signed-zero caveat) and its named step-asides to the
stock body — never a raise — for what it cannot skip (a real template, an unlicensed trace, an alien licence, a reshaped mask, a biased
projection, a traced mask), counted on its census line; E10's unrolled loop matching the checkpointed scan to
rounding; P6's matmul-precision words applied per joltz region, checked both on stand-in classes and against the stock source; and P5's
chunked/grouped/remat arithmetic against the unchunked stock calls (ragged row counts, ragged pairformer groups, short stacks), plus its
trace ledger and fail-closed gate. Merged from test_dead_template.py (E1), test_layer_unroll.py (E10), test_precision.py (P6) and
test_memlevers.py (P5)."""


import contextlib
import importlib.util
import io
import os
import re
import sys
import types
import unittest

import numpy as np

from . import _stubs
from mosaic_opt import outputs, registry, settings


E1_KIT_FILE = os.path.join(_stubs.KIT, "mosaic_fast", "dead_template.py")


try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_platforms", "cpu") if hasattr(jax.config, "update") else None
    HAVE_JAX = True
except Exception:  # noqa: BLE001
    jax = jnp = None
    HAVE_JAX = False
try:
    import equinox as eqx
    HAVE_EQX = HAVE_JAX
except Exception:  # noqa: BLE001
    eqx = None
    HAVE_EQX = False


XP = jnp if HAVE_JAX else np


DT_COUNT = {"template_body": 0, "build_loss": 0, "build_multisample_loss": 0, "model_output": 0}
B, T, N, D, DZ = 1, 1, 6, 4, 5


def _load_dead_template():
    spec = importlib.util.spec_from_file_location("dead_template_under_test", E1_KIT_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _relu(x):
    return XP.maximum(x, 0)


if HAVE_EQX:
    class _Model(eqx.Module):                    # the stand-in for joltz.Joltz2: a pytree carrying the template module
        template_module: object

    class _Linear(eqx.Module):
        weight: object
        bias: object

        def __call__(self, x):
            o = x @ self.weight.T
            return o if self.bias is None else o + self.bias

    class TemplateV2Module(eqx.Module):          # the stand-in: upstream's signature and its masked-mean / bias-free-projection tail
        u_proj: _Linear
        mix: object

        def __call__(self, z, feats, pair_mask, *, key, deterministic):
            DT_COUNT["template_body"] += 1
            template_mask = XP.any(feats["template_mask"], axis=2)                     # [B, T]
            num_templates = XP.clip(template_mask.sum(axis=1), 1, None)
            v = XP.tanh(z[:, None] @ self.mix) * pair_mask[:, None, :, :, None]      # [B, T, N, N, D] — "the pairformer"
            u = (v * template_mask[:, :, None, None, None]).sum(axis=1) / num_templates[:, None, None, None]
            return self.u_proj(_relu(u))

    class Boltz2(eqx.Module):                    # the stand-in for mosaic.models.boltz2.Boltz2: keyword-only entry points with `features`
        model: object

        def build_loss(self, *, loss=None, features, recycling_steps=1, sampling_steps=None):
            DT_COUNT["build_loss"] += 1
            return StubLoss(self.model, features)

        def build_multisample_loss(self, *, loss=None, features, recycling_steps=1, num_samples=4, sampling_steps=None):
            DT_COUNT["build_multisample_loss"] += 1
            return StubLoss(self.model, features)

        @eqx.filter_jit
        def model_output(self, *, PSSM=None, features, recycling_steps=1, sampling_steps=None, key=None):
            DT_COUNT["model_output"] += 1
            return StubLoss(self.model, features)(features["z"])
else:
    class _Model:
        def __init__(self, template_module):
            self.template_module = template_module

    class _Linear:
        def __init__(self, weight, bias):
            self.weight, self.bias = weight, bias

        def __call__(self, x):
            o = x @ self.weight.T
            return o if self.bias is None else o + self.bias

    class TemplateV2Module:
        def __init__(self, u_proj, mix):
            self.u_proj, self.mix = u_proj, mix

        def __call__(self, z, feats, pair_mask, *, key, deterministic):
            DT_COUNT["template_body"] += 1
            template_mask = XP.any(feats["template_mask"], axis=2)
            num_templates = XP.clip(template_mask.sum(axis=1), 1, None)
            v = XP.tanh(z[:, None] @ self.mix) * pair_mask[:, None, :, :, None]
            u = (v * template_mask[:, :, None, None, None]).sum(axis=1) / num_templates[:, None, None, None]
            return self.u_proj(_relu(u))

    class Boltz2:
        def __init__(self, model):
            self.model = model

        def build_loss(self, *, loss=None, features, recycling_steps=1, sampling_steps=None):
            DT_COUNT["build_loss"] += 1
            return StubLoss(self.model, features)

        def build_multisample_loss(self, *, loss=None, features, recycling_steps=1, num_samples=4, sampling_steps=None):
            DT_COUNT["build_multisample_loss"] += 1
            return StubLoss(self.model, features)

        def model_output(self, *, PSSM=None, features, recycling_steps=1, sampling_steps=None, key=None):
            DT_COUNT["model_output"] += 1
            return StubLoss(self.model, features)(features["z"])


class StubLoss:                                   # what build_loss returns: set_binder_sequence's `features | {...}` then the trunk's `z + template_module(z, feats, ...)` line
    def __init__(self, model, features):
        self.model, self.features = model, features

    def __call__(self, z):
        feats = self.features | {"z": z}                                              # upstream set_binder_sequence: features | {res_type, msa, profile} — every other key kept
        pair_mask = XP.ones((z.shape[0], z.shape[1], z.shape[2]), dtype=z.dtype)
        out = z + self.model.template_module(z, feats, pair_mask, key=None, deterministic=True)   # the trunk: batched feats
        single = _tree_map(lambda v: v[0], feats)                                     # upstream boltz2_forward_from_trunk: every features leaf de-batched (a leaf without a batch axis raises here)
        assert single["template_mask"].shape == feats["template_mask"].shape[1:]
        return out


def _tree_map(f, tree):
    if HAVE_JAX:
        return jax.tree.map(f, tree)
    return {k: f(v) for k, v in tree.items()}


def _features(nonzero=0, t=T, n=N):
    rng = np.random.default_rng(0)
    tm = np.zeros((B, t, n), dtype=np.float32)
    if nonzero:
        tm[0, 0, :nonzero] = 1.0
    z = rng.standard_normal((B, n, n, DZ)).astype(np.float32)
    return {"template_mask": XP.asarray(tm), "z": XP.asarray(z)}


def _model(bias=None):
    rng = np.random.default_rng(1)
    u_proj = _Linear(weight=XP.asarray(rng.standard_normal((DZ, D)).astype(np.float32)), bias=bias)
    tm = TemplateV2Module(u_proj=u_proj, mix=XP.asarray(rng.standard_normal((DZ, D)).astype(np.float32)))
    return Boltz2(model=_Model(template_module=tm))


class TestDeadTemplate(unittest.TestCase):

    def setUp(self):
        self.saved = {k: sys.modules.get(k) for k in ("joltz", "mosaic", "mosaic.models", "mosaic.models.boltz2")}
        joltz = types.ModuleType("joltz"); joltz.TemplateV2Module = TemplateV2Module
        mosaic = types.ModuleType("mosaic"); models = types.ModuleType("mosaic.models"); mb = types.ModuleType("mosaic.models.boltz2")
        mb.Boltz2 = Boltz2; models.boltz2 = mb; mosaic.models = models
        sys.modules.update({"joltz": joltz, "mosaic": mosaic, "mosaic.models": models, "mosaic.models.boltz2": mb})
        self.raw = {(TemplateV2Module, "__call__"): TemplateV2Module.__dict__["__call__"],
                    **{(Boltz2, n): Boltz2.__dict__[n] for n in ("build_loss", "build_multisample_loss", "model_output")}}
        self.dt = _load_dead_template()
        for k in DT_COUNT:
            DT_COUNT[k] = 0

    def tearDown(self):
        try:
            self.dt.uninstall()
        finally:
            for (cls, attr), raw in self.raw.items():          # belt and braces: the classes are this file's, restore them regardless
                setattr(cls, attr, raw)
            for k, v in self.saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_stock_mode_runs_upstreams_body_and_uninstall_restores_every_entry(self):
        dt = self.dt
        self.assertEqual(dt.describe()["installed"], 0)
        dt.install(); dt.configure("stock")
        model, feats = _model(), _features()
        out = model.build_loss(features=feats)(feats["z"])
        self.assertEqual(DT_COUNT["template_body"], 1)                                    # upstream's body ran
        self.assertTrue(np.array_equal(np.asarray(out), np.asarray(feats["z"])))         # and contributed zeros (all-zero mask), as upstream does
        d = dt.describe()
        self.assertEqual((d["mode"], d["contribution"], d["skips_traced"], d["engaged"], d["licensed"]), ("stock", "stock", 0, "no", 0))
        self.assertGreaterEqual(d["stock_traced"], 1)                                    # upstream's body, counted
        self.assertIs(dt.license(feats), feats)                                           # stock: license() hands the dict back untouched (inert)
        real = _features(nonzero=2); self.assertIs(dt.license(real), real)                # ... a real template too: stock semantics, nothing refused
        self.assertEqual(dt.attest(real)["template_mask_nonzero"], 2)                     # facts on request
        dt.uninstall()
        for (cls, attr), raw in self.raw.items():
            self.assertIs(cls.__dict__[attr], raw, f"{cls.__name__}.{attr} not restored by uninstall()")
        self.assertEqual(dt.describe()["installed"], 0)
        self.assertEqual(dt.install()["installed"], 1); dt.install()                     # idempotent
        self.assertIs(TemplateV2Module.__dict__["__call__"], dt._template_call)

    @unittest.skipUnless(HAVE_JAX, "the skip branch builds its zero addend with jax.numpy: needs jax importable")
    def test_skip_on_an_all_zero_mask_skips_the_body_exactly(self):
        dt = self.dt
        dt.install(); dt.configure("skip")
        model, feats = _model(), _features()
        d0 = dt.require_engaged()                                                       # configured, nothing traced yet: reported (engaged=no), never raised
        self.assertEqual((d0["engaged"], d0["skips_traced"], d0["stock_traced"]), ("no", 0, 0)); self.assertNotIn("aside", d0)
        loss = model.build_loss(features=feats)
        self.assertEqual(DT_COUNT["build_loss"], 1)
        self.assertEqual(dt.CFG["licensed"][-1]["template_mask_nonzero"], 0); self.assertEqual(dt.CFG["licensed"][-1]["entry"], "build_loss")
        lic = [k for k in loss.features if k.startswith(dt.LICENCE_PREFIX)]
        self.assertEqual(lic, [dt.LICENCE_PREFIX + dt.CFG["nonce"]]); self.assertEqual(loss.features[lic[0]].shape, (B, 0, T, N))   # batch-leading zero-size leaf, travels with the dict the loss holds
        self.assertEqual(np.asarray(loss.features[lic[0]])[0].shape, (0, T, N))                                                          # de-batches like every leaf (upstream tree.map v[0])
        self.assertNotIn(lic[0], feats)                                                 # the caller's dict is untouched (a copy was licensed)
        out = loss(feats["z"])
        self.assertEqual(DT_COUNT["template_body"], 0)                                     # the pairformer stand-in never ran
        a, b = np.asarray(out), np.asarray(feats["z"])
        self.assertTrue(np.array_equal(a, b) and np.array_equal(np.signbit(a), np.signbit(b)))
        d = dt.require_engaged()                                                        # engaged now: returns describe()
        self.assertEqual((d["mode"], d["contribution"], d["skips_traced"], d["engaged"], d["lever"], d["origin"], d["impl"]), ("skip", "skipped", 1, "yes", "E1", "kit", "dead_template_under_test"))
        self.assertTrue(d["last_licence"].startswith("nonzero:0,templates:1,n_tokens:6,entry:build_loss")); self.assertEqual(len(d["file_sha256"]), 16)
        # model_output (the refold door) licenses too and skips inside its (filter_)jit
        out2 = model.model_output(features=feats, key=None)
        self.assertEqual(DT_COUNT["template_body"], 0)
        self.assertTrue(np.array_equal(np.asarray(out2), b))
        self.assertEqual(dt.CFG["licensed"][-1]["entry"], "model_output")
        # build_multisample_loss likewise
        model.build_multisample_loss(features=feats)(feats["z"]); self.assertEqual(DT_COUNT["template_body"], 0)

    def _aside_lines(self, text, reason=None):
        rx = r"^\[mosaic-opt\] E1 dead_template: .* stepping aside by name \(aside=%s\); the template module runs as stock here$" % (re.escape(reason) if reason else r"\w+")
        return [l for l in text.splitlines() if re.match(rx, l)]

    def test_a_real_template_steps_aside_by_name_and_the_stock_body_runs(self):
        """A featurization with a real template (mosaic `TargetChain(template_chain=…)`) is an input stock accepts: under mode skip every entry point
        passes it through UNLICENSED, says so once on stderr (aside=real_template), counts it, and the trace runs upstream's template module — the
        same numbers as mode stock. Nothing raises, before or after the run (gate())."""
        dt = self.dt
        dt.install(); dt.configure("skip")
        model, feats = _model(), _features(nonzero=3)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loss = model.build_loss(features=feats)
        self.assertEqual(DT_COUNT["build_loss"], 1)                                       # the entry point's body entered: nothing refused
        self.assertIs(loss.features, feats)                                               # handed on untouched ...
        self.assertEqual([k for k in loss.features if k.startswith(dt.LICENCE_PREFIX)], [])   # ... carrying NO licence leaf
        self.assertEqual(dt.CFG["aside"], {"real_template": 1}); self.assertEqual(dt.CFG["licensed"], []); self.assertEqual(dt.describe()["refused"], 0)
        lines = self._aside_lines(err.getvalue(), "real_template")
        self.assertEqual(len(lines), 1, err.getvalue()); self.assertIn("3 nonzero entries over 1 template(s)", lines[0]); self.assertIn("at build_loss", lines[0])
        err2 = io.StringIO()
        with contextlib.redirect_stderr(err2):
            out = loss(feats["z"])                                                        # the trunk traces the template module on the unlicensed dict
            for entry in ("build_loss", "build_multisample_loss"):
                getattr(model, entry)(features=feats)
            out2 = model.model_output(features=feats, key=None)
        self.assertEqual(DT_COUNT["template_body"], 2)                                    # upstream's body ran: the loss call and the refold door
        self.assertFalse(np.array_equal(np.asarray(out), np.asarray(feats["z"])))         # a real template contributes
        self.assertEqual(self._aside_lines(err2.getvalue(), "real_template"), [])         # said ONCE per reason and process (the first time), counted every time
        self.assertEqual(len(self._aside_lines(err2.getvalue(), "unlicensed_loss_path")), 1)   # the trace names its own step-aside once: no licence on the traced dict
        self.assertEqual(dt.CFG["aside"], {"real_template": 4, "unlicensed_loss_path": 2})
        d = dt.gate()                                                                     # after the run: reported, not raised
        self.assertEqual((d["engaged"], d["skips_traced"], d["stock_traced"], d["aside"]), ("no", 0, 2, "real_template:4,unlicensed_loss_path:2"))
        self.assertEqual(d["contribution"], "skipped"); self.assertEqual(d["mode"], "skip")   # the configured treatment; aside= says what this run met
        # the same features under mode stock: nothing licensed, nothing counted, upstream's body runs and gives the same numbers
        dt.configure("stock"); self.assertEqual(dt.CFG["aside"], {}); self.assertNotIn("aside", dt.describe())
        ref = model.build_loss(features=feats)(feats["z"])
        self.assertEqual(DT_COUNT["template_body"], 3); self.assertEqual(dt.CFG["licensed"], [])
        self.assertTrue(np.array_equal(np.asarray(out), np.asarray(ref)))                # the same upstream function on the same operands, both un-jitted: equal bytes
        self.assertTrue(np.allclose(np.asarray(out2), np.asarray(ref), rtol=1e-5, atol=1e-6))   # the refold door runs it inside (filter_)jit when equinox is here: equal to rounding

    @unittest.skipUnless(HAVE_JAX, "the skip branch builds its zero addend with jax.numpy: needs jax importable")
    def test_traces_the_lever_cannot_skip_step_aside_to_the_stock_body(self):
        """Under mode skip a template-module trace skips ONLY on this process's licence for the traced mask's (T, N); every other trace runs upstream's
        body, named once per reason on stderr and counted (aside=unlicensed_loss_path | stale_licence | shape | no_mask) — never a raise."""
        dt = self.dt
        dt.install(); dt.configure("skip")
        model, feats = _model(), _features()
        sentinel = {"n": 0}; orig = dt._ORIG["call:template"]

        def counting(self_, z, f, pair_mask, *, key, deterministic):                     # the stored upstream entry, observed: every step-aside routes here
            sentinel["n"] += 1
            return orig(self_, z, f, pair_mask, key=key, deterministic=deterministic)
        dt._ORIG["call:template"] = counting
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            # (1) a hand-built loss (mosaic.losses.boltz2.Boltz2Loss built directly): bypasses every licensing entry point -> the stock body, named
            out = StubLoss(model.model, feats)(feats["z"])
            self.assertEqual((sentinel["n"], DT_COUNT["template_body"], dt.CFG["aside"]), (1, 1, {"unlicensed_loss_path": 1}))
            self.assertTrue(np.array_equal(np.asarray(out), np.asarray(feats["z"])))     # all-zero mask: upstream's body contributes zeros, as stock does
            licensed = dt.license(feats)                                                  # the documented route to the skip: trace the licensed dict
            self.assertTrue(np.array_equal(np.asarray(StubLoss(model.model, licensed)(feats["z"])), np.asarray(feats["z"])))
            self.assertEqual((sentinel["n"], dt.CFG["skips_traced"]), (1, 1))            # skipped: upstream's body not entered again
            # (2) THE STALE-ATTESTATION CASE: after an all-zero licence, a SAME-SHAPE real template reaching the trunk by a non-licensing path runs the stock body (never skipped)
            model.build_loss(features=_features())                                        # licenses an all-zero dict, shape [1,1,6]
            real = _features(nonzero=3)                                                   # same shape, a real template, hand-built loss
            out_real = StubLoss(model.model, real)(real["z"])
            self.assertEqual(sentinel["n"], 2); self.assertFalse(np.array_equal(np.asarray(out_real), np.asarray(real["z"])))   # the real template contributed
            self.assertIs(dt.license(real), real); self.assertEqual(dt.CFG["aside"]["real_template"], 1)                          # ... and the licensing path passes it through by its real name
            # (3) a licence issued by another install (another process, a features file read back) is stale here -> the stock body
            alien = dict(feats); alien[dt.LICENCE_PREFIX + "0123abcd"] = np.zeros((B, 0, T, N), np.float32)
            StubLoss(model.model, alien)(feats["z"]); self.assertEqual((sentinel["n"], dt.CFG["aside"]["stale_licence"]), (3, 1))
            # (4) a licensed dict whose template_mask was swapped for another shape -> the stock body, by shape
            reshaped = dict(licensed); reshaped["template_mask"] = _features(n=N + 2)["template_mask"]
            pm = XP.ones((B, N, N))
            model.model.template_module(feats["z"], reshaped | {"z": feats["z"]}, pm, key=None, deterministic=True)
            self.assertEqual((sentinel["n"], dt.CFG["aside"]["shape"]), (4, 1))
            flat = dict(licensed); flat[[k for k in flat if k.startswith(dt.LICENCE_PREFIX)][0]] = np.zeros((0, T, N), np.float32)   # a licence leaf without the batch axis: by shape (never an IndexError downstream)
            model.model.template_module(feats["z"], flat | {"z": feats["z"]}, pm, key=None, deterministic=True)
            self.assertEqual((sentinel["n"], dt.CFG["aside"]["shape"]), (5, 2))
            # (5) no template_mask at all under skip: passed through by name (upstream decides what a dict without it means); a second licence replaces the first leaf
            nomask = model.build_loss(features={"z": feats["z"]})
            self.assertEqual(list(nomask.features), ["z"]); self.assertEqual(dt.CFG["aside"]["no_mask"], 1)
            twice = dt.license(dt.license(feats)); self.assertEqual(len([k for k in twice if k.startswith(dt.LICENCE_PREFIX)]), 1)
            # (6) a real template arriving through an entry point inside a dict licensed EARLIER: the stale licence is dropped, the trace runs the stock body
            edited = dict(licensed); edited["template_mask"] = _features(nonzero=2)["template_mask"]
            relic = model.build_loss(features=edited)
            self.assertEqual([k for k in relic.features if k.startswith(dt.LICENCE_PREFIX)], []); self.assertEqual(dt.CFG["aside"]["real_template"], 2)
            relic(feats["z"]); self.assertEqual(sentinel["n"], 6)
        text = err.getvalue()
        for reason, n in (("unlicensed_loss_path", 1), ("real_template", 1), ("stale_licence", 1), ("shape", 1), ("no_mask", 1), ("traced_mask", 0), ("u_proj_bias", 0)):
            self.assertEqual(len(self._aside_lines(text, reason)), n, f"{reason}: {text}")
        d = dt.gate()                                                                     # engaged (one licensed skip) AND asides: both on the census, nothing raised
        self.assertEqual((d["engaged"], d["skips_traced"]), ("yes", 1)); self.assertEqual(d["stock_traced"], sentinel["n"])
        self.assertEqual(d["aside"], "unlicensed_loss_path:3,real_template:2,stale_licence:1,shape:2,no_mask:1")   # first-seen order, name:count
        self.assertEqual(set(k.split(":")[0] for k in d["aside"].split(",")) - set(dt.ASIDES), set())
        # uninstall drops the nonce and the counts: a dict licensed before is stale after a re-install -> the stock body again, named afresh
        dt.uninstall(); self.assertEqual(dt.CFG["aside"], {}); dt.install(); dt.configure("skip")
        sentinel["n"] = 0; orig = dt._ORIG["call:template"]; dt._ORIG["call:template"] = counting
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            StubLoss(model.model, licensed)(feats["z"])
        self.assertEqual((sentinel["n"], dt.CFG["aside"]), (1, {"stale_licence": 1})); self.assertEqual(len(self._aside_lines(err.getvalue(), "stale_licence")), 1)

    def test_a_biased_u_proj_steps_aside(self):
        dt = self.dt
        dt.install(); dt.configure("skip")
        model, feats = _model(bias=XP.asarray(np.ones((DZ,), np.float32))), _features()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            out = model.build_loss(features=feats)(feats["z"])                           # licensed (all-zero mask), but the projection's bias makes the addend nonzero
        self.assertEqual((DT_COUNT["template_body"], dt.CFG["aside"], dt.CFG["skips_traced"]), (1, {"u_proj_bias": 1}, 0))
        self.assertEqual(len(self._aside_lines(err.getvalue(), "u_proj_bias")), 1)
        self.assertTrue(np.allclose(np.asarray(out) - np.asarray(feats["z"]), np.ones((DZ,), np.float32)))   # upstream's tail: relu(0) @ W.T + bias = the bias, added to z
        self.assertEqual(dt.gate()["aside"], "u_proj_bias:1")

    def test_gate_refuses_only_the_levers_name_over_mode_stock(self):
        """gate() (levers.finalize) raises for ONE thing: installed but left in mode stock (kit-internal misuse). A run that met a step-aside, or one
        that traced nothing, is reported on the census line and passes."""
        dt = self.dt
        self.assertEqual(dt.gate()["installed"], 0)                                       # not installed: nothing to check
        dt.install()                                                                      # installed, mode stock (configure() never called)
        with self.assertRaises(dt.LeverRefused) as cm:
            dt.gate()
        self.assertIn("dead_template_not_configured", str(cm.exception))
        dt.configure("stock")
        with self.assertRaises(dt.LeverRefused) as cm:                                    # the explicit stock word under the lever's name: the same refusal
            dt.gate()
        self.assertIn("dead_template_not_configured", str(cm.exception))
        dt.configure("skip")
        self.assertEqual(dt.gate()["engaged"], "no")                                      # mode skip, nothing traced: passes, engaged=no
        with contextlib.redirect_stderr(io.StringIO()):
            _model().build_loss(features=_features(nonzero=1))
        d = dt.gate(); self.assertEqual((d["engaged"], d["aside"]), ("no", "real_template:1"))
        for k in ("setting_of_record", "contribution", "engaged", "skips_traced", "stock_traced", "licensed", "refused", "last_licence"):
            self.assertIn(k, d)                                                           # today's census keys, all present

    def test_spec_words_and_describe_grammar(self):
        dt = self.dt
        self.assertEqual(dt.MODES, ("stock", "skip")); self.assertEqual(dt.ENV_REQUIRED, {}); self.assertEqual((dt.LEVER, dt.NAME), ("E1", "dead_template"))
        for spec in ("skip", None, ""):
            with self.assertRaises(dt.LeverRefused) as cm:
                dt.configure(spec)                                                      # mode skip before install(): would change nothing -> refused by name
            self.assertIn("dead_template_not_installed", str(cm.exception))
        self.assertEqual(dt.configure("stock")["mode"], "stock")                        # the explicit stock word needs nothing installed
        with self.assertRaises(ValueError):
            dt.configure("bogus")
        dt.install()
        for spec, mode in ((None, "skip"), ("", "skip"), ("stock", "stock"), ("skip", "skip"), ("SKIP", "skip"), ("stock", "stock"), (None, "skip")):
            self.assertEqual(dt.configure(spec)["mode"], mode, f"configure({spec!r})")                    # None / '' apply the default SETTING (skip)
        self.assertEqual((dt.SETTING, dt.describe()["setting_of_record"]), ("skip", "skip"))
        self.assertNotIn("name", dt.describe()); self.assertEqual(dt.describe()["lever_name"], "dead_template")   # the LEVER line's reserved slots (name/state/tag/reason) are never describe() keys
        for k in ("state", "tag", "reason"):
            self.assertNotIn(k, dt.describe())
        dt.configure("skip"); _model().build_loss(features=_features())
        self.assertNotIn("aside", dt.describe())                                          # no step-aside: the census carries exactly today's keys
        with contextlib.redirect_stderr(io.StringIO()):
            _model().build_loss(features=_features(nonzero=2))
        with_aside = dt.describe(); self.assertEqual(with_aside["aside"], "real_template:1")
        for state in (with_aside, dt.describe(), dt.configure("stock"), dt.uninstall()):
            for k, v in state.items():
                self.assertFalse(any(c.isspace() for c in str(v)), f"describe()[{k!r}]={v!r} carries a blank (lever_line values are single tokens)")
        try:
            from opt_core import report
        except Exception:  # noqa: BLE001 — the core is not importable here: the grammar check above stands alone
            return
        d = dt.describe()
        line = report.lever_line("mosaic-opt", d["lever_name"], "off", impl=d["impl"], origin=d["origin"], reason=None, mode=d["mode"], engaged=d["engaged"], last_licence=d["last_licence"])
        self.assertIn("LEVER name=dead_template state=off impl=dead_template_under_test origin=kit", line)

    @unittest.skipUnless(HAVE_JAX, "jax not installed: tracers come from jax")
    def test_stock_mode_is_inert_on_traced_features_and_skip_steps_aside_on_them(self):
        dt = self.dt
        dt.install(); dt.configure("stock")
        feats = _features()
        seen = {}

        def f(tm):
            traced = dict(feats); traced["template_mask"] = tm
            seen["stock"] = dt.license(traced) is traced                                # stock: handed back untouched, no refusal on a tracer
            seen["facts"] = dt.attest(traced)                                           # nothing concrete to read: {}
            return tm.sum()
        jax.jit(f)(feats["template_mask"])
        self.assertEqual((seen["stock"], seen["facts"]), (True, {}))
        dt.configure("skip")

        def g(tm):
            traced = dict(feats); traced["template_mask"] = tm
            seen["skip"] = dt.license(traced) is traced                                 # a traced mask: nothing concrete to check -> handed back unlicensed, named
            return tm.sum()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            jax.jit(g)(feats["template_mask"])
        self.assertEqual((seen["skip"], dt.CFG["aside"], dt.CFG["licensed"]), (True, {"traced_mask": 1}, []))
        self.assertEqual(len(self._aside_lines(err.getvalue(), "traced_mask")), 1)

    @unittest.skipUnless(HAVE_JAX, "jax not installed: the signed-zero arithmetic check runs on jax CPU")
    def test_the_stock_tail_is_a_signed_zero(self):
        """Upstream's tail with an all-zero mask: (v*0).sum(1)/1 -> relu -> bias-free projection is +-0.0 elementwise, and z + it == z for every z
        except an exact -0.0 (IEEE: -0.0 + +0.0 = +0.0) — the one bit-level difference between `z + 0.0` (stock) and `z` (skip)."""
        rng = np.random.default_rng(2)
        v = jnp.asarray(rng.standard_normal((B, T, N, N, D)).astype(np.float32))
        mask = jnp.zeros((B, T, N), jnp.float32)
        tmask = jnp.any(mask, axis=2)
        u = (v * tmask[:, :, None, None, None]).sum(axis=1) / jnp.clip(tmask.sum(axis=1), 1, None)[:, None, None, None]
        w = jnp.asarray(rng.standard_normal((DZ, D)).astype(np.float32))
        t = jax.nn.relu(u) @ w.T
        self.assertTrue(bool((t == 0).all()))
        z = jnp.asarray(rng.standard_normal((B, N, N, DZ)).astype(np.float32))
        s = z + t
        self.assertTrue(np.array_equal(np.asarray(s), np.asarray(z)) and np.array_equal(np.signbit(np.asarray(s)), np.signbit(np.asarray(z))))
        zneg = z.at[0, 0, 0, 0].set(-0.0)
        sneg = np.asarray(zneg + jnp.zeros_like(t))                                     # stock's best case: t exactly +0.0
        self.assertTrue(np.signbit(np.asarray(zneg))[0, 0, 0, 0]); self.assertFalse(np.signbit(sneg)[0, 0, 0, 0])   # -0.0 + +0.0 -> +0.0: the caveat is real

    def test_registry_entry_and_row(self):
        """E1 is a per-step lever of the fast tier (first in its install order): wired, route install, module mosaic.fast.dead_template, reached by
        the ONE flag `--levers fast`; its probe is the installer's evidence; no driver flag of its own, no adapter module, no tools/ twin."""
        from mosaic_opt import modes, outputs, registry
        lv = registry.LEVERS["E1"]
        self.assertEqual(lv.kit_file, f"{registry.KIT_FAST_DIR}/dead_template.py")
        self.assertEqual((lv.route, lv.module, lv.flag, lv.wired, lv.klass), ("install", "mosaic.fast.dead_template", "--levers fast", True, "fast"))
        self.assertEqual(lv.probe, registry.probe_key("E1")); self.assertIn("E1", registry.WIRED); self.assertIn("E1", registry.IN_PROCESS)
        self.assertEqual(modes.install_levers_of("fast")[0], "E1")                                   # exact-algebra lever first, then the fast-class levers
        drv = open(os.path.join(_stubs.KIT, "tools", "public_design_run.py"), encoding="utf-8").read()
        self.assertNotIn("dead_template", drv); self.assertNotIn("--dead-template", drv)
        self.assertFalse(os.path.exists(os.path.join(_stubs.KIT, "tools", "dead_template.py")))
        self.assertIsNone(importlib.util.find_spec("mosaic_opt.graph_arms"))
        self.assertEqual(self.dt.configure.__defaults__, (None,)); self.assertEqual(self.dt.SETTING, "skip"); self.assertEqual(self.dt.ENV_REQUIRED, {})


E10_KIT_FILE = os.path.join(_stubs.KIT, "mosaic_fast", "layer_unroll.py")


try:
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import einops  # noqa: F401
    HAVE_STACK = True
except Exception:  # noqa: BLE001
    HAVE_STACK = False


if HAVE_STACK:
    LU_COUNT = {"scan_body": 0}

    class Layer(eqx.Module):                       # a DiffusionTransformerLayer2 stand-in: a(n, d) mixed with s and a per-layer pair bias
        w: object
        v: object

        def __call__(self, *, a, s, bias, mask, to_keys=None):
            h = jnp.tanh(a @ self.w + s @ self.v) * mask[..., None]
            return a + h + 0.01 * jnp.einsum("bnmp->bn", bias)[..., None]

    class DiffusionTransformer2(eqx.Module):       # upstream's fields and __call__ (joltz DiffusionTransformer2), the scan body checkpointed
        pair_bias_attn: bool
        stacked_parameters: Layer
        static: Layer
        depth: int

        def __call__(self, a, s, bias=None, mask=None, to_keys=None, multiplicity=1):
            if self.pair_bias_attn:
                B, N, M, D = bias.shape
                L = self.depth
                bias = bias.reshape(B, N, M, L, D // L)
            bias = einops.rearrange(bias, "... l p -> l ... p")

            @jax.checkpoint
            def body_fn(a, params_and_bias):
                LU_COUNT["scan_body"] += 1
                params, bias = params_and_bias
                layer = eqx.combine(self.static, params)
                return layer(a=a, s=s, bias=bias, mask=mask, to_keys=to_keys), None

            return jax.lax.scan(body_fn, a, (self.stacked_parameters, bias))[0]

    def _module(depth=4, d=8):
        rng = np.random.default_rng(0)
        layers = [Layer(w=jnp.asarray(rng.standard_normal((d, d)).astype(np.float32) / 4), v=jnp.asarray(rng.standard_normal((d, d)).astype(np.float32) / 4)) for _ in range(depth)]
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        stacked = jax.tree.map(lambda *v: jnp.stack(v, 0), *[eqx.filter(l, eqx.is_inexact_array) for l in layers])
        return DiffusionTransformer2(True, stacked, static, depth)

    def _inputs(depth=4, d=8, B=1, N=5, P=3):
        rng = np.random.default_rng(1)
        return (jnp.asarray(rng.standard_normal((B, N, d)).astype(np.float32)), jnp.asarray(rng.standard_normal((B, N, d)).astype(np.float32)),
                jnp.asarray(rng.standard_normal((B, N, N, depth * P)).astype(np.float32)), jnp.ones((B, N), jnp.float32))


def _load_layer_unroll():
    spec = importlib.util.spec_from_file_location("layer_unroll_under_test", E10_KIT_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAVE_STACK, "jax + equinox + einops not installed: the stand-in transformer needs them")
class TestLayerUnroll(unittest.TestCase):

    def setUp(self):
        self.saved = sys.modules.get("joltz")
        joltz = types.ModuleType("joltz"); joltz.DiffusionTransformer2 = DiffusionTransformer2
        sys.modules["joltz"] = joltz
        self.raw = DiffusionTransformer2.__dict__["__call__"]
        self.lu = _load_layer_unroll()
        LU_COUNT["scan_body"] = 0

    def tearDown(self):
        try:
            self.lu.uninstall()
        finally:
            setattr(DiffusionTransformer2, "__call__", self.raw)
            if self.saved is None:
                sys.modules.pop("joltz", None)
            else:
                sys.modules["joltz"] = self.saved

    def test_stock_mode_is_upstreams_scan_and_uninstall_restores(self):
        lu = self.lu
        lu.install(); lu.configure("stock")
        m, (a, s, bias, mask) = _module(), _inputs()
        out = jax.jit(lambda a: m(a, s, bias=bias, mask=mask))(a)
        self.assertGreaterEqual(LU_COUNT["scan_body"], 1); self.assertEqual(lu.describe()["unrolled_traced"], 0)
        self.assertEqual(out.shape, a.shape)
        lu.uninstall()
        self.assertIs(DiffusionTransformer2.__dict__["__call__"], self.raw); self.assertFalse(lu.installed())

    def test_sampler_mode_unrolls_and_equals_the_scan(self):
        lu = self.lu
        m, (a, s, bias, mask) = _module(), _inputs()
        ref = np.asarray(jax.jit(lambda a: m(a, s, bias=bias, mask=mask))(a))              # upstream's scan, before install
        gref = np.asarray(jax.jit(jax.grad(lambda a: m(a, s, bias=bias, mask=mask).sum()))(a))
        lu.install(); lu.configure("sampler")
        LU_COUNT["scan_body"] = 0
        out = np.asarray(jax.jit(lambda a: m(a, s, bias=bias, mask=mask))(a))
        gout = np.asarray(jax.jit(jax.grad(lambda a: m(a, s, bias=bias, mask=mask).sum()))(a))
        self.assertEqual(LU_COUNT["scan_body"], 0)                                            # the scan body never traced: the static loop ran
        d = lu.describe()
        self.assertEqual((d["mode"], d["stacks"], d["depths"]), ("sampler", "DiffusionTransformer2", "4")); self.assertGreaterEqual(d["unrolled_traced"], 1)
        # the same layers in the same order — equal to rounding: XLA fuses across the layer boundaries the static loop exposes (observed on CPU:
        # 1-2 ulp), so BITWISE equality with the scan is NOT a property of this lever; GPU runs give its class (expected fast, not exact)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(gout, gref, rtol=1e-5, atol=1e-6)
        lu.configure("stock"); LU_COUNT["scan_body"] = 0
        np.asarray(jax.jit(lambda a: m(a, s, bias=bias, mask=mask))(a)); self.assertGreaterEqual(LU_COUNT["scan_body"], 1)   # back on the scan

    def test_spec_words_and_describe_grammar(self):
        lu = self.lu
        self.assertEqual((lu.SETTING, lu.ENV_REQUIRED, lu.LEVER, lu.NAME), ("sampler", {}, "E10", "layer_unroll"))
        for spec in ("sampler", None, ""):
            with self.assertRaises(lu.LeverRefused) as cm:
                lu.configure(spec)                                                      # a non-stock mode before install(): refused by name
            self.assertIn("layer_unroll_not_installed", str(cm.exception))
        with self.assertRaises(ValueError):
            lu.configure("all-of-them")
        self.assertEqual(lu.gate()["installed"], 0)                                     # not installed: nothing to gate
        lu.install()
        with self.assertRaises(lu.LeverRefused) as cm:                                  # installed, mode stock: the silent-off trap
            lu.gate()
        self.assertIn("layer_unroll_not_configured", str(cm.exception))
        for spec, mode in ((None, "sampler"), ("", "sampler"), ("stock", "stock"), ("sampler", "sampler"), ("SAMPLER", "sampler")):
            self.assertEqual(lu.configure(spec)["mode"], mode, f"configure({spec!r})")   # None / '' = the default SETTING
        for check in (lu.gate, lu.require_engaged):                                    # configured, nothing traced yet: the gate fails closed (one function)
            with self.assertRaises(lu.LeverRefused) as cm:
                check()
            self.assertIn("layer_unroll_not_engaged", str(cm.exception))
        states = {"configured": dict(lu.describe())}
        model, (a, s_, bias, mask) = _module(), _inputs()
        np.asarray(jax.jit(lambda a: model(a, s_, bias=bias, mask=mask))(a))                # served traffic through the unrolled path
        states["after_traffic"] = dict(lu.gate())                                       # engaged: the gate passes
        self.assertEqual((states["after_traffic"]["engaged"], states["after_traffic"]["unrolled_traced"], states["after_traffic"]["depths"]), ("yes", 1, "4"))
        states["stock"] = dict(lu.configure("stock")); states["uninstalled"] = dict(lu.uninstall()); states["before_install"] = dict(lu.describe())
        self.assertEqual(lu.describe(), states["before_install"])                                                 # restored exactly
        try:
            from mosaic_opt import levers as LV
            token_ok = LV._token_ok
        except Exception:  # noqa: BLE001 — the package is not importable here: the local single-token rule stands in
            token_ok = lambda v: isinstance(v, (str, int, float, bool, type(None))) and not any(c.isspace() for c in str(v)) and str(v) != ""
        for name, d in states.items():
            for k in ("name", "state", "tag", "reason"):
                self.assertNotIn(k, d, f"{name}: describe() carries the LEVER line's reserved slot {k!r}")
            self.assertEqual((d["lever_name"], d["origin"], d["setting_of_record"]), ("layer_unroll", "kit", "sampler"))
            for k, v in d.items():
                self.assertTrue(token_ok(k) and token_ok(v), f"{name}: describe()[{k!r}]={v!r} is not a single-token scalar")
        try:
            from opt_core import report  # noqa: F401
        except Exception:  # noqa: BLE001
            return
        import contextlib, io
        lu.install(); lu.configure(None)
        with contextlib.redirect_stderr(io.StringIO()):
            on = lu.emit_line("mosaic-opt"); lu.configure("stock"); off = lu.emit_line("mosaic-opt"); lu.uninstall(); ni = lu.emit_line("mosaic-opt")
        self.assertIn("LEVER name=E10.layer_unroll state=on impl=", on); self.assertIn("origin=kit", on); self.assertIn("engaged=no", on)
        self.assertIn("state=off reason=not_configured", off); self.assertIn("state=off reason=not_installed", ni)


P6_KIT_FILE = os.path.join(_stubs.KIT, "mosaic_fast", "precision.py")
JOLTZ_SRC = os.path.join(_stubs.TREE, "stock", "src", "joltz", "src", "joltz", "__init__.py")


def _load_precision():
    spec = importlib.util.spec_from_file_location("mosaic_fast_precision_under_test", P6_KIT_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _have(*names):
    return all(importlib.util.find_spec(n) is not None for n in names)


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestPrecisionWords(unittest.TestCase):

    def setUp(self):
        self.P = _load_precision()

    def test_words_are_the_cores_matmul_vocabulary(self):
        from opt_core.precision.policy import MATMUL_PRECISIONS
        self.assertEqual(self.P.words(), tuple(MATMUL_PRECISIONS))
        self.assertEqual(set(self.P.JAX_WORDS), set(MATMUL_PRECISIONS))
        self.assertEqual(self.P.JAX_WORDS, {"highest": "highest", "high": "tensorfloat32", "medium": "BF16_BF16_F32"})

    def test_regions_and_stock_words(self):
        self.assertEqual(self.P.REGIONS, ("trunk", "diffusion", "confidence"))
        self.assertEqual(set(self.P.TARGETS), set(self.P.REGIONS))
        self.assertEqual(set(self.P.STOCK_WORDS), set(self.P.REGIONS))
        self.assertEqual(self.P.STOCK_WORDS["diffusion"], "highest")          # mosaic's float32 island around structure_module.sample
        src = open(os.path.join(_stubs.TREE, "stock", "src", "mosaic", "src", "mosaic", "losses", "boltz2.py"), encoding="utf-8").read()
        self.assertRegex(src, r'with jax\.default_matmul_precision\("float32"\):\n\s+structure_coordinates = model\.structure_module\.sample\(')

    def test_parse_grammar(self):
        P = self.P
        self.assertEqual(P.parse("stock"), {"trunk": None, "diffusion": None, "confidence": None})
        self.assertEqual(P.parse(None), P.parse(P.SETTING))                          # None = the default setting
        self.assertEqual(P.SETTING, "diffusion=high")
        self.assertEqual(P.ENV_REQUIRED, {})
        self.assertEqual(P.parse("diffusion=high")["diffusion"], "high")
        both = P.parse(" confidence=medium , trunk=medium ")
        self.assertEqual((both["trunk"], both["diffusion"], both["confidence"]), ("medium", None, "medium"))
        self.assertEqual(P.canonical(both), "trunk=medium,confidence=medium")        # REGIONS order, one spelling per policy
        self.assertEqual(P.canonical(P.parse("stock")), "stock")
        for bad in ("", "  ", "diffusion", "foo=high", "trunk=bf16", "trunk=tensorfloat32", "diffusion=high,trunk", "=high"):   # an empty spec refuses by name (never silently stock)
            with self.assertRaises(P.PrecisionSpecError, msg=bad):
                P.parse(bad)

    def test_every_wrapped_entry_point_exists_in_joltz_at_the_pin(self):
        src = open(JOLTZ_SRC, encoding="utf-8").read()
        for region, targets in self.P.TARGETS.items():
            for cls_name, attr in targets:
                m = re.search(r"^class %s\(" % re.escape(cls_name), src, re.M)
                self.assertIsNotNone(m, f"{region}: class {cls_name} not in joltz")
                nxt = re.search(r"^class \w+\(", src[m.end():], re.M)
                body = src[m.end(): m.end() + nxt.start()] if nxt else src[m.end():]
                self.assertRegex(body, r"\n    def %s\(" % re.escape(attr), f"{region}: {cls_name}.{attr} not defined in joltz")

    def test_configure_refuses_before_install(self):
        P = self.P
        self.assertFalse(P.installed())
        with self.assertRaises(RuntimeError):
            P.configure("diffusion=high")
        self.assertEqual(P.canonical(), "stock")                                   # a refused word is not recorded

    def test_registry_entry_and_row(self):
        """P6 is the fast tier's per-step lever: wired, class fast, switched by the ONE flag `--levers fast` (row T_fast) / `levers.install("fast")`,
        its module the installed `mosaic.fast.precision`, its probe the driver's post-phase record `manifest.levers.P6.state`."""
        from mosaic_opt import modes
        lv = registry.LEVERS["P6"]
        self.assertEqual((lv.cls, lv.tier, lv.wired, lv.klass, lv.route, lv.module, lv.flag, lv.origin), ("forward", "tier2", True, "fast", "install", "mosaic.fast.precision", "--levers fast", "kit"))
        self.assertTrue(os.path.isfile(os.path.join(_stubs.KIT, lv.kit_file)))
        self.assertIn("P6", registry.WIRED); self.assertIn("P6", registry.INSTALL)
        self.assertEqual(lv.probe, registry.probe_key("P6"))
        fast = list(modes.install_levers_of("fast")); self.assertIn("P6", fast)
        self.assertLess(fast.index("P6"), fast.index("K1")); self.assertEqual([l for l in fast[:fast.index("P6")] if l != "E1"], [])   # only the exact-algebra lever E1 installs before P6 lever
        self.assertIn("fast", modes.MODES); self.assertEqual(modes.ROWS[modes.KIT_MODES["fast"]["row"]]["text"], 'run_arm T_fast "JAX_COMPILATION_CACHE_DIR=$C1 JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0" --seed 0 --weights fastinit --levers fast')
        on = {"manifest": {"levers": {"P6": {"state": "on", "spec": None, "numerics_class": "fast"}}}}
        off = {"manifest": {"precision_statement": {"note": "kit levers never change precision, dtypes or XLA numerics flags"}}}
        self.assertTrue(outputs.classify(on)["P6"]); self.assertFalse(outputs.classify(off)["P6"])
        self.assertNotIn("--precision", settings.driver_defaults(_stubs.KIT))                  # no lever-specific driver flag: the ONE flag is --levers
        drv = open(os.path.join(_stubs.KIT, "tools", "public_design_run.py"), encoding="utf-8").read()
        self.assertNotIn("args.precision", drv); self.assertIn('p.add_argument("--levers"', drv)


@unittest.skipUnless(_stubs.tree_present() and _have("jax", "equinox"), "jax + equinox not importable here (the stack's image runs this class)")
class TestPrecisionInstallRoundTrip(unittest.TestCase):
    """install() wraps stand-in classes registered as `joltz`, a configured word is the matmul precision jax sees inside the region's call and
    nowhere else, uninstall() puts the very objects back."""

    def setUp(self):
        import jax
        self.jax = jax
        self.P = _load_precision()
        seen = self.seen = {}

        class Joltz2:
            def embed_inputs(self, feats):
                seen["embed_inputs"] = str(jax.config.jax_default_matmul_precision); return ("embed", feats)

            def trunk_iteration(self, state):
                seen["trunk_iteration"] = str(jax.config.jax_default_matmul_precision); return ("trunk", state)

        class DistogramModule2:
            def __call__(self, z):
                seen["distogram"] = str(jax.config.jax_default_matmul_precision); return z

        class DiffusionConditioning2:
            def __call__(self, s, z):
                seen["conditioning"] = str(jax.config.jax_default_matmul_precision); return (s, z)

        class AtomDiffusion2:
            def sample(self, **kw):
                seen["sample"] = str(jax.config.jax_default_matmul_precision); return kw

        class ConfidenceModule2:
            def __call__(self, *, s):
                seen["confidence"] = str(jax.config.jax_default_matmul_precision); return s

        fake = types.ModuleType("joltz")
        for c in (Joltz2, DistogramModule2, DiffusionConditioning2, AtomDiffusion2, ConfidenceModule2):
            setattr(fake, c.__name__, c)
        self._saved = sys.modules.get("joltz")
        sys.modules["joltz"] = fake
        self.fake = fake
        self.originals = {(c, a): getattr(fake, c).__dict__[a] for r, ts in self.P.TARGETS.items() for c, a in ts}

    def tearDown(self):
        if self.P.installed():
            self.P.uninstall()
        if self._saved is None:
            sys.modules.pop("joltz", None)
        else:
            sys.modules["joltz"] = self._saved

    def test_round_trip(self):
        P, J = self.P, self.fake
        unset = str(self.jax.config.jax_default_matmul_precision)
        P.install()
        with self.assertRaises(RuntimeError):                                       # one installer per process
            P.install()
        self.assertTrue(P.installed())
        rec0 = P.configure(None)                                                    # the default setting
        self.assertEqual((rec0["spec"], rec0["diffusion"], rec0["trunk"]), ("diffusion=high", "high", None))
        rec = P.configure("diffusion=high,trunk=medium")
        self.assertEqual(rec["spec"], "trunk=medium,diffusion=high")
        self.assertEqual(rec["numerics_class"], "fast")
        self.assertEqual((rec["jax_trunk"], rec["jax_diffusion"], rec["jax_confidence"]), ("BF16_BF16_F32", "tensorfloat32", None))
        self.assertFalse(any(isinstance(v, (dict, list)) for v in rec.values()))     # a flat record
        m = J.Joltz2()
        self.assertEqual(m.embed_inputs({"a": 1}), ("embed", {"a": 1}))              # results pass through unchanged
        self.assertEqual(m.trunk_iteration(3), ("trunk", 3))
        self.assertEqual(J.AtomDiffusion2().sample(x=1), {"x": 1})
        self.assertEqual(J.ConfidenceModule2()(s=5), 5)
        J.DistogramModule2()(0); J.DiffusionConditioning2()(1, 2)
        with self.jax.default_matmul_precision("float32"):                        # mosaic's island: the region's own word still wins inside the call
            J.AtomDiffusion2().sample()
        self.assertEqual(self.seen["sample"], "tensorfloat32")
        for k in ("embed_inputs", "trunk_iteration", "distogram", "conditioning"):
            self.assertEqual(self.seen[k], "BF16_BF16_F32", k)
        self.assertEqual(self.seen["confidence"], unset)                          # an untouched region sees the caller's precision
        self.assertEqual(str(self.jax.config.jax_default_matmul_precision), unset)   # nothing leaks out of the calls
        self.assertIn("class=fast", P.line())
        P.configure("stock")
        J.AtomDiffusion2().sample()
        self.assertEqual(self.seen["sample"], unset)
        self.assertEqual(P.describe()["numerics_class"], "stock")
        P.uninstall()
        self.assertFalse(P.installed())
        for (c, a), raw in self.originals.items():
            self.assertIs(getattr(J, c).__dict__[a], raw, f"{c}.{a} not restored")


P5_KIT_FILE = os.path.join(_stubs.KIT, "mosaic_fast", "memlevers.py")                 # the lever file (registry.LEVERS["P5"].kit_file), not the pre-0.3 tools/ path


def _load_memlevers():
    spec = importlib.util.spec_from_file_location("memlevers_under_test", P5_KIT_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestMemleversWiring(unittest.TestCase):

    def test_registry_entry_and_row(self):
        """P5 is the big tier's memory lever: wired, route install, module mosaic.fast.memlevers, reached by the ONE flag `--levers big`
        (row U_big) — installed after the fast tier's levers; its probe is the installer's evidence; the driver carries no flag of its own."""
        from mosaic_opt import modes
        lv = registry.LEVERS["P5"]
        self.assertEqual((lv.route, lv.module, lv.flag, lv.wired, lv.klass), ("install", "mosaic.fast.memlevers", "--levers big", True, "big"))
        self.assertEqual(lv.probe, registry.probe_key("P5")); self.assertTrue(outputs.classify({"manifest": {"levers": {"P5": {"state": "on"}}}})["P5"])
        big = modes.install_levers_of("big"); fast = modes.install_levers_of("fast"); self.assertEqual(big[len(fast) - 1], "P5"); self.assertEqual(tuple(l for l in big if l != "P5"), fast)   # the fast tier's levers with P5 before the last of them (P7 wraps P5's scan)
        self.assertEqual(modes.KIT_MODES["big"]["row"], "U_big"); self.assertIn("big", modes.MODES)
        d = open(os.path.join(_stubs.KIT, "tools", "public_design_run.py"), encoding="utf-8").read()
        self.assertNotIn("--memlevers", d); self.assertNotIn("memlevers.configure", d)


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestMemleversArithmetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        missing = [m for m in ("jax", "equinox", "einops") if importlib.util.find_spec(m) is None]
        if missing:
            raise unittest.SkipTest(f"memlevers arithmetic needs {missing} (CPU jax is enough)")
        import jax
        jax.config.update("jax_platforms", "cpu") if hasattr(jax.config, "update") else None
        cls.M = _load_memlevers()

    def setUp(self):
        self.M._ORIG["triatt"] = type(self._triatt_stub(True)).stock_call   # the call install() would have captured: joltz's own body (unbound: takes self; touches only self.*)
        self.M.MEM.update({"triatt_chunk": None, "pf_group": None, "sub_remat": False}); self.M.TRACES.clear()

    @staticmethod
    def _triatt_stub(starting, c_in=8, heads=2, d=4, seed=0):
        import jax, jax.numpy as jnp
        ks = jax.random.split(jax.random.key(seed), 5)
        Wq, Wk, Wv = (jax.random.normal(k, (c_in, heads, d)) * 0.3 for k in ks[:3]); Wo = jax.random.normal(ks[3], (heads * d, c_in)) * 0.3; Wb = jax.random.normal(ks[4], (c_in, heads)) * 0.3

        class Stub:
            inf = 1e9

            def layer_norm(self, x):
                mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
                return (x - mu) / jnp.sqrt(var + 1e-5)

            def linear(self, x):
                return x @ Wb                                                    # [..., I, J, H]

            def mha(self, q_x, kv_x, biases):
                q = jnp.einsum("...qc,chd->...qhd", q_x, Wq); k = jnp.einsum("...kc,chd->...khd", kv_x, Wk); v = jnp.einsum("...kc,chd->...khd", kv_x, Wv)
                logits = jnp.einsum("...qhd,...khd->...hqk", q, k) / (d ** 0.5)
                for b in biases:
                    logits = logits + b
                p = jax.nn.softmax(logits, axis=-1)
                o = jnp.einsum("...hqk,...khd->...qhd", p, v)
                return o.reshape(*o.shape[:-2], heads * d) @ Wo
            def stock_call(self, x, mask):                                        # joltz.TriangleAttention.__call__'s body (the reference the chunked schedule must equal)
                import einops
                if not self.starting:
                    x = einops.rearrange(x, "... I J C_in -> ... J I C_in"); mask = einops.rearrange(mask, "... I J -> ... J I")
                x = self.layer_norm(x)
                mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
                triangle_bias = einops.rearrange(self.linear(x), "... J I H -> ... 1 H J I")
                x = self.mha(q_x=x, kv_x=x, biases=[mask_bias, triangle_bias])
                if not self.starting:
                    x = einops.rearrange(x, "... J I C_in -> ... I J C_in")
                return x
            stock_call.__module__ = "joltz"; stock_call.__qualname__ = "TriangleAttention.__call__"   # what install() finds on the stock class
        st = Stub(); st.starting = starting
        return st

    @staticmethod
    def _pairformer_stub(L=8, ds=6, dz=5, seed=0):
        import jax, jax.numpy as jnp, equinox as eqx

        class Block(eqx.Module):
            ws: jax.Array
            wz: jax.Array

            def __call__(self, s, z, mask, pair_mask, *, key, deterministic=False):
                s = s + jnp.tanh(s @ self.ws) * mask[..., None]
                z = z + jnp.tanh(z @ self.wz) * pair_mask[..., None] + 1e-3 * jax.random.normal(key, z.shape)
                return s, z, jax.random.fold_in(key, 1)
        k1, k2 = jax.random.split(jax.random.key(seed))
        stacked = Block(jax.random.normal(k1, (L, ds, ds)) * 0.2, jax.random.normal(k2, (L, dz, dz)) * 0.2)
        params, static = eqx.partition(stacked, eqx.is_array)

        class PF:
            pass
        pf = PF(); pf.stacked_parameters = params; pf.static = static
        return pf

    def _triatt_case(self, starting, lead, n, chunk):
        import jax, jax.numpy as jnp, numpy as np
        st = self._triatt_stub(starting)
        kx, km = jax.random.split(jax.random.key(7))
        x = jax.random.normal(kx, (*lead, n, n, 8)); mask = (jax.random.uniform(km, (*lead, n, n)) > 0.15).astype(jnp.float32)
        self.M.MEM["triatt_chunk"] = None; ref = self.M._triatt_call(st, x, mask)
        self.M.MEM["triatt_chunk"] = chunk; got = self.M._triatt_call(st, x, mask)
        self.assertEqual(got.shape, ref.shape)
        err = float(np.abs(np.asarray(got) - np.asarray(ref)).max())
        self.assertLess(err, 1e-4, f"starting={starting} lead={lead} n={n} chunk={chunk}: max |chunked - stock| = {err}")
        return err

    def test_triatt_chunked_equals_stock_ragged_starting(self):
        self._triatt_case(True, (1,), 10, 4)                                      # 2 full chunks + a remainder of 2 rows

    def test_triatt_chunked_equals_stock_exact_multiple_ending(self):
        self._triatt_case(False, (1,), 12, 4)                                     # ending orientation (the transpose path), no remainder

    def test_triatt_chunked_equals_stock_batch2_chunk1(self):
        self._triatt_case(True, (2,), 7, 1)                                       # a leading batch of 2, one row per chunk

    def test_triatt_single_chunk_is_named(self):
        self._triatt_case(True, (1,), 6, 64)                                      # chunk >= rows: the stock call, recorded as single_chunk
        modes = [t["mode"] for t in self.M.traces() if t["site"] == "triangle_attention"]
        self.assertEqual(modes, ["stock", "single_chunk"])

    def test_unchunked_calls_delegate_to_the_owner_and_chunk_over_a_served_attention_is_refused(self):
        import jax, jax.numpy as jnp
        st = self._triatt_stub(True); x = jax.random.normal(jax.random.key(5), (1, 10, 10, 8)); mask = jnp.ones((1, 10, 10))
        calls = []

        def served(self_, x_, mask_):                                              # another lever's served TriangleAttention call (a fused kernel)
            calls.append(x_.shape); return type(st).stock_call(self_, x_, mask_)
        served.__module__ = "flashattn"; served.__qualname__ = "_triangle_attention_call"
        self.M._ORIG["triatt"] = served
        for C in (None, 16):                                                       # no chunk / a chunk covering every row: the owner's call as is
            self.M.MEM["triatt_chunk"] = C; self.M._triatt_call(st, x, mask)
        self.assertEqual(len(calls), 2)
        self.M.MEM["triatt_chunk"] = 4
        with self.assertRaisesRegex(ValueError, r"memlevers_refused: triatt_chunk=4 over a served TriangleAttention \(flashattn\._triangle_attention_call\)"):
            self.M._triatt_call(st, x, mask)

    def test_describe_values_are_single_tokens(self):
        """The installer's evidence rule (mosaic_opt.levers: every describe() key and value one non-blank token, no containers) over describe()
        in the off state and with a spec configured and executables traced."""
        import re as _re
        M = self.M; pf = self._pairformer_stub(L=8)
        tokre = _re.compile(r"^\S+$")                                             # levers' rule: str(value) has no blank; lists/dicts are refused outright

        def check(d):
            for k, v in d.items():
                self.assertNotIsInstance(v, (list, tuple, dict, set), k); self.assertIsNotNone(v, k)
                self.assertRegex(str(k), tokre); self.assertRegex(str(v), tokre, f"{k}={v!r}")
        check(M.describe())
        M._ORIG["installed"] = True
        try:
            M.configure("tri4+pf4+sub"); self._pf_run(pf, 4); d = M.describe(); check(d)
            self.assertEqual((d["patched"], d["spec"], d["requested"]), (len(M.PATCHED), "tri4+pf4+sub", "tri4+pf4+sub")); self.assertIn("pairformer.grouped=1", d["trace_census"])
        finally:
            M._ORIG.pop("installed", None); M.configure("stock")

    def test_gate_fail_closed_and_lever_line(self):
        """gate(): silent when off; not_configured after install() alone; pf_not_in_force when the configured group never grouped a traced
        stack (G >= L runs single_group); passes once a grouped executable is traced; the LEVER line carries spec, requested and the census."""
        import jax
        M = self.M; pf = self._pairformer_stub(L=8)
        M._ORIG["installed"] = True                                                 # as if install() ran (the stubs stand in for joltz)
        try:
            M.STATE["requested"] = None
            with self.assertRaisesRegex(M.GateRefusal, "memlevers_gate:not_configured"): M.gate()
            M.configure("pf16"); self._pf_run(pf, 16)                               # G >= L: single_group, named — the group never took effect
            with self.assertRaisesRegex(M.GateRefusal, "memlevers_gate:pf_not_in_force"): M.gate()
            M.configure("pf4"); self._pf_run(pf, 4); M.gate()                       # grouped: in force
            line = M.emit_line("t")
            self.assertIn("name=P5.memlevers", line); self.assertIn("state=on", line); self.assertIn("spec=pf4", line); self.assertIn("requested=pf4", line); self.assertIn("pairformer.grouped=1", line)
            M.MEM["pf_group"] = 2                                                    # state edited behind configure()'s back
            with self.assertRaisesRegex(M.GateRefusal, "memlevers_gate:spec_mismatch"): M.gate()
            M.configure("stock"); M.gate(); self.assertIn("state=off", M.emit_line("t"))
        finally:
            M._ORIG.pop("installed", None); M.configure("stock")

    def _pf_run(self, pf, G):
        import jax, jax.numpy as jnp
        ks = jax.random.split(jax.random.key(3), 3)
        s = jax.random.normal(ks[0], (1, 9, 6)); z = jax.random.normal(ks[1], (1, 9, 9, 5))
        mask = jnp.ones((1, 9)); pair_mask = jnp.ones((1, 9, 9)).at[0, 0, 1].set(0.0)
        self.M.MEM["pf_group"] = G
        return self.M._pairformer2_call(pf, s, z, mask, pair_mask, key=ks[2], deterministic=True)

    def test_pairformer_grouped_equals_stock(self):
        import numpy as np
        pf = self._pairformer_stub(L=8)
        s0, z0 = self._pf_run(pf, None)
        for G in (4, 2, 1):
            sG, zG = self._pf_run(pf, G)
            es, ez = float(np.abs(np.asarray(sG) - np.asarray(s0)).max()), float(np.abs(np.asarray(zG) - np.asarray(z0)).max())
            self.assertLess(max(es, ez), 1e-5, f"G={G}: max |grouped - stock| s={es} z={ez}")
        self.assertEqual([t["mode"] for t in self.M.traces()], ["stock", "grouped", "grouped", "grouped"])

    def test_pairformer_grouped_gradient_equals_stock(self):
        import jax, numpy as np
        pf = self._pairformer_stub(L=8)

        def loss(zin, G):
            import jax.numpy as jnp
            ks = jax.random.split(jax.random.key(3), 3)
            s = jax.random.normal(ks[0], (1, 9, 6)); mask = jnp.ones((1, 9)); pair_mask = jnp.ones((1, 9, 9))
            self.M.MEM["pf_group"] = G
            s1, z1 = self.M._pairformer2_call(pf, s, zin, mask, pair_mask, key=ks[2], deterministic=True)
            return (z1 ** 2).sum() + (s1 ** 2).sum()
        z = jax.random.normal(jax.random.key(11), (1, 9, 9, 5))
        g0 = jax.grad(lambda t: loss(t, None))(z); g4 = jax.grad(lambda t: loss(t, 4))(z)
        self.assertLess(float(np.abs(np.asarray(g4) - np.asarray(g0)).max()), 1e-4)

    def test_pairformer_ragged_groups_and_short_stacks_equal_stock_and_are_named(self):
        import numpy as np
        pf = self._pairformer_stub(L=8)
        s0, z0 = self._pf_run(pf, None)
        for G, mode in ((3, "grouped+remainder"), (5, "grouped+remainder"), (8, "single_group"), (16, "single_group"), (4, "grouped")):
            self.M.TRACES.clear()
            sG, zG = self._pf_run(pf, G)                                           # 3: 2 groups of 3 + 2 remaining blocks; 8, 16: one group = the stock scan
            err = max(float(np.abs(np.asarray(sG) - np.asarray(s0)).max()), float(np.abs(np.asarray(zG) - np.asarray(z0)).max()))
            self.assertLess(err, 1e-5, f"G={G}: max |grouped - stock| = {err}")
            self.assertEqual([(t["n"], t["setting"], t["mode"]) for t in self.M.traces()], [(8, G, mode)], f"G={G}")

    def test_sub_remat_wrapper_equals_stock_and_is_named(self):
        import jax, jax.numpy as jnp, numpy as np, equinox as eqx

        class Trans(eqx.Module):                                                   # joltz Transition's shape: arrays + a non-array leaf (the activation callable)
            w1: jax.Array
            w2: jax.Array
            act: callable

            def __call__(self, x, scale=1.0):
                return (self.act(x @ self.w1) @ self.w2) * scale
        k1, k2, kx = jax.random.split(jax.random.key(5), 3)
        t = Trans(jax.random.normal(k1, (5, 20)) * 0.3, jax.random.normal(k2, (20, 5)) * 0.3, jax.nn.silu)
        x = jax.random.normal(kx, (1, 6, 6, 5))
        wrapped = self.M._remat_call("transition", Trans.__call__)
        f = lambda xx: (wrapped(t, xx, scale=2.0) ** 2).sum()
        self.M.MEM["sub_remat"] = False; y0, g0 = f(x), jax.grad(f)(x)
        self.M.MEM["sub_remat"] = True; y1, g1 = jax.jit(f)(x), jax.jit(jax.grad(f))(x)
        self.assertLess(abs(float(y1) - float(y0)), 1e-4); self.assertLess(float(np.abs(np.asarray(g1) - np.asarray(g0)).max()), 1e-4)
        self.assertEqual([(r["site"], r["n"], r["setting"], r["mode"]) for r in self.M.traces()][:2], [("transition", 6, None, "stock"), ("transition", 6, None, "stock")])
        self.assertIn(("transition", 6, "sub", "remat"), [(r["site"], r["n"], r["setting"], r["mode"]) for r in self.M.traces()])
        self.assertEqual(set(self.M.SUB_SITES), {"trimul_out", "trimul_in", "transition", "pair_weighted_averaging", "outer_product_mean"})

    def test_spec_grammar(self):
        P = self.M.parse
        self.assertEqual(P("tri64+pf8"), {"triatt_chunk": 64, "pf_group": 8, "sub_remat": False})
        self.assertEqual(P("pf4"), {"triatt_chunk": None, "pf_group": 4, "sub_remat": False}); self.assertEqual(P("tri32"), {"triatt_chunk": 32, "pf_group": None, "sub_remat": False})
        self.assertEqual(P("stock"), {"triatt_chunk": None, "pf_group": None, "sub_remat": False})
        self.assertEqual(P("tri64+pf8+sub"), {"triatt_chunk": 64, "pf_group": 8, "sub_remat": True}); self.assertEqual(P("sub"), {"triatt_chunk": None, "pf_group": None, "sub_remat": True})
        for bad in ("off", "pf8+off", "tri", "trix", "pf0", "tri-4", "bogus", "tri64,pf8"):
            with self.assertRaisesRegex(ValueError, "memlevers_refused"):
                P(bad)
        self.assertEqual(self.M.configure("tri16+pf2"), {"triatt_chunk": 16, "pf_group": 2, "sub_remat": False}); self.assertEqual(dict(self.M.MEM), {"triatt_chunk": 16, "pf_group": 2, "sub_remat": False})
        self.assertEqual(self.M.configure("stock"), {"triatt_chunk": None, "pf_group": None, "sub_remat": False})
        self.assertEqual(self.M.parse(None), self.M.parse(self.M.SETTING)); self.assertEqual(self.M.spec_of(self.M.configure(None)), self.M.SETTING)   # None = the default setting
        self.assertEqual(self.M.configure(), self.M.parse(self.M.SETTING)); self.M.configure("stock")
        self.assertEqual(self.M.ENV_REQUIRED, {}); d0 = self.M.describe(); self.assertNotIn("name", d0); self.assertEqual((d0["lever_name"], d0["setting"]), ("memlevers", self.M.SETTING))
        self.assertFalse(self.M.installed())                                       # install() patches joltz; nothing here imported it
        self.assertEqual(self.M.spec_of({"triatt_chunk": 64, "pf_group": 8}), "tri64+pf8"); self.assertEqual(self.M.spec_of({"triatt_chunk": None, "pf_group": 4}), "pf4")
        self.assertEqual(self.M.spec_of({"triatt_chunk": 64, "pf_group": 8, "sub_remat": True}), "tri64+pf8+sub")
        self.assertEqual(self.M.spec_of(self.M.parse("stock")), "stock")
        d = self.M.describe()
        self.assertEqual((d["lever"], d["installed"], d["spec"], d["traces"]), ("P5", 0, "stock", 0))
        self.M.uninstall()                                                         # a no-op without install()

    def test_default_setting_is_configure_none(self):
        M = self.M
        self.assertEqual(M.SETTING, "pf8+sub"); self.assertEqual(M.ENV_REQUIRED, {}); self.assertEqual(M.parse(None), M.parse(M.SETTING))


if __name__ == "__main__":
    unittest.main()
