"""The per-fold PHASE line (esmfold2_opt.phases): the four boundaries are wrapped on the model instance at whatever the names resolve to
(upstream's method or a replacement already bound on the instance), a window's line carries one field per phase with the loop's own wall
as total_s, a phase absent on the model prints `-`, a phase rebound past the timer or never called prints `NA` with a PHASE-NOTE (never an
estimate), nested calls of one phase count once, and the wrappers change no argument or result."""
import time
import unittest

from esmfold2_opt import phases


class _Head:
    def __init__(self, tag):
        self.tag = tag

    def sample(self, **kw):
        time.sleep(0.002)
        return ("sampled", self.tag, kw)

    def forward(self, *a, **kw):
        time.sleep(0.001)
        return ("conf", a, kw)


class _Model:
    """The four upstream names of ESMFold2Model the PHASE line times, on a plain object."""
    def __init__(self):
        self.structure_head = _Head("sh")
        self.confidence_head = _Head("ch")

    def _compute_lm_hidden_states(self, ids):
        time.sleep(0.001)
        return ("lm", ids)

    def _run_one_loop(self, z, total_steps=2):
        for _ in range(total_steps):                       # a nested call of the same phase counts once (the outermost)
            if total_steps > 1:
                self._run_one_loop(z, total_steps=1)
        time.sleep(0.002)
        return ("z", z)

    def forward(self, x):
        lm = self._compute_lm_hidden_states(x)
        z = self._run_one_loop(lm)
        s = self.structure_head.sample(z_trunk=z, num_diffusion_samples=2)
        c = self.confidence_head.forward(s)
        return lm, z, s, c


class PhaseLineTest(unittest.TestCase):
    def test_boundaries_wrap_the_instance_and_name_the_callable(self):
        m = _Model()
        bound = phases.install(m)
        self.assertEqual(set(bound), {"lm", "trunk", "sampler", "conf"})
        self.assertTrue(bound["trunk"].endswith("_Model._run_one_loop"), bound)
        self.assertTrue(bound["sampler"].endswith("_Head.sample"), bound)
        line = phases.boundary_line(bound)
        self.assertTrue(line.startswith("PHASE-BOUNDARY lm=model._compute_lm_hidden_states->"), line)
        self.assertEqual(getattr(m._run_one_loop, phases.MARK), "trunk")
        self.assertEqual(phases.install(m), bound)                          # idempotent: no double wrap
        self.assertIsNone(getattr(getattr(m._run_one_loop, "__wrapped__"), phases.MARK, None))

    def test_window_line_fields_and_total_guard(self):
        m = _Model(); phases.install(m)
        self.assertEqual(phases.begin(), [])
        t0 = time.perf_counter(); out = m.forward("x"); wall = time.perf_counter() - t0
        self.assertEqual(out[0], ("lm", "x"))                               # results untouched through the wrappers
        self.assertEqual(out[2][2], {"z_trunk": ("z", ("lm", "x")), "num_diffusion_samples": 2})
        line, notes = phases.line("p1", 7, wall)
        self.assertEqual(notes, [], line)
        self.assertTrue(line.startswith("PHASE item=p1 seed=7 lm_s="), line)
        f = dict(kv.split("=") for kv in line.split()[1:])
        self.assertEqual(list(f), ["item", "seed", "lm_s", "trunk_s", "sampler_s", "conf_s", "total_s"])
        vals = [float(f[k]) for k in ("lm_s", "trunk_s", "sampler_s", "conf_s")]
        self.assertTrue(all(v > 0 for v in vals), line)
        self.assertLessEqual(sum(vals), float(f["total_s"]) + 1e-3, line)
        self.assertEqual(phases._STATE["calls"]["trunk"], 1)                 # the nested loop call counted once
        # a total below the phases is named, not hidden
        _, notes = phases.line("p1", 7, 0.0)
        self.assertTrue(any(n.startswith("PHASE-NOTE phases sum") for n in notes), notes)

    def test_absent_phase_prints_dash_and_object_model_is_all_dashes(self):
        class NoConf(_Model):
            def __init__(self):
                super().__init__(); self.confidence_head = None
        m = NoConf(); bound = phases.install(m)
        self.assertIsNone(bound["conf"])
        phases.begin(); m._compute_lm_hidden_states(1); m._run_one_loop(1); m.structure_head.sample()
        line, notes = phases.line("q", 0, 1.0)
        self.assertIn("conf_s=-", line); self.assertEqual(notes, [])
        o = object()
        self.assertEqual(phases.install(o), {"lm": None, "trunk": None, "sampler": None, "conf": None})
        phases.begin()
        line, notes = phases.line("r", 1, 0.5)
        self.assertEqual(line, "PHASE item=r seed=1 lm_s=- trunk_s=- sampler_s=- conf_s=- total_s=0.500")
        self.assertEqual(notes, [])

    def test_rebound_callable_is_rewrapped_at_begin_and_unobserved_phase_is_NA(self):
        m = _Model(); phases.install(m)
        inner = m.structure_head.sample
        m.structure_head.sample = lambda **kw: inner(**kw)                  # a lever rebinding the sampler after install (outside a window)
        notes = phases.begin()
        self.assertTrue(any(n.startswith("PHASE-NOTE sampler rebound since install") for n in notes), notes)
        self.assertEqual(getattr(m.structure_head.sample, phases.MARK), "sampler")
        m.forward("x")
        line, notes = phases.line("p", 0, 10.0)
        self.assertNotIn("NA", line); self.assertEqual(notes, [])
        # rebound INSIDE the window (past the timer): that phase is NA with a reason, never a number
        phases.begin()
        m.structure_head.sample = lambda **kw: ("bypass", kw)
        m.forward("x")
        line, notes = phases.line("p", 1, 10.0)
        self.assertIn("sampler_s=NA", line)
        self.assertTrue(any(n.startswith("PHASE-NOTE sampler not observed") for n in notes), notes)
        self.assertNotIn("trunk_s=NA", line)

    def test_fold_items_prints_boundary_and_phase_lines(self):
        """stock_fold.fold_items is the one loop of every route: its log carries PHASE-BOUNDARY once and one PHASE line per fold, after the fold line."""
        import inspect
        from esmfold2_opt import stock_fold
        src = inspect.getsource(stock_fold.fold_items)
        self.assertIn("phases.boundary_line(phases.install(model))", src)
        self.assertIn("phases.begin()", src)
        self.assertLess(src.index("phases.begin()"), src.index("t1 = time.perf_counter()"))
        self.assertLess(src.index('log(f"fold {item['"'"'id'"'"']} s{seed} samples={k} {wall:.2f}s")'), src.index("log(phase_line)"))


if __name__ == "__main__":
    unittest.main()
