"""The headroom guard of the captured-graph generation (report.headroom_*): an input of the live generation's own signature whose measured eager
prologue does not fit in the memory the generation's pools / static buffers leave free is decided `headroom` — the generation is released FIRST
(ef2_opt.clear_graphs reason=headroom), ef2_opt's per-shape graph budget is lowered to tok-1 (this size and larger run the graph levers eagerly for the
rest of the process), one `GRAPHS item=<id> released=<n> reason=headroom need_gib=… free_gib=… budget_tokens=<tok-1>` line before the GRAPHGEN line.
It never fires while the prologue fits, and is inert without CUDA statistics."""
import re
import sys
import types
import unittest
from unittest import mock

from esmfold2_opt import report

GIB = 2 ** 30
GRAPHGEN_RE = re.compile(r"^\[esmfold2-opt\] GRAPHGEN item=(\S+) reuse=(\d+) rebuild=(\d+) shape=tok:(\d+),atom:(\d+),msa:(\d+),nds:(\d+)$")
HEADROOM_RE = re.compile(r"^\[esmfold2-opt\] GRAPHS item=(\S+) released=(\d+) reason=headroom need_gib=([\d.]+) free_gib=([\d.]+) budget_tokens=(\d+) ")


class Shape:
    def __init__(self, *shape):
        self.shape = tuple(shape)


def call(tok, atom, msa=0, nds=1):
    kw = {"token_attention_mask": Shape(1, tok), "atom_attention_mask": Shape(1, atom), "num_diffusion_samples": nds, "num_loops": 10}
    kw["msa"] = Shape(1, msa, tok) if msa else None
    return kw


def fake_kit(budget=0):
    log = []
    m = types.ModuleType("ef2_opt")
    m.CFG = types.SimpleNamespace(lru_sampler=8, graph_budget_tokens=budget)
    trunk = types.SimpleNamespace(_ef2opt_graphs={})
    sampler = types.SimpleNamespace(_ef2opt_step_graphs={})
    m._GRAPHED_MODULES, m._GRAPHED_SAMPLERS = [trunk], [sampler]

    def clear_graphs(model=None, reason="shape"):
        log.append(("clear", reason, len(trunk._ef2opt_graphs) + len(sampler._ef2opt_step_graphs)))
        trunk._ef2opt_graphs.clear(); sampler._ef2opt_step_graphs.clear()
    m.clear_graphs = clear_graphs

    def capture(sig):                                                       # what the fold does inside the kit — unless the budget makes it eager
        if m.CFG.graph_budget_tokens and sig[0] > m.CFG.graph_budget_tokens:
            return
        trunk._ef2opt_graphs.setdefault(("pair", sig[0]), object())
        sampler._ef2opt_step_graphs.setdefault((sig[0], sig[1], sig[3]), object())
    m.capture = capture
    return m, log


class Mem:
    """Injected CUDA memory statistics: free / total / allocated / running peak (bytes)."""
    def __init__(self, total=80 * GIB):
        self.total = total; self.free = total; self.alloc = 0; self.peak = 0

    def __call__(self):
        return self.free, self.total, self.alloc, self.peak


def fresh():
    report.GENERATION.update(sig=None, atoms=[], reuse=0, rebuild=0, armed=None, decided=None, base_alloc=None, prologue_seen=False,
                             prologue_need={}, headroom_events=[], headroom_budget=None)


class Headroom(unittest.TestCase):
    def setUp(self):
        fresh()
        self.mem = Mem()

    def fold(self, kit, item_id, kw, prologue_gib, resident_after_gib):
        """One fold as the loop + hooks see it: arm, the model call (decision lines), the prologue's peak, the first graphed region (probe),
        the kit's captures, and the memory the fold leaves resident."""
        report.graphgen_arm(item_id)
        self.mem.peak = self.mem.alloc                                          # peakmem.begin(): the fold window's peak starts at the resident set
        lines = report.graphgen_observe(kw)
        self.mem.peak = self.mem.alloc + int(prologue_gib * GIB)                # the eager prologue ran
        report.headroom_probe()                                                 # lm_encoder / folding_trunk pre-hook
        kit.capture(report.shape_signature(kw))
        self.mem.alloc = int(resident_after_gib * GIB); self.mem.free = self.mem.total - self.mem.alloc
        return lines

    def test_second_large_fold_releases_the_generation_and_goes_eager_for_that_size(self):
        kit, log = fake_kit()
        with mock.patch.dict(sys.modules, {"ef2_opt": kit}), mock.patch.dict("os.environ", {report.ENV_CACHE_SCOPE: "input"}), \
                mock.patch.object(report, "_cuda_mem", self.mem):
            self.mem.alloc = int(12.4 * GIB); self.mem.free = self.mem.total - self.mem.alloc            # the LM resident
            l1 = self.fold(kit, "7B9C", call(1945, 15360), prologue_gib=35.0, resident_after_gib=43.5)   # cold: captures, leaves 43.5 GiB resident
            self.assertEqual(len(l1), 1); self.assertTrue(GRAPHGEN_RE.match(l1[0]) and "rebuild=1" in l1[0], l1)
            self.assertEqual(report.GENERATION["prologue_need"][(1945, 0, 1)], int(35.0 * GIB))
            l2 = self.fold(kit, "7B9C_b", call(1945, 15360), prologue_gib=35.0, resident_after_gib=12.4)  # same signature: 43.5 + 35 > 80 -> headroom
            self.assertEqual(len(l2), 2, l2)
            m = HEADROOM_RE.match(l2[0]); self.assertTrue(m, l2[0])
            self.assertEqual((m.group(1), m.group(2), m.group(5)), ("7B9C_b", "2", "1944"))
            self.assertEqual(m.group(3), "35.00"); self.assertEqual(m.group(4), "36.50")
            self.assertTrue(GRAPHGEN_RE.match(l2[1]) and "reuse=0 rebuild=2" in l2[1], l2[1])
            self.assertEqual(log, [("clear", "headroom", 2)])
            self.assertEqual(kit.CFG.graph_budget_tokens, 1944)
            self.assertEqual(report.GENERATION["headroom_budget"], 1944)
            l3 = self.fold(kit, "7B9C_c", call(1945, 15360), prologue_gib=35.0, resident_after_gib=12.4)  # eager now: nothing held, nothing to release
            self.assertEqual(len(l3), 1); self.assertIn("reuse=1 rebuild=2", l3[0])
            l4 = self.fold(kit, "small", call(400, 3200), prologue_gib=1.5, resident_after_gib=14.0)      # a smaller input still captures
            self.assertEqual(len(l4), 1); self.assertIn("rebuild=3", l4[0])
            self.assertEqual(len(kit._GRAPHED_MODULES[0]._ef2opt_graphs), 1)
            self.assertEqual(len(log), 1)                                                                 # the small input released nothing (nothing was held)

    def test_reuse_while_the_prologue_fits(self):
        kit, log = fake_kit()
        with mock.patch.dict(sys.modules, {"ef2_opt": kit}), mock.patch.dict("os.environ", {report.ENV_CACHE_SCOPE: "input"}), \
                mock.patch.object(report, "_cuda_mem", self.mem):
            self.mem.alloc = int(12.4 * GIB); self.mem.free = self.mem.total - self.mem.alloc
            self.fold(kit, "a", call(1400, 11040), prologue_gib=18.0, resident_after_gib=28.5)
            for i in range(3):
                lines = self.fold(kit, f"b{i}", call(1400, 11040), prologue_gib=18.0, resident_after_gib=28.5)
                self.assertEqual(len(lines), 1); self.assertIn(f"reuse={i + 1} rebuild=1", lines[0])
            self.assertEqual(log, []); self.assertEqual(kit.CFG.graph_budget_tokens, 0)
            self.assertEqual(report.GENERATION["headroom_events"], [])

    def test_an_existing_lower_budget_is_kept_and_a_higher_one_lowered(self):
        kit, log = fake_kit(budget=2500)
        with mock.patch.dict(sys.modules, {"ef2_opt": kit}), mock.patch.object(report, "_cuda_mem", self.mem):
            report.GENERATION["sig"] = (1945, 15360, 0, 1)
            report.headroom_apply((1945, 15360, 0, 1), {"need": 35 * GIB, "free": 36 * GIB, "total": 80 * GIB, "margin": 3 * GIB})
            self.assertEqual(kit.CFG.graph_budget_tokens, 1944)
            report.headroom_apply((2894, 23000, 0, 1), {"need": 60 * GIB, "free": 30 * GIB, "total": 80 * GIB, "margin": 3 * GIB})
            self.assertEqual(kit.CFG.graph_budget_tokens, 1944)                                           # never raised
            self.assertEqual(report.GENERATION["headroom_budget"], 1944)

    def test_inert_without_cuda_statistics(self):
        kit, log = fake_kit()
        with mock.patch.dict(sys.modules, {"ef2_opt": kit}), mock.patch.dict("os.environ", {report.ENV_CACHE_SCOPE: "input"}), \
                mock.patch.object(report, "_cuda_mem", lambda: None):
            for i, item in enumerate(("x", "y", "z")):
                report.graphgen_arm(item)
                lines = report.graphgen_observe(call(1945, 15360))
                report.headroom_probe()
                kit.capture((1945, 15360, 0, 1))
                self.assertEqual(len(lines), 1)
            self.assertEqual(log, []); self.assertEqual(report.GENERATION["prologue_need"], {})


if __name__ == "__main__":
    unittest.main()
