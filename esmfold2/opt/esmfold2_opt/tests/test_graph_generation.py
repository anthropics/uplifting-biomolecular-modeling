"""Scope `input` keys the kit's captured-graph generation on the input's shape signature (report.shape_signature: tok, atom, msa, nds): an input of the
live generation's signature REUSES it (nothing released, the graphs replay), an input of another signature releases it FIRST (report.graphs_clear ->
ef2_opt.clear_graphs) and re-captures; one `GRAPHGEN item=<id> reuse=<n> rebuild=<n> shape=<signature>` line per input, decided at the input's first
model call (a forward pre-hook: after featurisation, before the network allocates)."""
import inspect
import re
import sys
import types
import unittest
from unittest import mock

from esmfold2_opt import report, stock_fold

GRAPHGEN_RE = re.compile(r"^\[esmfold2-opt\] GRAPHGEN item=(\S+) reuse=(\d+) rebuild=(\d+) shape=tok:(\d+),atom:(\d+),msa:(\d+),nds:(\d+)$")
GRAPHS_RE = re.compile(r"^\[esmfold2-opt\] GRAPHS item=(\S+) released=(\d+) reason=new_input ")


class Shape:
    def __init__(self, *shape):
        self.shape = tuple(shape)


def call(tok, atom, msa=0, nds=1):
    """The keyword arguments of upstream's model(**features, ...) call, reduced to what the signature reads."""
    kw = {"token_attention_mask": Shape(1, tok), "atom_attention_mask": Shape(1, atom), "num_diffusion_samples": nds, "num_loops": 10}
    kw["msa"] = Shape(1, msa, tok) if msa else None
    return kw


def fake_kit(lru_sampler=8):
    """An ef2_opt stand-in: one graphed trunk module, one graphed sampler; clear_graphs empties both and logs the reason; capture() mimics a fold."""
    log = []
    m = types.ModuleType("ef2_opt")
    m.CFG = types.SimpleNamespace(lru_sampler=lru_sampler)
    trunk = types.SimpleNamespace(_ef2opt_graphs={})
    sampler = types.SimpleNamespace(_ef2opt_step_graphs={})
    m._GRAPHED_MODULES, m._GRAPHED_SAMPLERS = [trunk], [sampler]

    def clear_graphs(model=None, reason="shape"):
        log.append(("clear", reason, dict(report.GENERATION), len(trunk._ef2opt_graphs) + len(sampler._ef2opt_step_graphs)))
        trunk._ef2opt_graphs.clear(); sampler._ef2opt_step_graphs.clear()
    m.clear_graphs = clear_graphs

    def capture(sig):                                                        # what the fold then does inside the kit: trunk graphs per tok, a step graph per (tok, atom, nds)
        trunk._ef2opt_graphs.setdefault(("pair", sig[0]), object())
        sampler._ef2opt_step_graphs.setdefault((sig[0], sig[1], sig[3]), object())
    m.capture = capture
    return m, log


def fresh():
    report.GENERATION.update(sig=None, atoms=[], reuse=0, rebuild=0, armed=None, decided=None)


class Signature(unittest.TestCase):
    def test_signature_reads_token_atom_msa_and_sample_extents(self):
        self.assertEqual(report.shape_signature(call(1400, 11040)), (1400, 11040, 0, 1))
        self.assertEqual(report.shape_signature(call(1400, 11040, msa=1024, nds=5)), (1400, 11040, 1024, 5))
        self.assertEqual(report.shape_signature(call(1400, 11040, msa=7662)), (1400, 11040, 1024, 1))        # subsampled to msa_max_depth rows per recycle (upstream's default 1024)
        self.assertEqual(report.shape_signature(dict(call(1400, 11040, msa=7662), msa_max_depth=512)), (1400, 11040, 512, 1))
        self.assertEqual(report.shape_signature(dict(call(1400, 11040, msa=7662), msa_subsample_at_inference=False)), (1400, 11040, 7662, 1))
        self.assertEqual(report.shape_signature(call(1400, 11040, msa=700)), (1400, 11040, 700, 1))          # shallower than the draw: the encoder sees every row
        self.assertIsNone(report.shape_signature({"num_loops": 10}))                      # not a fold call
        self.assertEqual(report.signature_word((1400, 11040, 1024, 5)), "tok:1400,atom:11040,msa:1024,nds:5")


class Decision(unittest.TestCase):
    def setUp(self):
        fresh()

    def run_items(self, kit, items, scope="input"):
        """items: [(id, kwargs)] -> the lines printed per item (the loop arms, the hook observes, the kit captures)."""
        out = []
        with mock.patch.dict(sys.modules, {"ef2_opt": kit}), mock.patch.dict("os.environ", {report.ENV_CACHE_SCOPE: scope}):
            for item_id, kw in items:
                report.graphgen_arm(item_id)
                lines = report.graphgen_observe(kw)
                kit.capture(report.shape_signature(kw))
                self.assertEqual(report.graphgen_observe(kw), [])                   # the input's later seeds: silent replay, no decision
                out.append(lines)
        return out

    def test_same_shape_inputs_reuse_one_generation(self):
        kit, log = fake_kit()
        lines = self.run_items(kit, [(f"xl1400_{i}", call(1400, 11040)) for i in range(1, 9)])
        self.assertEqual(log, [])                                                    # nothing was ever released
        for i, ls in enumerate(lines, start=1):
            self.assertEqual(len(ls), 1, ls)
            m = GRAPHGEN_RE.match(ls[0])
            self.assertIsNotNone(m, ls[0])
            self.assertEqual((m.group(1), int(m.group(2)), int(m.group(3))), (f"xl1400_{i}", i - 1, 1))   # reuse increments, rebuild stays 1 (the first capture)
        self.assertEqual((report.GENERATION["reuse"], report.GENERATION["rebuild"]), (7, 1))

    def test_a_new_token_count_releases_first_then_rebuilds(self):
        kit, log = fake_kit()
        lines = self.run_items(kit, [("a", call(1400, 11040)), ("b", call(1397, 11008)), ("c", call(1397, 11008))])
        self.assertEqual([l[:2] for l in log], [("clear", "new_input")])
        _, _, gen_at_clear, held_at_clear = log[0]
        self.assertEqual(gen_at_clear["sig"], (1400, 11040, 0, 1))                   # released while the OLD generation was still the record: release first, record after
        self.assertEqual((gen_at_clear["rebuild"], held_at_clear), (1, 2))
        self.assertTrue(GRAPHS_RE.match(lines[1][0]) and "released=2" in lines[1][0], lines[1])
        self.assertTrue(GRAPHGEN_RE.match(lines[1][1]) and "reuse=0 rebuild=2 shape=tok:1397,atom:11008,msa:0,nds:1" in lines[1][1], lines[1])
        self.assertTrue("reuse=1 rebuild=2" in lines[2][0], lines[2])
        n, _ = fake_kit()                                                            # an MSA-row change and a sample-count change are generation changes too
        fresh()
        self.run_items(n, [("p", call(1400, 11040, msa=512)), ("q", call(1400, 11040, msa=900)), ("r", call(1400, 11040, msa=900, nds=5))])
        self.assertEqual((report.GENERATION["reuse"], report.GENERATION["rebuild"]), (0, 3))
        n, _ = fake_kit()                                                            # two deep MSAs (both above msa_max_depth) are ONE per-recycle shape: reuse
        fresh()
        self.run_items(n, [("p", call(1400, 11040, msa=7662)), ("q", call(1400, 11040, msa=9484))])
        self.assertEqual((report.GENERATION["reuse"], report.GENERATION["rebuild"]), (1, 1))

    def test_a_new_atom_count_alone_reuses_the_generation_whatever_the_sampler_budget(self):
        """ef2_opt v4.3: the step-graph sampler's budget is its own SUB-generation (clear_sampler_graphs drops the step graphs and their pool at the
        fold's first graphed step; the trunk graphs stay) — so a new atom count at a live (tok, msa, nds) never releases the generation here
        (v4.2 released it whole when sampler_graphs_held() >= the budget)."""
        kit, log = fake_kit(lru_sampler=3)
        items = [(f"d{i}", call(1400, 11040 + 32 * i)) for i in range(5)]            # equal token count, five distinct atom counts, budget 3
        lines = self.run_items(kit, items)
        words = [ls[-1].split(" reuse=")[1] for ls in lines]
        self.assertEqual(words, ["0 rebuild=1 shape=tok:1400,atom:11040,msa:0,nds:1", "1 rebuild=1 shape=tok:1400,atom:11072,msa:0,nds:1",
                                 "2 rebuild=1 shape=tok:1400,atom:11104,msa:0,nds:1", "3 rebuild=1 shape=tok:1400,atom:11136,msa:0,nds:1",
                                 "4 rebuild=1 shape=tok:1400,atom:11168,msa:0,nds:1"])       # the trunk graphs replay throughout; no release at the input boundary
        self.assertEqual(log, [])                                                    # clear_graphs never ran
        fresh()
        kit1, log1 = fake_kit(lru_sampler=1)                                         # budget 1: still no generation change (the sampler resets its own pool)
        self.run_items(kit1, items[:3])
        self.assertEqual((report.GENERATION["reuse"], report.GENERATION["rebuild"], len(log1)), (2, 1, 0))
        fresh()
        kit2, _ = fake_kit(lru_sampler=2)                                            # a RETURNING atom count is a hit whatever the budget
        self.run_items(kit2, [("x", call(9, 96)), ("y", call(9, 128)), ("x2", call(9, 96)), ("y2", call(9, 128))])
        self.assertEqual((report.GENERATION["reuse"], report.GENERATION["rebuild"]), (3, 1))

    def test_stock_arm_and_global_scope_are_silent(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop("ef2_opt", None)
            report.graphgen_arm("s")
            self.assertEqual(report.graphgen_observe(call(10, 96)), [])              # the stock arm: no kit module, no decision, no line
            self.assertEqual((report.graphs_held(), report.graphs_clear()), (0, 0))
        fresh()
        kit, log = fake_kit()
        self.assertEqual(self.run_items(kit, [("g", call(10, 96)), ("h", call(20, 160))], scope="global"), [[], []])   # scope global: the package leaves the generation to the kit
        self.assertEqual((log, report.GENERATION["rebuild"]), ([], 0))

    def test_install_is_a_forward_pre_hook_with_kwargs(self):
        seen = {}

        class Model:
            def register_forward_pre_hook(self, fn, with_kwargs=False):
                seen.update(fn=fn, with_kwargs=with_kwargs)
                return types.SimpleNamespace(remove=lambda: seen.update(removed=True))
        h = report.graphgen_install(Model())
        self.assertTrue(seen["with_kwargs"] and callable(seen["fn"]))
        self.assertIsNone(seen["fn"](None, (), {"num_loops": 1}))                    # the hook never alters the call
        h.remove(); self.assertTrue(seen["removed"])
        self.assertIsNone(report.graphgen_install(object()))                          # an object without hooks: nothing installed


class LoopOrder(unittest.TestCase):
    def test_the_loop_arms_at_the_input_boundary_and_hooks_the_model_before_the_first_fold(self):
        src = inspect.getsource(stock_fold.fold_items)
        i_install, i_items, i_scope, i_cache, i_arm, i_seeds, i_begin, i_fold, i_remove = (src.index(k) for k in (
            "report.graphgen_install(model)", "for item in items:", 'report.cache_scope() == "input"', "report.cache_clear()", 'report.graphgen_arm(item["id"])',
            "for seed in item_seeds:", "peakmem.begin()", "builder.fold(", "graphgen.remove()"))
        self.assertTrue(i_install < i_items < i_scope < i_cache < i_arm < i_seeds < i_begin < i_fold < i_remove)
        self.assertNotIn("report.graphs_clear()", src)                                # the release is the decision's (release-first on a signature change), never per input


if __name__ == "__main__":
    unittest.main()
