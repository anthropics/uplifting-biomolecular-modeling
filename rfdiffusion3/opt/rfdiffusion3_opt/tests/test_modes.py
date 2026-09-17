"""The mode table: exact | fast | off over the core's ModeTable; exact = the hoist, the fused transition and the graph sampler (identical
designs), fast = exact + the compile, token-attention and gather-attention levers (forward-pass band; tier 2) and the DEFAULT — exact and off
are selected by name; one variant, the table is the one source of each delta (no README copy), words that are not modes are refused, the mode
argument follows the core; the hoist's five parts are the kit file's constant."""
import dataclasses
import os
import re
import sys
import types
import unittest
from unittest import mock

from .. import modes, registry, stack




class TestModeTable(unittest.TestCase):
    def test_modes_and_default(self):
        self.assertEqual(modes.MODES, ("exact", "fast", "off"))
        self.assertEqual(modes.DEFAULT_MODE, "fast")                             # the release line's uniform rule: fast is the default; exact (identical designs) and off (stock) are selected by name
        self.assertEqual((modes.TABLE.modes, modes.TABLE.default), (modes.MODES, "fast"))
        self.assertEqual(modes.VARIANTS, ("design",))
        self.assertEqual(set(modes.KIT_MODES), {"exact", "fast"})
        self.assertEqual([m.name for m in modes.KIT_MODES.values() if m.default], [modes.DEFAULT_MODE])   # KitMode.default derives from DEFAULT_MODE, the one source

    def test_fast_row(self):
        km = modes.KIT_MODES["fast"]
        self.assertEqual(km.env, {"RFD3_HOIST": "1", "RFD3_FZT": "1", "RFD3_CUDAGRAPH": "1", "RFD3_INIT_CHUNK": "1", "RFD3_COMPILE": "1", "RFD3_TOKEN_SDPA": "1", "RFD3_GATHER_ATTN": "1"})   # the exact delta plus the compile lever, the token-attention lever and the gather-attention lever
        self.assertEqual((km.tier, km.default), ("tolerance", True))
        res = modes.resolve("fast")
        self.assertEqual((res.mode, res.env, res.tier, res.default), ("fast", {"RFD3_HOIST": "1", "RFD3_FZT": "1", "RFD3_CUDAGRAPH": "1", "RFD3_INIT_CHUNK": "1", "RFD3_COMPILE": "1", "RFD3_TOKEN_SDPA": "1", "RFD3_GATHER_ATTN": "1"}, "tolerance", True))
        self.assertEqual(modes.describe_env(res.env), "RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1,RFD3_INIT_CHUNK=1,RFD3_COMPILE=1,RFD3_TOKEN_SDPA=1,RFD3_GATHER_ATTN=1")
        self.assertIn("inference_sampler.py:610-611", res.activation_point)

    def test_mode_argument_follows_the_core(self):
        """--mode when given, else RFDIFFUSION3_OPT, else the default (opt_core.modes.mode_argument); anything that is not a mode (a
        lever subset such as `exact_lowmem` included) raises ModeError."""
        self.assertEqual(modes.mode_of(None, None), "fast")                     # --mode omitted and RFDIFFUSION3_OPT unset: the default
        self.assertEqual(modes.mode_of(None, ""), "fast")
        self.assertEqual(modes.mode_of("exact", None), "exact")                  # identical designs: selected by name
        self.assertEqual(modes.mode_of("fast", "exact"), "fast")                 # the command line wins over the variable
        self.assertEqual(modes.mode_of(None, " Off "), "off")
        with self.assertRaises(modes.ModeError):
            modes.mode_of("exact_lowmem", None)
        with self.assertRaises(modes.ModeError):
            modes.mode_of(None, "turbo")
        with self.assertRaises(modes.ModeError):
            modes.mode_of("turbo", "exact")

    def test_exact_row(self):
        km = modes.KIT_MODES["exact"]
        self.assertEqual(km.env, {"RFD3_HOIST": "1", "RFD3_FZT": "1", "RFD3_CUDAGRAPH": "1", "RFD3_INIT_CHUNK": "1"})   # the hoist, the fused transition, the graph sampler — no core opt-in name: the core's cell table serves the route's cells itself
        self.assertEqual((km.tier, km.default), ("exact", False))
        self.assertEqual([f.name for f in dataclasses.fields(modes.KitMode)], ["name", "env", "tier", "kit_class", "default", "doc"])   # the row is the one source of the delta: no README-line field, nothing to agree with
        self.assertTrue(all("OPT_CORE_PF_ALLOW_CANDIDATE" not in m.env for m in modes.KIT_MODES.values()))   # no mode exports the core's opt-in name (the core's cell table serves the cells itself) ...
        self.assertIn("OPT_CORE_PF_ALLOW_CANDIDATE", stack.lever_names())            # ... while the stock arm still asserts it absent (stock/PINS.json must_be_absent)
        self.assertEqual(km.kit_class, registry.manifest()["class"])
        res = modes.resolve("exact")
        self.assertEqual(res.env, {"RFD3_HOIST": "1", "RFD3_FZT": "1", "RFD3_CUDAGRAPH": "1", "RFD3_INIT_CHUNK": "1"})   # the graph sampler is part of exact: the eager step's kernels, captured and replayed
        self.assertEqual(modes.describe_env(res.env), "RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1,RFD3_INIT_CHUNK=1")
        self.assertIn(f"{modes.KIT_SWITCH} is read at the import of {modes.KIT_HOIST_MODULE}", res.activation_point)   # its hoist.py pointer: test_hoist_py_pointers_land_where_they_say
        self.assertIn(modes.KIT_FZT_MODULE, res.activation_point)
        self.assertEqual(set(res.env), {"RFD3_HOIST", "RFD3_FZT", "RFD3_CUDAGRAPH", "RFD3_INIT_CHUNK"})   # the lever is all five hoist parts: no part selector exists (modes.KIT_PARTS, hoist.py PARTS)

    def test_off_row(self):
        res = modes.resolve("off")
        self.assertEqual(res.env, {})
        self.assertIsNone(res.tier)
        self.assertEqual(modes.describe_env(res.env), "none")

    def test_default_and_normalize(self):
        self.assertEqual(modes.resolve(None).mode, "fast")                       # no mode named: the default
        self.assertEqual(modes.resolve(" EXACT ").mode, "exact")
        self.assertEqual(modes.normalize(""), "fast")

    def test_words_that_are_not_modes_are_refused(self):
        self.assertFalse(hasattr(modes, "CANDIDATES"))                             # no lever composition outside the mode set is named anywhere
        for name in ("exact_lowmem", "turbo", " EXACT_LOWMEM "):
            with self.assertRaises(ValueError) as cm:
                modes.resolve(name)
            self.assertIn("is not a mode (exact|fast|off)", str(cm.exception))       # one sentence (the core's), every route
        with self.assertRaises(ValueError):
            modes.resolve("exact", variant="predict")

    def test_every_mode_switch_is_a_stock_forbidden_name(self):
        forbidden = set(stack.lever_names())
        for km in modes.KIT_MODES.values():
            for k in km.env:
                self.assertIn(k, forbidden)
        self.assertIn("RFDIFFUSION3_OPT", forbidden)

    def test_jit_cache_key_raises_never_silent_unknown(self):
        """jit_cache_key() raises with the cause on every failure mode instead of a bare 'unknown' part — a wrong-but-plausible
        key would silently blend JIT caches across incompatible stacks; the happy-path key shape is unchanged."""
        no_cuda_build = types.SimpleNamespace(__version__="2.13.0+cpu", version=types.SimpleNamespace(cuda=None))
        with mock.patch.dict(sys.modules, {"torch": no_cuda_build}):
            with self.assertRaises(RuntimeError) as cm:
                modes.jit_cache_key()
            self.assertIn("no CUDA build", str(cm.exception))

        def _raise_no_gpu(_idx):
            raise RuntimeError("no CUDA GPUs are available")
        no_gpu = types.SimpleNamespace(__version__="2.13.0+cu130", version=types.SimpleNamespace(cuda="13.0"),
                                        cuda=types.SimpleNamespace(get_device_capability=_raise_no_gpu))
        with mock.patch.dict(sys.modules, {"torch": no_gpu}):
            with self.assertRaises(RuntimeError) as cm:
                modes.jit_cache_key()
            self.assertIn("no CUDA GPUs are available", str(cm.exception))

        with mock.patch.dict(sys.modules, {"torch": None}):                        # sys.modules[name] = None: import raises ImportError
            with self.assertRaises(ImportError):
                modes.jit_cache_key()

        happy = types.SimpleNamespace(__version__="2.13.0+cu130", version=types.SimpleNamespace(cuda="13.0"),
                                       cuda=types.SimpleNamespace(get_device_capability=lambda _idx: (9, 0)))
        with mock.patch.dict(sys.modules, {"torch": happy}):
            self.assertEqual(modes.jit_cache_key(), "torch2.13.0-cu130-sm90")

    def test_kit_parts_match_hoist_py(self):
        src = open(f"{registry.kit_home()}/patched/rfd3/model/hoist.py", encoding="utf-8").read().splitlines()
        parts = [l for l in src if l.startswith("PARTS = ")]
        self.assertEqual(len(parts), 1)
        self.assertNotIn("environ", parts[0])                                        # a constant of the kit file: no part selector
        self.assertIn(", ".join('"%s"' % p for p in modes.KIT_PARTS), parts[0])       # the kit file's five parts, in the kit's order
        self.assertEqual(",".join(modes.KIT_PARTS), "pll,downcast,valid,where,dedup")

    def test_hoist_py_pointers_land_where_they_say(self):
        """Every `hoist.py:<n>` source pointer the package cites (docstrings, comments, the late-activation refusal, ACTIVATION_POINT) lands
        on a line of the carried kit file that reads a hoist switch or is its describe() — a renumbered kit file moves the pointer with it."""
        src = open(f"{registry.kit_home()}/patched/rfd3/model/hoist.py", encoding="utf-8").read().splitlines()
        pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cited = {}
        for name in sorted(os.listdir(pkg)):
            if name.endswith(".py"):
                for n in re.findall(r"hoist\.py:(\d+)", open(os.path.join(pkg, name), encoding="utf-8").read()):
                    cited.setdefault(int(n), set()).add(name)
        for n, files in sorted(cited.items()):
            line = src[n - 1] if 0 < n <= len(src) else ""
            self.assertTrue("RFD3_HOIST" in line or line.startswith("def describe("), f"hoist.py:{n} cited in {sorted(files)} is {line.strip()[:60]!r}")


class TestCommandLineModeList(unittest.TestCase):
    """`--mode` lists exactly the table (exact|fast|off) on design, check and warm — no candidate, no lever name; a documented
    candidate is refused by argparse BY NAME with the table's pointer (exit 2, before any verb body runs); any other unknown value is
    argparse's own invalid choice."""

    def _mode_action(self, parser):
        return next(a for a in parser._actions if a.dest == "mode")

    def _subparsers(self):
        import argparse
        from .. import cli
        ap = cli.build_parser()
        return ap, next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction)).choices

    def test_mode_choices_are_the_table(self):
        ap, subs = self._subparsers()
        for verb in ("design", "check", "warm"):
            act = self._mode_action(subs[verb])
            self.assertEqual(tuple(act.choices), modes.MODES, verb)
            self.assertEqual(act.metavar, "|".join(modes.MODES), verb)
            self.assertIsNone(act.default, verb)                                    # RFDIFFUSION3_OPT, else modes.DEFAULT_MODE, decide later
            self.assertNotIn("lowmem", subs[verb].format_help(), verb)

    def test_a_word_that_is_not_a_mode_is_refused_at_the_parser(self):
        import io
        ap, _ = self._subparsers()
        lines = {"design": [], "check": [], "warm": []}                              # design's upstream tokens never reach argparse (cli.split_design_line); the kit flags do
        for verb, rest in lines.items():
            for name in ("exact_lowmem",):                                          # a lever subset: argparse's choices, like any unknown word
                err = io.StringIO()
                with mock.patch("sys.stderr", err), self.assertRaises(SystemExit) as cm:
                    ap.parse_args([verb, "--mode", name] + rest)
                self.assertEqual(cm.exception.code, 2, verb)
                self.assertIn(f"invalid choice: '{name}'", err.getvalue(), verb)
            err = io.StringIO()
            with mock.patch("sys.stderr", err), self.assertRaises(SystemExit) as cm:
                ap.parse_args([verb, "--mode", "turbo"] + rest)
            self.assertEqual(cm.exception.code, 2, verb)
            self.assertIn("invalid choice: 'turbo'", err.getvalue(), verb)
            self.assertEqual(ap.parse_args([verb, "--mode", "fast"] + rest).mode, "fast", verb)


if __name__ == "__main__":
    unittest.main()
