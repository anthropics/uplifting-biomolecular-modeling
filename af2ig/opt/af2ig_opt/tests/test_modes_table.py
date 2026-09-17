"""The mode table: off | exact | fast | big, the composition read from the kit's own statement, the lever values substituted; the
deterministic recipe."""
import json
import os
import re
import tempfile
import unittest
from unittest import mock

from af2ig_opt import cli, det, modes, registry, stack
from . import _stubs


class TestModes(unittest.TestCase):
    def test_table(self):
        self.assertEqual(modes.MODES, ("off", "exact", "fast", "big"))
        self.assertEqual(modes.DEFAULT_MODE, "fast")                                          # the default is fast; exact is selected by name
        self.assertEqual(set(modes.KIT_MODES), {"exact", "fast", "big"})

    def test_kit_levers_block(self):
        lv = modes.kit_levers(_stubs.KIT)
        ex = modes.parse_exact_stack(lv["exact_default_stack"])
        self.assertEqual(ex["preset"], "-fast")
        self.assertEqual(ex["preset_expands_to"], ["-host_outputs", "-device_params", "-sort_by_length"])
        self.assertEqual(ex["extra"], ["-precompile", "N"])
        self.assertEqual(lv["tier2_opt_in"], ["-subbatch", "-flash_attn", "-fused_triattn", "-fused_trimul", "-opm_reassoc", "-tmpl_pointwise_sub"])
        self.assertEqual(set(lv), {"exact_default_stack", "tier2_opt_in", "tier2_env", "memory_line", "memory_opt_in", "deployment"})
        self.assertEqual((lv["memory_line"], lv["memory_opt_in"]), (["XLA_PYTHON_CLIENT_MEM_FRACTION"], ["-trimul_chunk"]))   # 0.7.1: big composes the pool fraction; the row-chunked TriangleMultiplication is an opt-in
        self.assertEqual(lv["deployment"], ["JAX_COMPILATION_CACHE_DIR"])                     # the ccache lever's one environment name (modes.DEPLOYMENT_SWITCH)

    def test_preset_expansion_matches_registry(self):
        ex = modes.parse_exact_stack(modes.kit_levers(_stubs.KIT)["exact_default_stack"])
        self.assertEqual([registry.LEVERS[l].flag for l in modes.PRESET_LEVERS], ex["preset_expands_to"])
        self.assertTrue(all(registry.LEVERS[l].in_preset for l in modes.PRESET_LEVERS))

    def test_precompile_threads_is_the_mode_table_value(self):
        self.assertEqual(modes.precompile_threads(), (modes.PRECOMPILE_THREADS, "modes.PRECOMPILE_THREADS"))   # N of a bare --precompile is the mode table's; an explicit --precompile N names its own

    def test_config_carries_no_lever_value(self):
        for card in ("h100", "a100"):
            cfg = open(os.path.join(_stubs.TREE, "configs", card + ".env"), encoding="utf-8").read()
            self.assertNotIn("-subbatch", cfg, card)                                                       # no lever value in any config
            self.assertNotIn("PRECOMPILE", cfg, card)                                                     # no thread-count switch in any config: N is the mode table's or the command line's
            self.assertNotIn("AF2IG_OPT=", cfg, card)
            self.assertNotIn("OPT_CORE_PALLAS_ALLOW_FALLBACK_CC", cfg, card)                             # no card runs another card's tile table: the core's own rows or a refusal by name

    def test_a_card_other_than_the_config_is_a_note(self):
        """MODEL_OPT_TARGET_GPU names the card a config targets; another card present is a note on the activation line, never a refusal."""
        a100 = {"name": "NVIDIA A100-SXM4-80GB", "cc": "8.0"}; h100 = {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0"}
        self.assertIsNone(stack.target_gpu_note(a100, {"MODEL_OPT_TARGET_GPU": "A100"}))
        self.assertIsNone(stack.target_gpu_note(h100, {"MODEL_OPT_TARGET_GPU": "H100"}))
        self.assertIsNone(stack.target_gpu_note(a100, {}))
        note = stack.target_gpu_note(h100, {"MODEL_OPT_TARGET_GPU": "A100"})
        self.assertEqual(note, "MODEL_OPT_TARGET_GPU=A100 but the GPU present is NVIDIA H100 80GB HBM3: the kit has no per-GPU table, nothing refuses")

    def test_tiles_gate_reads_the_core_rows_for_the_card(self):
        """fast/big's fused Pallas levers need the core's float32 tile rows for the card's compute capability (opt_core fpf_pallas_serve TILE_TABLES):
        a capability with rows passes with the core's label, one without is refused in the core's words; exact carries no fused lever and is not
        decided here; a card whose capability is unread is left to the driver. Table-driven: a new capability row in the core changes no expectation."""
        from opt_core.kernels import fpf_pallas_serve as S
        f32_ccs = sorted(cc for cc, t in S.TILE_TABLES.items() if "attn_f32_default" in t and "trimul_f32" in t)
        self.assertIn("9.0", f32_ccs)
        for mode in ("fast", "big"):
            res = modes.resolve(mode, _stubs.KIT)
            for cc in sorted(S.TILE_TABLES) + ["0.0"]:
                g = stack.tiles_gate(res, {"cc": cc}, {})
                self.assertEqual((g["checked"], g["cc"]), (["attn", "trimul"], cc), (mode, cc))
                self.assertEqual(g["ok"], cc in f32_ccs, (mode, cc, g["bad"]))
                if cc in f32_ccs:
                    self.assertEqual((g["tiles"], g["bad"]), (f"own:{cc}", []))
                elif cc in S.TILE_TABLES:
                    self.assertRegex(g["bad"][0], rf"^fpf_pallas: no_tiles — no-f32-tiles:{re.escape(cc)} ")
                else:
                    self.assertRegex(g["bad"][0], r"^fpf_pallas: no_tiles — no-tiles:0\.0 \(")
            undecided = stack.tiles_gate(res, {"cc": None}, {})
            self.assertEqual((undecided["ok"], undecided["checked"]), (True, []))
        exact = stack.tiles_gate(modes.resolve("exact", _stubs.KIT), {"cc": "0.0"}, {})
        self.assertEqual((exact["ok"], exact["checked"]), (True, []))
        with mock.patch.dict(os.environ, {S.FALLBACK_ENV: "1"}):                                   # the core's opt-in: another card's table, recorded by the core's label
            fb = stack.tiles_gate(modes.resolve("fast", _stubs.KIT), {"cc": "0.0"}, {})
        self.assertEqual((fb["ok"], fb["tiles"]), (True, f"fallback:0.0->{S.FALLBACK_CC}"))
        for g in (undecided, exact, fb, stack.tiles_gate(res, {"cc": "0.0"}, {})):
            self.assertEqual((g["quiet_ok"], g["forced"]), (True, False))                               # named only when unmet; never forced

    def test_tiles_gate_refuses_a_card_without_rows_by_name(self):
        """On a card the core has no tile rows for, the fused levers of fast / big cannot run: they STEP ASIDE BY NAME (`tiles=aside`,
        `skipped=L10,L11`, `LEVER name=L10|L11 state=skipped reason=cannot_run:…`) and the mode runs its remaining levers — never silent, never a
        refusal (0.6.0; before, the mode refused). With rows (the H100's 9.0) the line carries no tiles word at all — byte for byte the line a check
        without the gate prints. In-process on a box without the stack: the pins/checkout/weights gates are forced (AF2IG_OPT_FORCE=1) so only the
        tiles gate speaks; the tiles gate itself is never forced."""
        from af2ig_opt import cli, report
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            env = dict(env, AF2IG_OPT_FORCE="1")
            refused = lambda rep: rep.get("gates") is None or any(not g["ok"] and not g.get("forced") and not g.get("aside") for g in rep["gates"].values())   # cli.cmd_pred's rule
            for cc, word in (("0.0", "unmet"), ("9.0", None)):
                with mock.patch.object(stack, "gpu_probe", return_value={"name": f"Stub card cc{cc}", "memory": "81920 MiB", "cc": cc, "present": True}), mock.patch.dict(os.environ, env, clear=False):
                    dry = stack.activate("fast", route="cli", dry_run=True, gpu=False, environ=env)
                    run = stack.activate("fast", route="cli", dry_run=False, gpu=False, environ=env)
                line = report.activation_line(dry)
                self.assertEqual((dry["gates"]["tiles"]["quiet_ok"], dry["gates"]["tiles"]["forced"]), (True, False))
                if word == "unmet":                                                                     # 0.6.0 doctrine: the fused levers step aside BY NAME, the mode runs its remaining levers (never a refusal)
                    self.assertIn(" tiles=aside ", line, line); self.assertIn(" skipped=L10,L11,L19 ", line, line)   # 0.7.6: L19 (the bridge core word) steps aside with L10; self.assertNotIn("would_refuse", line)   # 0.7.2: the folded line runs big's producers gate too (tiles=aside producers=ok jax=…)
                    self.assertIn(" line=-fast -precompile 6 -subbatch 128 -flash_attn -opm_reassoc -tmpl_pointwise_sub 8192 -program_cache ", line); self.assertIn(" levers=L6,L1,U1,L7,L9,L8,L12,L18,L13,L15,L16,ccache skipped=L10,L11,L19 precompile=6 ", line)
                    self.assertFalse(refused(run)); self.assertTrue(run["active"], run.get("reason")); self.assertIsNone(run.get("reason"))
                    self.assertEqual(sorted(run["skipped"]), ["L10", "L11", "L19"]); self.assertTrue(all(str(v).startswith("cannot_run:fpf_pallas: no_tiles — no-tiles:0.0 (") for k, v in run["skipped"].items() if k != "L19"), run["skipped"]); self.assertTrue(str(run["skipped"]["L19"]).startswith("cannot_run:needs_L10"))   # 0.7.6: L19 steps aside with L10
                    ll = [l for l in report.lever_lines(run) if " name=L10 " in l or " name=L11 " in l]
                    self.assertEqual(len(ll), 2); self.assertTrue(all(" state=skipped reason=cannot_run:fpf_pallas:_no_tiles" in l for l in ll), ll)
                    self.assertIn("note=", line); self.assertIn("stepped aside by name", line)
                else:                                                                                   # a card with rows (the H100): no tiles word — the line is byte for byte the one rendered without the gate
                    self.assertNotIn("tiles", line, line); self.assertNotIn("would_refuse", line)
                    without = dict(dry, gates={k: v for k, v in dry["gates"].items() if k != "tiles"})
                    self.assertEqual(line, report.activation_line(without))
                    self.assertFalse(refused(run)); self.assertTrue(run["active"], run.get("reason")); self.assertIsNone(run.get("reason"))
            with mock.patch.object(stack, "gpu_probe", return_value={"name": "Stub card cc0.0", "memory": "81920 MiB", "cc": "0.0", "present": True}), mock.patch.dict(os.environ, env, clear=False):
                exact = stack.activate("exact", route="cli", dry_run=True, gpu=False, environ=env)
            self.assertNotIn("tiles", exact["gates"]); self.assertNotIn(" tiles=", report.activation_line(exact))   # exact carries no fused lever: no word

    def test_card_configs_have_one_shape(self):
        """configs/a100.env is configs/h100.env with the target GPU's default changed: the same statements in the same order (comments aside),
        MODEL_OPT_TARGET_GPU defaulting to the card's name."""
        def statements(card):
            text = open(os.path.join(_stubs.TREE, "configs", card + ".env"), encoding="utf-8").read()
            return [l.split("#")[0].rstrip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
        h100, a100 = statements("h100"), statements("a100")
        self.assertEqual(len(h100), len(a100))
        for lh, la in zip(h100, a100):
            if lh.startswith("export MODEL_OPT_TARGET_GPU="):
                self.assertEqual((lh, la), ("export MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-H100}", "export MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-A100}"))
            else:
                self.assertEqual(lh, la)


    def test_resolve(self):
        ex = modes.resolve("exact", _stubs.KIT)
        self.assertEqual(ex.flags, ["-fast", "-precompile", "6", "-program_cache", "DIR", "-prefetch", "2", "-overlap_output", "1"]); self.assertEqual(ex.precompile, 6)   # the default composition: the preset + the program line (L7 AOT precompile on PRECOMPILE_THREADS, L13 program store; DIR is substituted at activation)
        self.assertEqual(ex.levers, ["L6", "L1", "U1", "L7", "L13", "L15", "L16", "ccache"])                                   # every kit mode carries the deployment lever (no argv token)
        fa = modes.resolve("fast", _stubs.KIT)
        self.assertEqual(fa.flags, ["-fast", "-precompile", "6", "-subbatch", "128", "-flash_attn", "-fused_triattn", "-fused_trimul", "-opm_reassoc", "-tmpl_pointwise_sub", "8192", "-program_cache", "DIR", "-prefetch", "2", "-overlap_output", "1"]); self.assertEqual(fa.subbatch, modes.SUBBATCH_ROWS)   # the fast line: no padding
        self.assertEqual(fa.levers, ["L6", "L1", "U1", "L7", "L9", "L8", "L10", "L11", "L12", "L18", "L19", "L13", "L15", "L16", "ccache"]); self.assertIsNone(fa.folded_into)   # 0.7.4: un-folded (fast is its own line again)
        # --precompile [N] (L7 is on in every kit mode): 0 / None = the default N (PRECOMPILE_THREADS), N = that many threads
        exp = modes.resolve("exact", _stubs.KIT, precompile=0)
        self.assertEqual((exp.flags, exp.levers, exp.precompile), (["-fast", "-precompile", "6", "-program_cache", "DIR", "-prefetch", "2", "-overlap_output", "1"], ["L6", "L1", "U1", "L7", "L13", "L15", "L16", "ccache"], 6))
        self.assertEqual(modes.resolve("fast", _stubs.KIT, precompile=3).flags, ["-fast", "-precompile", "3", "-subbatch", "128", "-flash_attn", "-fused_triattn", "-fused_trimul", "-opm_reassoc", "-tmpl_pointwise_sub", "8192", "-program_cache", "DIR", "-prefetch", "2", "-overlap_output", "1"])
        self.assertEqual(modes.resolve("fast", _stubs.KIT, precompile=0).flags, ["-fast", "-precompile", "6", "-subbatch", "128", "-flash_attn", "-fused_triattn", "-fused_trimul", "-opm_reassoc", "-tmpl_pointwise_sub", "8192", "-program_cache", "DIR", "-prefetch", "2", "-overlap_output", "1"])   # 0 = the mode table's N
        self.assertEqual(modes.resolve("exact", _stubs.KIT, precompile=2).flags, ["-fast", "-precompile", "2", "-program_cache", "DIR", "-prefetch", "2", "-overlap_output", "1"])
        self.assertEqual(modes.resolve("big", _stubs.KIT, precompile=2).flags[:3], ["-fast", "-precompile", "2"])   # the memory mode precompiles too (AOT: no forward, no activation memory)
        with self.assertRaises(ValueError):
            modes.resolve("off", _stubs.KIT, precompile=0)
        off = modes.resolve("off", _stubs.KIT)
        self.assertEqual((off.flags, off.levers), ([], []))

    def test_subbatch_rows_is_the_mode_table_value(self):
        self.assertEqual(modes.SUBBATCH_ROWS, 128)
        fa = modes.resolve("fast", _stubs.KIT)
        self.assertEqual(fa.subbatch, 128)   # the composition reads no configuration variable: resolve() takes no environment
        with self.assertRaises(ValueError):
            modes.resolve("turbo", _stubs.KIT)

    def test_levers_off(self):
        """MODEL_OPT_LEVERS_OFF (the tree's uniform ablation switch): the composition minus the named ids — their argv tokens dropped (a lever of the -fast preset
        dropped spells the remaining preset levers by their own flags, in the driver's order), environment levers not placed; ids unknown or outside the mode are
        returned as ignored (named on the activation line), never an error; the stock line has no lever to drop."""
        self.assertEqual(modes.ENV_LEVERS_OFF, "MODEL_OPT_LEVERS_OFF"); self.assertEqual(modes.levers_off_from({}), [])
        self.assertEqual(modes.levers_off_from({"MODEL_OPT_LEVERS_OFF": "L8, ccache,,FOO L8"}), ["L8", "ccache", "FOO"])
        fa = modes.resolve("fast", _stubs.KIT, levers_off=["L8", "FOO", "trimul_chunk"])
        self.assertEqual(fa.flags, ["-fast", "-precompile", "6", "-subbatch", "128", "-fused_triattn", "-fused_trimul", "-opm_reassoc", "-tmpl_pointwise_sub", "8192", "-program_cache", "DIR", "-prefetch", "2", "-overlap_output", "1"]); self.assertEqual(fa.levers, ["L6", "L1", "U1", "L7", "L9", "L10", "L11", "L12", "L18", "L19", "L13", "L15", "L16", "ccache"])
        self.assertEqual((fa.levers_off, fa.levers_off_ignored), (["L8"], ["FOO", "trimul_chunk"])); self.assertIn("minus MODEL_OPT_LEVERS_OFF=L8", fa.source)
        ex = modes.resolve("exact", _stubs.KIT, levers_off=["L1", "ccache"])
        self.assertEqual((ex.flags, ex.levers, ex.levers_off), (["-host_outputs", "-sort_by_length", "-precompile", "6", "-program_cache", "DIR", "-prefetch", "2", "-overlap_output", "1"], ["L6", "U1", "L7", "L13", "L15", "L16"], ["L1", "ccache"]))
        sb = modes.resolve("fast", _stubs.KIT, levers_off=["L9"])
        self.assertEqual((sb.flags, sb.subbatch), (["-fast", "-precompile", "6", "-flash_attn", "-fused_triattn", "-fused_trimul", "-opm_reassoc", "-tmpl_pointwise_sub", "8192", "-program_cache", "DIR", "-prefetch", "2", "-overlap_output", "1"], None))
        bg = modes.resolve("big", _stubs.KIT, levers_off=["mem_fraction", "trimul_chunk"], trimul="256:1473")   # opted in (AF2IG_OPT_TRIMUL_CHUNK) then switched off: both memory levers leave
        self.assertEqual((bg.flags, bg.levers), (["-fast", "-precompile", "6", "-subbatch", "128", "-flash_attn", "-fused_triattn", "-fused_trimul", "-opm_reassoc", "-program_cache", "DIR", "-prefetch", "2", "-overlap_output", "1"], ["L6", "L1", "U1", "L7", "L9", "L8", "L10", "L11", "L12", "L19", "L13", "L15", "L16", "ccache"]))
        pc = modes.resolve("exact", _stubs.KIT, precompile=0, levers_off=["L7", "L13"])
        self.assertEqual((pc.flags, pc.precompile, pc.levers), (["-fast", "-prefetch", "2", "-overlap_output", "1"], 0, ["L6", "L1", "U1", "L15", "L16", "ccache"]))   # the program line off: the lazy in-loop compile of stock, as before 0.6.0
        self.assertEqual(modes.resolve("fast", _stubs.KIT).levers_off, []); self.assertEqual(modes.resolve("off", _stubs.KIT, levers_off=["L8"]).levers, [])
        from af2ig_opt import big
        self.assertEqual(big.child_env(bg.levers), {}); self.assertEqual(big.child_env(modes.resolve("big", _stubs.KIT).levers), {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"}); self.assertEqual(big.child_env(), {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"})



    def test_ccache_recipe_namespace(self):
        """K9 (0.6.0): the numerics recipe the child compiles under is part of the cache namespace — <root>/<stack key>/<recipe>/{jax,programs},
        recipe = det<level>-<8 hex of the child's XLA_FLAGS words> under --det, `default` without XLA flags, `default-<8 hex>` with the caller's own —
        and under --det the child carries JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none (XLA's per-fusion autotune cache neither read nor written):
        entries and autotune results written by a default-numerics process are never read by a det-1 process (which must write the stock line's bytes)."""
        from af2ig_opt import ccache, det
        up = {"jax": "0.5.3", "jaxlib": "0.5.3", "jax-cuda12-plugin": "0.5.3"}; gpu = {"cc": "9.0"}
        self.assertEqual(ccache.recipe_key("", None), "default"); self.assertEqual(ccache.recipe_key(None, 0), "default")
        d1 = ccache.recipe_key(det.env(1, {})["XLA_FLAGS"], 1)
        self.assertRegex(d1, r"^det1-[0-9a-f]{8}$"); self.assertEqual(d1, ccache.recipe_key("  --xla_gpu_autotune_level=0 ", 1))          # whitespace / order do not matter
        self.assertNotEqual(ccache.recipe_key("--xla_gpu_autotune_level=0 --xla_gpu_foo=1", 1), d1)
        self.assertRegex(ccache.recipe_key("--xla_gpu_foo=1", None), r"^default-[0-9a-f]{8}$")
        with tempfile.TemporaryDirectory() as t:
            root = os.path.join(t, "jit")
            st0 = ccache.place(True, {"AF2IG_OPT_JIT_ROOT": root}, up, gpu, os.path.join(t, "pkg"), det_level=None, child_xla_flags="")
            st1 = ccache.place(True, {"AF2IG_OPT_JIT_ROOT": root}, up, gpu, os.path.join(t, "pkg"), det_level=1, child_xla_flags=det.env(1, {})["XLA_FLAGS"])
            key = st0["key"]
            self.assertEqual(st0["dir"], os.path.join(root, key, "default", "jax")); self.assertEqual(st1["dir"], os.path.join(root, key, d1, "jax"))
            self.assertNotIn(ccache.XLA_CACHES_ENV, st0["env"]); self.assertEqual(st1["env"][ccache.XLA_CACHES_ENV], "none")
            self.assertEqual((st0["recipe"], st0["xla_caches"], st1["recipe"], st1["xla_caches"]), ("default", "jax_default", d1, "none"))
            self.assertEqual(ccache.programs_dir(st0, os.path.join(t, "pkg"))[0], os.path.join(root, key, "default", "programs"))
            self.assertEqual(ccache.programs_dir(st1, os.path.join(t, "pkg"))[0], os.path.join(root, key, d1, "programs"))
            self.assertIn(f"recipe={d1} xla_caches=none ", ccache.evidence(st1))
            kept = ccache.place(True, {"JAX_COMPILATION_CACHE_DIR": os.path.join(t, "mine")}, up, gpu, os.path.join(t, "pkg"), det_level=1, child_xla_flags=det.env(1, {})["XLA_FLAGS"])
            self.assertEqual((kept["dir"], kept["env"][ccache.XLA_CACHES_ENV]), (os.path.join(t, "mine"), "none"))                          # a kept directory is the caller's namespace; the fence still holds

class TestCcache(unittest.TestCase):
    """The compile-cache deployment lever (ccache.py): the directory placed in the driver child's environment, by source."""
    UP = {"jax": "0.5.3", "jaxlib": "0.5.3", "jax-cuda12-plugin": "0.5.3"}
    GPU = {"cc": "9.0"}
    KEY = "jax0.5.3-jaxlib0.5.3-cuda12plugin0.5.3-sm90"

    def test_stack_key(self):
        from af2ig_opt import ccache
        self.assertEqual(ccache.stack_key(self.UP, self.GPU), self.KEY)
        self.assertEqual(ccache.stack_key({}, {}), "jaxabsent-jaxlibabsent-cuda12pluginabsent-smnone")      # a box without the stack or a card: named, never an error
        self.assertEqual((ccache.PRESET_ENV, ccache.ROOT_ENV, ccache.COMMON_ROOT_ENV), ("JAX_COMPILATION_CACHE_DIR", "AF2IG_OPT_JIT_ROOT", "MODEL_OPT_JIT_ROOT"))
        self.assertIn("AF2IG_OPT_JIT_ROOT", stack.DECLARED_ENV)                                                # the root is a declared AF2IG_OPT* name (cli.main refuses undeclared ones)

    def test_placement(self):
        from af2ig_opt import ccache
        thresholds = {"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0", "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1"}   # opt_core.capture.xla_cache.PERSISTENT_CACHE_ENV: every compilation stored
        with tempfile.TemporaryDirectory() as tmp:
            default = os.path.join(tmp, "cache", "jit")
            d = ccache.place(True, {}, self.UP, self.GPU, default)                                             # no variable: the package default root
            self.assertEqual((d["state"], d["source"], d["root"], d["dir"], d["entries_before"], d["reason"]), ("on", "default", default, os.path.join(default, self.KEY, "default", "jax"), 0, None))
            self.assertEqual(d["env"], dict({"JAX_COMPILATION_CACHE_DIR": d["dir"]}, **thresholds)); self.assertTrue(os.path.isdir(d["dir"]))
            self.assertEqual(ccache.word(d), "on:" + d["dir"])
            own = ccache.place(True, {"AF2IG_OPT_JIT_ROOT": os.path.join(tmp, "own"), "MODEL_OPT_JIT_ROOT": os.path.join(tmp, "common")}, self.UP, self.GPU, default)
            self.assertEqual((own["state"], own["source"], own["dir"]), ("on", "AF2IG_OPT_JIT_ROOT", os.path.join(tmp, "own", self.KEY, "default", "jax")))
            self.assertEqual(own["note"], f"MODEL_OPT_JIT_ROOT={os.path.join(tmp, 'common')} is set too: AF2IG_OPT_JIT_ROOT={os.path.join(tmp, 'own')} wins (the kit's own word)")
            same = ccache.place(True, {"AF2IG_OPT_JIT_ROOT": os.path.join(tmp, "own"), "MODEL_OPT_JIT_ROOT": os.path.join(tmp, "own")}, self.UP, self.GPU, default)
            self.assertIsNone(same["note"])                                                                     # both words at one path: nothing to note
            com = ccache.place(True, {"MODEL_OPT_JIT_ROOT": os.path.join(tmp, "common")}, self.UP, self.GPU, default)
            self.assertEqual((com["state"], com["source"], com["dir"]), ("on", "MODEL_OPT_JIT_ROOT", os.path.join(tmp, "common", self.KEY, "default", "jax")))
            preset = os.path.join(tmp, "preset")
            kept = ccache.place(True, {"JAX_COMPILATION_CACHE_DIR": preset, "AF2IG_OPT_JIT_ROOT": os.path.join(tmp, "own")}, self.UP, self.GPU, default)
            self.assertEqual((kept["state"], kept["source"], kept["dir"], kept["root"], kept["env"]), ("on", "kept", preset, None, dict({"JAX_COMPILATION_CACHE_DIR": preset}, **thresholds)))
            self.assertEqual(ccache.word(kept), "kept:" + preset); self.assertIn("key=adopted", ccache.evidence(kept))
            off = ccache.place(False, {"AF2IG_OPT_JIT_ROOT": os.path.join(tmp, "own")}, self.UP, self.GPU, default)      # MODEL_OPT_LEVERS_OFF=ccache
            self.assertEqual((off["state"], off["reason"], off["env"], off["dir"]), ("off", "levers_off", {}, None)); self.assertEqual(ccache.word(off), "off:levers_off")
            open(os.path.join(d["dir"], "an-executable"), "w").close()
            self.assertEqual(ccache.evidence(d), f"child_env: JAX_COMPILATION_CACHE_DIR={d['dir']} source=default key={self.KEY} recipe=default xla_caches=jax_default entries=0->1 stored=1")
            ro = os.path.join(tmp, "ro"); os.makedirs(ro); os.chmod(ro, 0o500)
            try:
                if os.access(ro, os.W_OK):
                    self.skipTest("this user writes read-only directories (root): the unwritable-root case is not constructible here")
                sk = ccache.place(True, {"AF2IG_OPT_JIT_ROOT": ro}, self.UP, self.GPU, default)                  # a root that cannot be written: the lever steps aside by name
                self.assertEqual((sk["state"], sk["env"]), ("skipped", {})); self.assertTrue(sk["reason"].startswith("cache_dir_unwritable:" + ro), sk["reason"])
                self.assertEqual(ccache.word(sk), "skipped:" + sk["reason"])
            finally:
                os.chmod(ro, 0o700)


class TestDet(unittest.TestCase):
    def test_env(self):
        self.assertEqual(det.env(1, {}), {"XLA_FLAGS": "--xla_gpu_autotune_level=0"})
        self.assertEqual(det.env(1, {"XLA_FLAGS": "--xla_gpu_foo=1"}), {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_foo=1"})
        with self.assertRaises(ValueError):
            det.env(2, {})

    def test_describe(self):
        self.assertEqual(det.describe(None, {})["on"], False)
        d = det.describe(1, det.env(1, {}))
        self.assertTrue(d["on"] and d["flag_present"] and d["level"] == 1)


if __name__ == "__main__":
    unittest.main()
