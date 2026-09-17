"""Regression lock: env.sh is the one switch table and modes.resolve() sources it. This test sources the kit's env.sh itself
(ARM=E and ARM=T) in a sandboxed bash, compares the resulting environment variable by variable with what resolve() returns, and
checks that the registry claims exactly the switches env.sh exports for each mode (names; values never live in the package).

The sandbox PATH carries only what env.sh needs (bash, python, coreutils); `nvidia-smi` is absent unless a test adds a stub, and the
`python` the torch probe runs is a shim reporting a chosen version when a test asks for one, so the cc 9.0 branch (FPF_TRIMUL_EXACT_NMAX)
and env.sh's own prebuilt fast-LN selection (L93-104) are exercised explicitly. Precondition failures (no env.sh, no bash) are test
failures, not skips."""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from protenix_opt import modes, registry, stack

from protenix_opt.tests import _registry_views as views

SKIP = {"PATH", "PWD", "SHLVL", "_", "OLDPWD", "FPF_HOME", "ARM", "ENVSH", "OUTA", "OUTB", "PYTHONPATH"}   # shell bookkeeping and env.sh's own inputs
TOOLS = ("bash", "dirname", "head", "tr", "tail", "basename", "env")
KNOB_SETS = [                                                                            # knob matrix (views.KNOBS + pre-set honoured vars)
    {},
    {"DEADSKIP": "0"},
    {"PTX_BLK_GRAPH": "0"},
    {"PTX_SAMPLER_GRAPH": "0"},
    {"PTX_FPF_XL": "0"},
    {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
    {"PTX_T_MIN_TOKENS": "512"},
    {"PTX_BLK_CHUNKED": "0"},
    {"PTX_LEVER_REPORT": "/tmp/x.jsonl", "PTX_BLK_GRAPH_MAX": "4", "INFOPT_GRAPHS_MAX_ENTRIES": "8", "LAYERNORM_TYPE": "torch"},
    {"INFOPT_FASTLN_PREBUILT": "/opt/fastln_prebuilt"},
    {"PTX_BLK_ATT": "stale", "FPF_OPS": "stale=x:y"},
    {"PTX_LAZY_INIT": "0", "PTX_E_PAD8": "1"},
]


def _sandbox_bin(tmp, nvidia_smi_cc=None, torch_version=None):
    b = os.path.join(tmp, "bin")
    os.makedirs(b, exist_ok=True)
    for t in TOOLS:
        p = shutil.which(t)
        assert p is not None, f"{t} not on PATH: env.sh cannot be sourced on this box"
        os.symlink(p, os.path.join(b, t))
    if torch_version is None:
        os.symlink(sys.executable, os.path.join(b, "python"))                          # env.sh L80/L83 probe `python -I`
    else:                                                                              # the torch-version probe (env.sh L80) answers `torch_version`; everything else runs the interpreter
        py = os.path.join(b, "python")
        with open(py, "w") as fh:
            fh.write(f"#!/bin/bash\ncase \"$*\" in *\"m.version('torch')\"*) echo {torch_version}; exit 0;; esac\nexec {sys.executable} \"$@\"\n")
        os.chmod(py, 0o755)
    if nvidia_smi_cc is not None:                                                      # env.sh L72 probes `nvidia-smi --query-gpu=compute_cap`
        smi = os.path.join(b, "nvidia-smi")
        with open(smi, "w") as fh:
            fh.write(f"#!/bin/bash\necho {nvidia_smi_cc}\n")
        os.chmod(smi, 0o755)
    return b


def source_env_sh(fpf_home, arm, presets, bin_dir):
    """Final environment after `ARM=<arm> source env.sh` on top of `presets` (minus shell bookkeeping) — independent of modes.py."""
    script = 'env -0 > "$OUTB"; ARM=%s source "$ENVSH" >/dev/null 2>&1; env -0 > "$OUTA"' % arm
    with tempfile.TemporaryDirectory() as d:
        env = {"PATH": bin_dir, "ENVSH": os.path.join(fpf_home, "env.sh"), "OUTB": os.path.join(d, "b"), "OUTA": os.path.join(d, "a")}
        env.update(presets)
        subprocess.run(["bash", "-c", script], env=env, check=True)
        parse = lambda f: dict(x.split("=", 1) for x in open(f).read().split("\0") if "=" in x)
        before, after = parse(env["OUTB"]), parse(env["OUTA"])
    return {k: v for k, v in before.items() if k not in SKIP}, {k: v for k, v in after.items() if k not in SKIP}


class TestModesMatchEnvSh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fpf = stack.kit_home()
        assert os.path.isfile(os.path.join(cls.fpf, "env.sh")), f"kit env.sh not found at {cls.fpf}"
        cls.tmp = tempfile.mkdtemp()
        cls.bin = _sandbox_bin(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _resolve(self, mode, presets, bin_dir=None, fpf=None, **kw):
        environ = dict(presets); environ["PATH"] = bin_dir or self.bin
        kw.setdefault("triton", "3.3.1"); kw.setdefault("compute_cap", None)
        return modes.resolve(mode, environ, fpf or self.fpf, **kw)

    def _compare(self, mode, presets, bin_dir=None, fpf=None):
        """resolve().final() == the environment env.sh produces, plus the extras resolve() adds after env.sh."""
        fpf = fpf or self.fpf
        r = self._resolve(mode, presets, bin_dir=bin_dir, fpf=fpf)
        pkg = {k: v for k, v in r.pre_exports.items() if k not in presets}       # the package's + the kit README row's pre exports (PTX_T_TRIMUL / PTX_T_ATT / PTX_E_TRIMUL words), as resolve() sets them before env.sh
        bpre = modes.BIG_PRE if mode == "big" else {}                          # big: BIG_PRE rides the row's pre and overrides the base's package cap (PTX_SAMPLER_GRAPH_MAXTOK=1, protenix_opt 0.3.41)
        self.assertEqual({k: v for k, v in modes.PACKAGE_PRE.get(modes.BIG_BASE if mode == "big" else mode, {}).items() if k not in presets and k not in bpre}, {k: v for k, v in pkg.items() if k not in (r.row or {}).get("pre", {}) and k not in bpre}, "pre exports = package pre + row pre")
        before, after = source_env_sh(fpf, modes.ARM[mode], {**presets, **pkg}, bin_dir or self.bin)
        self.assertEqual(before, {**presets, **pkg}, "sandbox leaked variables into the baseline")
        final = {k: v for k, v in r.final(dict(presets)).items() if k not in SKIP}
        expected = dict(after); expected.update(r.extras)
        self.assertEqual(final, expected, f"mode={mode} presets={presets}")
        self.assertEqual(set(r.extras) & (set(after) - set(before)), set(), "extras are never set by env.sh itself")
        return r

    def test_activation_module_proof_holds_under_the_tolerance_arms(self):
        """Under ARM T (fast / big rows of cc 9.0 AND 8.0) the FPF_OPS TriMul provider is the SEALED package
        third_party/fpf_smalln (published bytes: it resolves its callees lazily), whose two TriMul callees env.sh names as
        src/ptx_trimul_routes — the module the activation proof (stack.MODULE_PROOF) requires in sys.modules for lever trimul_core.
        Replays the kit's activation at import level in a clean interpreter with env.sh's real environment (CPU only): the FPF_OPS
        providers imported as the fpf registry does at start-up, then protenix_opt.route_preload.preload() as stack._apply does BEFORE the
        proof; asserts (1) every proof module of the mode's levers has loaded, (2) what fpf_smalln ACTUALLY resolved — its COUNTS words
        exact_fn / fast_fn and the two callables it binds — is ptx_trimul_routes:trimul_exact_c256 / :trimul_c256 (the shared core's provider
        by tier word), never the package's own defaults; and under ARM E (exact rows, both cards) the package is not on the route at all
        (FPF_OPS names ptx_trimul_routes directly, FPF_SMALLN unset, the package inactive if imported)."""
        import json
        from protenix_opt import stack, route_preload
        import opt_core
        core_parent = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))
        pypath = os.pathsep.join([os.path.join(self.fpf, "src"), os.path.join(self.fpf, "third_party"), self.fpf, os.path.dirname(os.path.dirname(os.path.abspath(stack.__file__))), core_parent])
        script = ("import importlib, json, os, sys\n"
                  "for ent in [e for e in os.environ.get('FPF_OPS', '').split(',') if e]:\n"
                  "    importlib.import_module(ent.split('=', 1)[1].split(':')[0].strip())      # what the fpf registry does with FPF_OPS at interpreter start-up\n"
                  "from protenix_opt import route_preload\n"
                  "pre = route_preload.preload()                                                  # what stack._apply does right after the kit's sitecustomize, before MODULE_PROOF is read\n"
                  "mods = json.loads(os.environ['PROOF_MODULES'])\n"
                  "out = {'loaded': {m: (m in sys.modules) for m in mods}, 'preload': pre, 'smalln_loaded': 'fpf_smalln' in sys.modules}\n"
                  "if os.environ.get('IMPORT_SMALLN') == '1' or 'fpf_smalln' in sys.modules:\n"
                  "    import fpf_smalln as S\n"
                  "    out['smalln'] = {'enabled': bool(S.ENABLED), 'exact_fn': S.COUNTS.get('exact_fn'), 'fast_fn': S.COUNTS.get('fast_fn')}\n"
                  "    if S.ENABLED:\n"
                  "        ex, fa = S._resolve(S.EXACT_FN_SPEC), S._resolve(S.FAST_FN_SPEC)             # the callables the package binds at its first TriMul call\n"
                  "        out['smalln'].update(exact_callee=ex.__module__ + ':' + ex.__name__, fast_callee=fa.__module__ + ':' + fa.__name__,\n"
                  "                             exact_is_route=bool(getattr(ex, '_ptx_trimul_routes', False)), fast_is_route=bool(getattr(fa, '_ptx_trimul_routes', False)))\n"
                  "print(json.dumps(out))\n")

        def run(mode, cc, extra=None):
            levers = stack._row_levers_full(mode)
            proofs = {name: stack.MODULE_PROOF[name] for name in levers if name in stack.MODULE_PROOF}
            r = self._resolve(mode, {}, compute_cap=cc, triton="3.7.1")
            env = {k: str(v) for k, v in r.final({}).items()}
            penv = dict(env, PATH=os.environ.get("PATH", ""), PYTHONPATH=pypath, PROTENIX_OPT=mode, PROOF_MODULES=json.dumps(sorted(set(proofs.values()))), HOME=os.environ.get("HOME", "/tmp"), **(extra or {}))
            out = subprocess.run([sys.executable, "-c", script], env=penv, capture_output=True, text=True, timeout=240)
            self.assertEqual(out.returncode, 0, out.stderr[-1500:])
            return env, proofs, json.loads([l for l in out.stdout.splitlines() if l.startswith("{")][-1]), out

        for mode in ("fast", "big"):
            for cc in ("9.0", "8.0"):
                with self.subTest(mode=mode, cc=cc):
                    env, proofs, res, out = run(mode, cc)
                    self.assertIn("trimul_core", proofs, (mode, proofs)); self.assertIn("smalln_size_gate", proofs)
                    self.assertIn("fpf_smalln:trimul_c256", env.get("FPF_OPS", ""), env.get("FPF_OPS"))          # ARM T routes the pair-stack TriMul through the gate package
                    self.assertEqual(route_preload.specs(env), {"FPF_SMALLN_TRIMUL_EXACT_FN": "ptx_trimul_routes:trimul_exact_c256", "FPF_SMALLN_TRIMUL_FAST_FN": "ptx_trimul_routes:trimul_c256"},
                                     f"mode={mode} cc={cc}: env.sh must name BOTH TriMul callees of the gate package (its own defaults are never consulted under the kit)")
                    self.assertTrue(route_preload.enabled(env), env.get("FPF_SMALLN"))
                    self.assertEqual(res["preload"]["errors"], {}, res["preload"]); self.assertEqual(set(res["preload"]["modules"].values()), {"ptx_trimul_routes"})
                    missing = {name: mod for name, mod in proofs.items() if not res["loaded"].get(mod)}
                    self.assertEqual(missing, {}, f"mode={mode} cc={cc}: activation proof modules not loaded before the proof -> the kit would refuse these levers by name; stderr tail: {out.stderr[-600:]}")
                    sm = res.get("smalln") or {}
                    self.assertTrue(res["smalln_loaded"] and sm.get("enabled"), res)
                    self.assertEqual((sm.get("exact_fn"), sm.get("fast_fn")), ("ptx_trimul_routes:trimul_exact_c256", "ptx_trimul_routes:trimul_c256"), f"mode={mode} cc={cc}: what fpf_smalln resolved: {sm}")
                    self.assertEqual((sm.get("exact_callee"), sm.get("fast_callee"), sm.get("exact_is_route"), sm.get("fast_is_route")),
                                     ("ptx_trimul_routes:trimul_exact_c256", "ptx_trimul_routes:trimul_c256", True, True), f"mode={mode} cc={cc}: the callables fpf_smalln binds: {sm}")
        for cc in ("9.0", "8.0"):
            with self.subTest(mode="exact", cc=cc):
                env, proofs, res, out = run("exact", cc, extra={"IMPORT_SMALLN": "1"})
                self.assertIn("trimul_core_exact", proofs)
                ops = dict(e.split("=", 1) for e in env["FPF_OPS"].split(",")[:2])
                self.assertEqual(ops, {"trimul_out": "ptx_trimul_routes:trimul_out", "trimul_in": "ptx_trimul_routes:trimul_in"}, env["FPF_OPS"])   # ARM E: the route module IS the provider; the gate package is not on the route
                self.assertNotIn("fpf_smalln", env["FPF_OPS"]); self.assertFalse(route_preload.enabled(env)); self.assertNotIn("FPF_SMALLN", env)
                self.assertEqual(res["preload"], {"enabled": False, "modules": {}, "errors": {}})
                self.assertTrue(res["loaded"].get("ptx_trimul_routes"), res)
                self.assertEqual((res["smalln"]["enabled"]), False, "imported under ARM E the gate package is inactive: its callee defaults can never serve")

    def test_clean_environment_both_modes(self):
        for mode in ("exact", "fast"):
            with self.subTest(mode=mode):
                r = self._compare(mode, {})
                self.assertIn("PTX_BLK", r.exports); self.assertEqual(r.exports["PTX_LAZY_INIT"], "1")
                self.assertTrue(r.kit_spec.startswith("KIT_SPEC kit="), r.kit_spec[:80])

    def test_registry_claims_exactly_what_env_sh_exports(self):
        """Registry switch names per mode == env.sh's clean-environment export set (names) plus, at most, the probe-conditional keys the
        registry names (INFOPT_FASTLN_PREBUILT when the box's torch matches a shipped prebuilt); extra keys are outside it."""
        for mode in ("exact", "fast"):
            with self.subTest(mode=mode):
                _, after = source_env_sh(self.fpf, modes.ARM[mode], {}, self.bin)
                plain = views.env_sh_keys_for(modes.MODES[mode])
                conditional = (set(after) & views.conditional_keys_for(modes.MODES[mode])) - plain    # a key one lever exports plainly and another names as conditional (PTX_BLK_ATT: k2b's switch, triattn_cuda's selected word) is plain
                self.assertEqual(set(after) - conditional, plain, mode)
                self.assertLessEqual(conditional, {"INFOPT_FASTLN_PREBUILT"}, "no GPU in the sandbox: only the torch probe can add a conditional key")

    def test_knob_matrix_both_modes(self):
        for presets in KNOB_SETS:
            for mode in ("exact", "fast"):
                with self.subTest(mode=mode, presets=presets):
                    r = self._compare(mode, presets)
                    unknown = set(r.exports) - views.env_keys_for(modes.MODES[mode]) - set(views.KNOB_KEYS)
                    self.assertEqual(unknown, set(), f"env.sh exported switches the registry does not name: {unknown}")

    def test_unsets_and_forced_values(self):
        r = self._compare("exact", {"PTX_BLK_ATT": "stale", "FPF_OPS": "stale=x:y"})
        self.assertEqual(sorted(r.unsets), ["PTX_BLK_ATT"])           # env.sh L9; FPF_OPS is re-exported (L66)
        self.assertNotEqual(r.exports["FPF_OPS"], "stale=x:y")
        self.assertNotIn("PTX_DEADSKIP", self._compare("exact", {"DEADSKIP": "0"}).exports)   # env.sh L23 knob honoured
        self.assertEqual(self._compare("exact", {"PTX_LAZY_INIT": "0"}).extras["PTX_LAZY_INIT"], "0")

    # ---- the kit README's per-(cc | triton) table, parsed from the shipped README and locked row by row against resolve()
    @staticmethod
    def _parse_readme_rows(fpf_home):
        """{key: {"exact": {"pre", "post"}, "fast": {...}}} from the README table rows; 'as above' inherits the previous row."""
        text = open(os.path.join(fpf_home, "README.md"), encoding="utf-8").read().splitlines()
        start = next(i for i, l in enumerate(text) if l.startswith("| stack (cc"))
        rows, prev = {}, None
        for l in text[start + 2:]:
            if not l.startswith("|"):
                break
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            # the key cell reads "9.0 \\| 3.3 (...)": the escaped pipe splits it into two cells
            m = re.match(r"(\d+\.\d+)\s*\\\\?$", cells[0]); m2 = re.match(r"(\d+\.\d+|\*)", cells[1]) if m else None
            if not (m and m2):
                rows["other"] = {"exact": {"pre": {}, "post": {}}, "fast": {"pre": {}, "post": {}}}
                continue
            key = f"{m.group(1)}|{m2.group(1)}"
            e_cell, t_cell = cells[2], cells[3]
            def parse(cell, arm, inherit):
                code = re.findall(r"`([^`]*)`", cell)
                pre, post = ({}, {}) if inherit is None else (dict(inherit["pre"]), dict(inherit["post"]))
                if cell.startswith("as above"):
                    for c in code:
                        for kv in c.split():
                            k, v = kv.split("="); post[k] = v
                    return {"pre": pre, "post": post}
                pre, post = {}, {}
                seen_arm = False
                for seg in code[0].split(";"):
                    seg = seg.strip()
                    if seg.startswith("ARM="):
                        seen_arm = True; continue
                    for kv in seg.split():
                        k, v = kv.split("=")
                        (post if seen_arm else pre)[k] = v
                return {"pre": pre, "post": post}
            rows[key] = {"exact": parse(e_cell, "E", prev["exact"] if prev else None), "fast": parse(t_cell, "T", prev["fast"] if prev else None)}
            prev = rows[key]
        return rows

    def test_readme_rows_transcribed_exactly(self):
        """modes.README_ROWS == the table in the shipped kit README (parsed, not typed)."""
        rows = self._parse_readme_rows(self.fpf)
        self.assertEqual(set(rows) - {"other"}, set(modes.README_ROWS), "row keys")
        for key, row in rows.items():
            if key == "other":
                self.assertEqual(modes.OTHER_ROW, row); continue
            self.assertEqual(modes.README_ROWS[key], row, key)
        self.assertEqual(rows["9.0|3.3"]["fast"]["pre"], {"PTX_T_ATT": "fast"}); self.assertEqual(rows["10.0|3.7"]["fast"]["pre"], {})
        self.assertEqual(rows["9.0|3.3"]["exact"]["pre"], {}); self.assertEqual(rows["8.0|3.7"]["exact"]["pre"], {})     # the exact TriMul provider word: cc 9.0 rows only
        self.assertEqual(rows["10.0|3.7"]["exact"]["post"], {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1,F3"})

    def test_every_readme_row_resolves_as_written(self):
        """Per key and mode: pre exports (the row's and the package's PACKAGE_PRE) are set before env.sh (env.sh sees them), post exports
        after; nothing else from the row vocabulary; the resolved environment equals env.sh sourced on top of the pre exports plus the post
        exports and lazy init."""
        rows = self._parse_readme_rows(self.fpf)
        keys = {k for row in rows.values() for arm in row.values() for part in arm.values() for k in part}
        for key, row in rows.items():
            cc, tv = (None, "3.3.1") if key == "other" else (key.split("|")[0], key.split("|")[1] + ".1")
            if key == "other":
                cc = "8.9"                                      # a cc with no row at all (8.0 / 9.0 / 10.0 have broad rows)
            elif key.endswith("|*"):
                tv = "3.5.0"                                    # a triton none of the exact rows lists: the cc's broad row applies
            for mode in ("exact", "fast"):
                with self.subTest(key=key, mode=mode):
                    r = self._resolve(mode, {}, compute_cap=cc, triton=tv)
                    self.assertEqual(r.row_key, key); self.assertEqual(r.pre_exports, {**modes.PACKAGE_PRE[mode], **row[mode]["pre"]})
                    self.assertEqual({k: v for k, v in r.extras.items() if k in keys}, row[mode]["post"])
                    self.assertEqual({k: v for k, v in r.exports.items() if k in keys}, {**row[mode]["pre"], **row[mode]["post"]})
                    presets = {"PATH": self.bin, **modes.PACKAGE_PRE[mode], **row[mode]["pre"]}
                    _, after = source_env_sh(self.fpf, modes.ARM[mode], presets, self.bin)
                    final = {k: v for k, v in r.final({"PATH": self.bin}).items() if k not in SKIP}
                    expected = dict(after); expected.update(r.extras)
                    self.assertEqual(final, expected)

    def test_sampler_graph_token_cap_is_the_packages(self):
        """exact and fast export PTX_SAMPLER_GRAPH_MAXTOK = modes.SAMPLER_GRAPH_MAXTOK before env.sh (env.sh L58 keeps a pre-set value; on a
        device below modes.SAMPLER_GRAPH_MEM_MIB of memory the value is env.sh's own 995: modes.sampler_graph_maxtok); a
        caller's own value wins; big carries the graphed sampler's modules but NOT the CUDA graph: its own pre export caps the graph route below every input
        (BIG_PRE PTX_SAMPLER_GRAPH_MAXTOK=1, protenix_opt 0.3.41: sampler_admit picks hoist_eager / stock per item; a caller's cap wins there too)."""
        for mode in ("exact", "fast"):
            r = self._resolve(mode, {}, compute_cap="9.0", triton="3.7.1")
            self.assertEqual(r.pre_exports["PTX_SAMPLER_GRAPH_MAXTOK"], modes.SAMPLER_GRAPH_MAXTOK); self.assertEqual(r.exports["PTX_SAMPLER_GRAPH_MAXTOK"], modes.SAMPLER_GRAPH_MAXTOK)
            self.assertEqual(r.exports.get("PTX_SAMPLER_GRAPH"), "1")
            c = self._resolve(mode, {"PTX_SAMPLER_GRAPH_MAXTOK": "995"}, compute_cap="9.0", triton="3.7.1")
            self.assertNotIn("PTX_SAMPLER_GRAPH_MAXTOK", c.pre_exports); self.assertNotIn("PTX_SAMPLER_GRAPH_MAXTOK", c.exports)
        self.assertEqual(int(modes.SAMPLER_GRAPH_MAXTOK), 1536)
        for mem, want in ((81559, "1536"), (81920, "1536"), (64 * 1024, "1536"), (40960, "995"), (None, "1536")):   # H100-80GB, A100-80GB, the threshold, A100-40GB, no device probed
            self.assertEqual(modes.sampler_graph_maxtok(mem), want, mem)
            for mode in ("exact", "fast"):
                r = self._resolve(mode, {}, compute_cap="8.0", triton="3.7.1", memory_mib=mem)
                self.assertEqual(r.pre_exports["PTX_SAMPLER_GRAPH_MAXTOK"], want, (mode, mem)); self.assertEqual(r.exports["PTX_SAMPLER_GRAPH_MAXTOK"], want, (mode, mem))
        self.assertEqual(modes.SAMPLER_GRAPH_MAXTOK_SMALL, "995")                       # = env.sh L58's own default
        b = self._resolve("big", {}, compute_cap="9.0", triton="3.7.1")
        self.assertEqual(b.exports.get("PTX_SAMPLER_GRAPH"), "1"); self.assertEqual(b.exports.get("PTX_SAMPLER_GRAPH_MAXTOK"), "1"); self.assertEqual(b.pre_exports.get("PTX_SAMPLER_GRAPH_MAXTOK"), "1")   # the memory row: modules on, graph route capped out
        c = self._resolve("big", {"PTX_SAMPLER_GRAPH_MAXTOK": "1536"}, compute_cap="9.0", triton="3.7.1")                      # a caller's cap wins (restores the graph route under the per-item admission)
        self.assertNotIn("PTX_SAMPLER_GRAPH_MAXTOK", c.pre_exports); self.assertNotIn("PTX_SAMPLER_GRAPH_MAXTOK", c.exports)

    def test_big_is_the_base_holding_no_extra_memory(self):
        """big = the base's row and arm (fast/ARM=T) with BIG_PRE set before
        env.sh (the memory-keyed XL threshold `auto` like the base, no trunk graphs, no sampler CUDA graph: PTX_SAMPLER_GRAPH_MAXTOK=1 caps the graph route out while
        the graphed sampler's modules — hoist, host path, reach — ride the row under the per-item admission of sampler_admit) and the PAD8 post exports not made; env.sh's delta follows: PTX_BLK_GRAPH absent, the sampler words those of
        the big row (PACKAGE_POST: admit=item release=item); the dropped levers are not in the row; a caller's own value wins; PTX_GUARD_LIFT rides the row."""
        for base, benv in (("fast", {}),):                                            # the one base
            for key in ("9.0|3.7", "9.0|3.3", "10.0|3.7"):
                cc, tv = key.split("|")
                b = self._resolve("big", benv, compute_cap=cc, triton=tv); e = self._resolve(base, {}, compute_cap=cc, triton=tv)
                self.assertEqual(b.base, base, (base, key))
                self.assertEqual(b.pre_exports, {**e.pre_exports, **modes.BIG_PRE, **({"PTX_T_ATT": "big"} if e.pre_exports.get("PTX_T_ATT") == "fast" else {})}, (base, key))   # big names its own tier word to the shared core's tri-attention provider where the base binds it (cc 9.0 / 8.0)
                self.assertEqual(b.exports.get("PTX_FPF_CHUNK_TOK"), "auto", key); self.assertEqual(e.exports.get("PTX_FPF_CHUNK_TOK"), "auto", key)
                for k in ("PTX_BLK_GRAPH",):                                                                  # no trunk graphs on the memory row
                    self.assertNotEqual(b.exports.get(k), e.exports.get(k), (key, k)); self.assertIn(e.exports.get(k), ("1",), (key, k)); self.assertNotEqual(b.exports.get(k), "1", (key, k))
                for k in ("PTX_SAMPLER_GRAPH", "PTX_SAMPLER_HOIST", "PTX_SAMPLER_PREP", "PTX_SAMPLER_REACH"):   # the graphed sampler's modules, its host path and its reach ride big (sampler_admit decides per item)
                    self.assertEqual(b.exports.get(k), "1", (key, k)); self.assertEqual(e.exports.get(k), "1", (key, k))
                self.assertEqual(b.exports.get("PTX_SAMPLER_GRAPH_MAXTOK"), "1", key)                          # … but never the CUDA graph itself: the graph route's cap below every input (protenix_opt 0.3.41, BIG_PRE)
                self.assertIn(e.exports.get("PTX_SAMPLER_GRAPH_MAXTOK"), (modes.SAMPLER_GRAPH_MAXTOK, modes.SAMPLER_GRAPH_MAXTOK_SMALL), key)   # the base keeps its own (device-sized) cap
                self.assertEqual({k: b.exports.get(k) for k in modes.PACKAGE_POST["big"]}, modes.PACKAGE_POST["big"], key)
                self.assertEqual((b.exports.get("PTX_SAMPLER_ADMIT"), e.exports.get("PTX_SAMPLER_ADMIT")), ("item", "memory"), key)
                for k in modes.BIG_POST_DROPPED:
                    self.assertNotIn(k, b.extras, (key, k))
                self.assertEqual({k: v for k, v in b.extras.items()}, {**{k: v for k, v in e.extras.items() if k not in modes.BIG_POST_DROPPED}, **modes.BIG_POST, **modes.PACKAGE_POST["big"]}, (base, key))
                differ = ("PTX_BLK_GRAPH", "PTX_BLK_GRAPH_MAX", "PTX_T_ATT", *modes.BIG_PRE, *modes.BIG_POST_DROPPED, *modes.BIG_POST, *modes.PACKAGE_POST["big"])   # PTX_T_ATT: each mode names its own tier word (fast | big) to the shared core's tri-attention provider
                if e.exports.get("PTX_T_ATT") == "fast": self.assertEqual(b.exports.get("PTX_T_ATT"), "big", (base, key))
                same = {k: v for k, v in e.exports.items() if k not in differ}                   # every other export equal on both sides
                self.assertEqual({k: v for k, v in b.exports.items() if k not in differ}, same, (base, key))
                for k in modes.BIG_POST_DROPPED:                                                   # the PAD8 exports: exact's where its row makes them, never big's
                    self.assertNotIn(k, b.exports, (key, k)); self.assertNotIn(k, b.extras, (key, k))
                if key == "9.0|3.7" and base == "exact":
                    for k in modes.BIG_POST_DROPPED: self.assertIn(k, e.exports, (key, k))
                self.assertEqual({k: b.exports.get(k) for k in modes.BIG_POST}, modes.BIG_POST, key)
                self.assertEqual(b.pythonpath, e.pythonpath, key)
                self.assertEqual(b.exports.get("PTX_GUARD_LIFT"), "1")
        self.assertEqual(modes.MODES["big"], modes.big_levers()); self.assertEqual(modes.ARM["big"], "T")   # the fast base
        self.assertEqual(modes.big_levers(), [n for n in modes.MODES["fast"] if n not in modes.BIG_DROPPED] + ["guard_lift"] + list(modes.MEM_LEVERS))
        self.assertEqual(modes.BIG_DROPPED, ("pad8", "stackgraph", "keep_pool")); self.assertEqual(modes.RUNNER_LEVERS, ("guard_lift",))   # pf_attn carried since 0.3.33 (no measured memory cost); dit_fused + dit_lowp carried since 0.3.45 (+0.45 GiB allocated, the row's peak below the stock path's at every measured size)
        self.assertIn("pf_attn", modes.MODES["big"]); self.assertNotIn("PTX_PF_ATTN", modes.BIG_POST)
        for other in ("exact", "fast"):                                                      # never exported by another mode
            o = self._resolve(other, {}, compute_cap="9.0", triton="3.7.1")
            self.assertNotIn("PTX_GUARD_LIFT", o.exports)
        for n in modes.BIG_DROPPED: self.assertNotIn(n, modes.MODES["big"])
        r = self._resolve("big", {"PTX_FPF_CHUNK_TOK": "1536"}, compute_cap="9.0", triton="3.7.1")
        self.assertNotIn("PTX_FPF_CHUNK_TOK", r.pre_exports); self.assertNotIn("PTX_FPF_CHUNK_TOK", r.exports)
        self.assertEqual(r.exports.get("PTX_BLK_GRAPH"), "0"); self.assertEqual(r.exports.get("PTX_SAMPLER_GRAPH"), "1"); self.assertEqual(r.exports.get("PTX_SAMPLER_HOIST"), "1")   # the sampler graph rides big (kit 0.3.29)

    def test_caller_values_win_over_the_row(self):
        r = self._resolve("fast", {"PTX_GLUE_V2": "0"}, compute_cap="9.0", triton="3.3.1")
        self.assertEqual(r.pre_exports, {**modes.PACKAGE_PRE["fast"], "PTX_T_ATT": "fast"}); self.assertNotIn("PTX_T_TRIMUL", r.exports); self.assertNotIn("PTX_GLUE_V2", r.extras)
        self.assertEqual(r.exports["PTX_BLK_ATT"], "ptx_native_core:attn", "the 9.0 row's PTX_T_ATT word is read BY env.sh"); self.assertNotIn("PTX_NATIVE_MIN_TOKENS", r.exports)
        r = self._resolve("fast", {"PTX_T_ATT": "big"}, compute_cap="9.0", triton="3.3.1")
        self.assertEqual(r.exports["PTX_BLK_ATT"], "ptx_native_core:attn", "a caller's PTX_T_ATT=big (the other tier word) reaches the same provider slot"); self.assertNotIn("PTX_NATIVE_MIN_TOKENS", r.exports)
        r = self._resolve("fast", {"PTX_T_ATT": "k2b", **{k: "995" for k in modes.PACKAGE_PRE["fast"]}}, compute_cap="9.0", triton="3.3.1")
        self.assertEqual(r.pre_exports, {}, "a caller's own value for a package / row pre export wins too"); self.assertEqual(r.exports["PTX_BLK_ATT"], "k2b")
        r = self._resolve("fast", {}, compute_cap="9.0", triton="3.3.1", skip_pre=("PTX_T_ATT",))
        self.assertNotIn("PTX_T_ATT", r.pre_exports); self.assertEqual(r.exports["PTX_BLK_ATT"], "k2b", "skip_pre: env.sh sourced without the word -> its own default k2b")
        r = self._resolve("fast", {}, compute_cap="8.0", triton="3.7.1")
        self.assertEqual(r.pre_exports.get("PTX_T_ATT"), "fast"); self.assertEqual(r.exports["PTX_BLK_ATT"], "ptx_native_core:attn", "cc 8.0 binds the provider by the tier word too (protenix_opt 0.3.51)")
        r = self._resolve("exact", {"PTX_MK_PF": "F1"}, compute_cap="9.0", triton="3.3.1")
        self.assertNotIn("PTX_MK_PF", r.exports, "a caller's own switch outside the row is neither overwritten nor removed")

    def _shipped_prebuilts(self):
        """{prebuilt dir: manifest torch} for the kit's own third_party/fastln_prebuilt*/manifest.json (what env.sh L81-85 scans)."""
        out = {}
        for d in sorted(glob.glob(os.path.join(self.fpf, "third_party", "fastln_prebuilt*", ""))):
            out[d.rstrip(os.sep)] = json.load(open(os.path.join(d, "manifest.json")))["torch"]
        return out

    def test_tree_prebuilt_builds(self):
        """The kit ships its prebuilt stream-LN builds in place, each with a self-describing manifest (torch version)."""
        shipped = self._shipped_prebuilts()
        self.assertGreaterEqual(len(shipped), 2, shipped)
        self.assertEqual(len(set(shipped.values())), len(shipped), "one build per torch version")
        for d in shipped:
            self.assertTrue(os.path.isfile(os.path.join(d, "fast_layer_norm_cuda_v2_stream.so")), d)

    def test_fastln_prebuilt_selected_by_env_sh_for_the_installed_torch(self):
        """env.sh's own rule (L93-104) picks the shipped prebuilt whose manifest torch equals the probed torch version, exports nothing
        when none matches, and honours a pre-set INFOPT_FASTLN_PREBUILT; resolve() carries the export as part of env.sh's delta and
        sets nothing of its own."""
        for i, (d, tv) in enumerate(sorted(self._shipped_prebuilts().items())):
            with self.subTest(torch=tv):
                b = _sandbox_bin(os.path.join(self.tmp, f"torch{i}"), torch_version=tv)
                r = self._compare("exact", {}, bin_dir=b)
                self.assertEqual(r.exports["INFOPT_FASTLN_PREBUILT"], d)
                self.assertNotIn("INFOPT_FASTLN_PREBUILT", r.extras); self.assertNotIn("INFOPT_FASTLN_PREBUILT", r.pre_exports)
                r = self._compare("exact", {"INFOPT_FASTLN_PREBUILT": "/preset"}, bin_dir=b)
                self.assertNotIn("INFOPT_FASTLN_PREBUILT", r.exports, "a caller's value is honoured by env.sh (L77)")
        b = _sandbox_bin(os.path.join(self.tmp, "torch_none"), torch_version="2.9.0+cu128")
        self.assertNotIn("INFOPT_FASTLN_PREBUILT", self._compare("exact", {}, bin_dir=b).exports, "no shipped build for this torch: nothing exported")

    def test_pythonpath_entries_and_sys_path(self):
        """The caller's own PYTHONPATH entries sit where env.sh L6 puts them (after the add-on, before the blockfuse add-on), in sys.path too."""
        r = self._resolve("exact", {"PYTHONPATH": "/mine:/theirs"})
        h = self.fpf
        self.assertEqual(r.pythonpath, [os.path.join(h, "src"), os.path.join(h, "third_party"), os.path.join(h, "levers_addon", "PTXV2_LEVERS_ADDON_v1"),
                                        "/mine", "/theirs", os.path.join(h, "third_party", "blockfuse_addon")])
        self.assertEqual(r.pythonpath_caller, ["/mine", "/theirs"])
        self.assertEqual(self._resolve("exact", {}).pythonpath, [e for e in r.pythonpath if e not in ("/mine", "/theirs")])
        sp = stack.kit_sys_path(r.pythonpath, h)
        self.assertEqual(sp, r.pythonpath, "env.sh's entries as sourced")
        self.assertEqual(sp[2:5], [os.path.join(h, "levers_addon", "PTXV2_LEVERS_ADDON_v1"), "/mine", "/theirs"])
        self.assertTrue(os.path.isfile(os.path.join(h, "src", "infopt_graphs", "protenix", "graphed.py")), "the kit's graph machinery under src/, first on the path")
        self.assertTrue(os.path.isdir(os.path.join(h, "src", "ptx_lazy_init")), "the kit's lazy-init package, on the path through $FPF_HOME/src")
        saved = list(sys.path)
        try:
            sys.path[:] = ["", "/mine", "/theirs", "/usr/lib/python3", "/site"]
            stack._install_sys_path(sp)
            self.assertEqual(sys.path, [""] + sp + ["/usr/lib/python3", "/site"], "the caller's entries move to env.sh's position")
        finally:
            sys.path[:] = saved

    def test_unknown_lazy_init_value_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            self._resolve("exact", {"PTX_LAZY_INIT": "yes"})
        self.assertEqual(str(cm.exception), "PTX_LAZY_INIT='yes' is not one of 0|1")
        for v in ("0", "1"):
            self.assertEqual(self._resolve("exact", {"PTX_LAZY_INIT": v}).extras["PTX_LAZY_INIT"], v)
        self.assertEqual(self._resolve("exact", {}).extras["PTX_LAZY_INIT"], modes.LAZY_INIT_DEFAULT)


if __name__ == "__main__":
    unittest.main()
