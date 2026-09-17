"""The sys.path the package installs resolves every kit module the activation sequence imports to the in-tree file env.sh's
PYTHONPATH order picks (the FlashPairformer kit's own copies first, its vendored _fallback after the caller's entries); the levers
add-on's module and the one sitecustomize are served by env.sh's entries — one copy each in the tree. Specs only — nothing is executed."""
import importlib.util
import json
import os
import subprocess
import sys
import unittest

from protenix_opt import stack

FPF = "forward/flashpairformer"
# module -> path relative to protenix_v2/opt that env.sh L6 (kit_sys_path) resolves it to
EXPECTED = {
    "sitecustomize": f"{FPF}/src/sitecustomize.py",                                      # env.sh L6: src FIRST — the one sitecustomize python imports
    "ptx_lazy_init": f"{FPF}/src/ptx_lazy_init/__init__.py",                             # sitecustomize L15: the kit's package, via $FPF_HOME/src
    "ptx_trunk2_levers": f"{FPF}/src/ptx_trunk2_levers.py",                              # sitecustomize L54
    "ptx_fpf_v02": f"{FPF}/src/ptx_fpf_v02.py",                                          # sitecustomize L57
    "fpf": f"{FPF}/src/fpf/__init__.py",                                                 # sitecustomize L76
    "deadskip": f"{FPF}/src/deadskip.py",                                                # ptx_trunk2_levers L1511
    "ptx_addon_levers": f"{FPF}/levers_addon/PTXV2_LEVERS_ADDON_v1/ptx_addon_levers.py",             # sitecustomize L105: the levers add-on
    "fpf_clisampler": f"{FPF}/src/fpf_clisampler/__init__.py",                           # sitecustomize L153
    "ptx_trimul_routes": f"{FPF}/src/ptx_trimul_routes.py",                             # env.sh FPF_OPS (ARM E) / fpf_smalln's TriMul callees (ARM T)
    "fpf_smalln": f"{FPF}/third_party/fpf_smalln/__init__.py",                           # env.sh L21 FPF_OPS
    "fpf_stackgraph": f"{FPF}/src/fpf_stackgraph/__init__.py",                           # ptx_fpf_v02 L304
    "ptx_msa_adapt": f"{FPF}/src/ptx_msa_adapt/__init__.py",                             # ptx_trunk2_levers L1434
    "infopt_graphs": f"{FPF}/src/infopt_graphs/__init__.py",                             # env.sh L6: src (HAZARD44-guarded graphed.py)
    "dit_hoist": f"{FPF}/src/dit_hoist.py",                                         # fpf_clisampler/clisampler.py L17 (PTX_SAMPLER_HOIST)
    "fastln_prebuilt": f"{FPF}/third_party/fastln_prebuilt.py",                          # env.sh L6: third_party (the loader; the builds are fastln_prebuilt*/ beside it)
}
ADDON_SERVED = ("ptx_addon_levers", "sitecustomize")         # served by env.sh's entries (the add-on / the kit's src)

PROBE = """
import importlib.util, json, os, sys
sys.path.insert(0, sys.argv[1])
from protenix_opt import stack
stack._install_sys_path(stack.kit_sys_path())
out = {}
for m in json.loads(sys.argv[2]):
    s = importlib.util.find_spec(m)
    out[m] = None if s is None else os.path.realpath(s.origin)
print(json.dumps(out))
"""


class TestKitPaths(unittest.TestCase):
    def test_every_kit_module_resolves_in_tree(self):
        r = subprocess.run([sys.executable, "-S", "-c", PROBE, stack.opt_home(), json.dumps(sorted(EXPECTED))], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        got = json.loads(r.stdout.strip().splitlines()[-1])
        for m, rel in EXPECTED.items():
            self.assertEqual(got[m], os.path.realpath(os.path.join(stack.opt_home(), rel)), m)

    def test_sys_path_order_matches_env_sh(self):
        sp = stack.kit_sys_path()
        h = stack.kit_home()
        self.assertEqual(sp, [os.path.join(h, "src"), os.path.join(h, "third_party"), os.path.join(h, "levers_addon", "PTXV2_LEVERS_ADDON_v1"),
                              os.path.join(h, "third_party", "blockfuse_addon")])
        self.assertEqual(sp, stack.env_sh_pythonpath(), "env.sh's entries, in its order")
        for mode in ("exact", "fast"):
            self.assertEqual(stack.kit_sys_path(mode=mode), sp, mode)
        self.assertEqual(stack.kit_sys_path(mode="off"), [])
        caller = ["/caller/one", "/caller/two"]
        with_caller = stack.kit_sys_path(sp[:3] + caller + sp[3:])
        self.assertEqual(with_caller, sp[:3] + caller + sp[3:], "the caller's entries stay where env.sh put them")
        for d in ("fastln_prebuilt", "fastln_prebuilt_cu130"):
            self.assertTrue(os.path.isdir(os.path.join(h, "third_party", d)), f"env.sh L75-86 lists third_party/{d}: carried in place")
        self.assertTrue(os.path.isdir(os.path.join(h, "src", "infopt_graphs")), "env.sh L6: the kit's graph machinery under src/")
        self.assertTrue(os.path.isdir(os.path.join(h, "src", "ptx_lazy_init")), "sitecustomize imports ptx_lazy_init: the kit's package under src/")

    def test_addon_modules_served_once(self):
        """The levers add-on's module and the one sitecustomize are served by env.sh's entries (the add-on / the kit's src) — one copy each in the tree."""
        sp = stack.kit_sys_path()
        for m in ADDON_SERVED:
            serving = [e for e in sp if os.path.isfile(os.path.join(e, m + ".py"))]
            self.assertEqual(len(serving), 1, f"{m}: exactly one env.sh entry serves it ({serving})")
        fpf_sc = open(os.path.join(stack.kit_home(), "src", "sitecustomize.py"), encoding="utf-8").read()
        for gate in ("PTX_TEMPL_DEDUPE", "PTX_DET"):
            self.assertIn(gate, fpf_sc, f"the FlashPairformer sitecustomize carries the add-on's {gate} import")

    def test_graphed_py_resolved(self):
        """The infopt_graphs copy the CLI route imports is the kit's HAZARD44-guarded graphed.py under src/."""
        sp = stack.kit_sys_path()
        rel = os.path.join(FPF, "src", "infopt_graphs", "protenix", "graphed.py")
        self.assertEqual(stack.graphed_py(sp), os.path.join(stack.opt_home(), rel))
        self.assertEqual(len(stack.graphed_py_sha256(sp)), 64)
        self.assertIsNone(stack.graphed_py(["/nonexistent"]))

    def test_home_override(self):
        env = dict(os.environ, PROTENIX_OPT_HOME=stack.opt_home())
        r = subprocess.run([sys.executable, "-S", "-c", "import sys; sys.path.insert(0, sys.argv[1]); from protenix_opt import stack; print(stack.opt_home())",
                            stack.opt_home()], env=env, capture_output=True, text=True, check=True)
        self.assertEqual(os.path.realpath(r.stdout.strip()), os.path.realpath(stack.opt_home()))
        env = dict(os.environ, PROTENIX_OPT_HOME="/nonexistent"); env.pop("MODEL_OPT", None)
        r = subprocess.run([sys.executable, "-S", "-c", "import sys; sys.path.insert(0, sys.argv[1]); from protenix_opt import stack; print(stack.opt_home())",
                            stack.opt_home()], env=env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0, "an override that names no kit is a named error, never a silent fallback to another tree")
        self.assertIn("PROTENIX_OPT_HOME=/nonexistent does not contain the kit", r.stderr)


if __name__ == "__main__":
    unittest.main()
