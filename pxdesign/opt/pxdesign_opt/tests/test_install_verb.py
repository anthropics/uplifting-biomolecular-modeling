"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), the weights step's digest gate (pxdesign_opt.weights.fetch with an injected fetcher and pin table; no network, no upstream), and the
one environment definition (environment/: the lock restates stock/PINS.json's pinned stack and stock lines, the Dockerfile builds exactly that).
CPU only."""
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tempfile
import unittest

from .conftest import TREE                                              # the kit tree: run.sh, stock/, opt/, environment/

RUN_SH = os.path.join(TREE, "run.sh")
ENV_DIR = os.path.join(TREE, "environment")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS / STUB_RC_PROBE
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m pxdesign_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pxdesign_install_verb_")
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        self.stub = os.path.join(self.bin, "python")
        with open(self.stub, "w") as f: f.write(STUB)
        os.chmod(self.stub, os.stat(self.stub).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = os.path.join(self.tmp, "calls.log")

    def run_sh(self, args, **rc):
        env = {"PATH": self.bin + os.pathsep + "/usr/bin:/bin", "HOME": self.tmp, "STUB_LOG": self.log}
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()})
        r = subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env, cwd=self.tmp)
        lines = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        self.probes = [c for c in lines if c.startswith("-I -c import os,sys")]           # the installed-from-this-tree probe (one per install call)
        return r.returncode, r.stdout + r.stderr, [c for c in lines if not c.startswith("-I -c import os,sys")]

    def test_install_runs_pip_then_the_pin_check(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt", f"-I {TREE}/stock/check_pins.py"], calls)

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/pxdesign"], ["install", "--weights=/data/pxdesign"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[2:], ["-m pxdesign_opt.weights /data/pxdesign"], calls)
            self.assertEqual(len(calls), 3, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--strict-stack"], ["install", "--weights="], ["install", "--bogus"],
                     ["install", "extra"], ["install", "--mode", "exact"], ["install", "--config", "h100"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out))
            self.assertEqual(calls, [], (args, calls))
            self.assertIn("run.sh:", out)

    def test_a_failed_step_stops_the_sequence_with_its_code(self):
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pip=1)
        self.assertEqual((rc, len(calls)), (1, 1), (out, calls))
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pins=3)
        self.assertEqual((rc, len(calls)), (3, 2), (out, calls))
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], weights=1)
        self.assertEqual((rc, len(calls)), (1, 3), (out, calls))

    def test_a_tree_already_installed_skips_pip_by_name(self):
        """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check (and
        --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py", "-m pxdesign_opt.weights /w"], calls)   # pin check, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_install_comes_before_any_gate_or_config(self):
        """install is handled before the importability probe, the core pin gate and configs/<config>.env: it is the step that makes them pass."""
        src = open(RUN_SH, encoding="utf-8").read()
        block_at = src.index('if [ "$VERB" = install ]; then')
        for later in ('python -c "import pxdesign_opt', 'source "$CFG"', "exec python -m pxdesign_opt"):
            self.assertLess(block_at, src.index(later), later)


class WeightsDigestGate(unittest.TestCase):
    """pxdesign_opt.weights.fetch: fetch through upstream's downloader (one call for every absent file), then every file's sha256 against the pin —
    a mismatch, or a file the downloader did not produce, fails by name; nothing is deleted."""

    def setUp(self):
        from pxdesign_opt import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="pxdesign_weights_")
        self.payload = {"checkpoint/model.pt": b"model-bytes", "checkpoint/companion.pt": b"companion", "ccd_cache/components.cif": b"ccd-bytes"}
        self.files = [{"local": k, "sha256": hashlib.sha256(v).hexdigest(), "bytes": len(v)} for k, v in self.payload.items()]
        self.calls = 0

    def fetcher(self):
        """upstream's rule: every file of the set that is absent is written; a present one is skipped."""
        self.calls += 1
        for local, data in self.payload.items():
            p = os.path.join(self.dir, local)
            if not os.path.exists(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f: f.write(data)

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.W.fetch(self.dir, files=kw.pop("files", self.files), fetcher=kw.pop("fetcher", self.fetcher), out=out)
        return rc, out.getvalue()

    def test_all_fetched_and_pinned(self):
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 3/3", out); self.assertEqual(self.calls, 1)
        self.assertIn(f"PXDESIGN_CKPT_DIR={self.dir}/checkpoint PROTENIX_DATA_ROOT_DIR={self.dir}/ccd_cache", out.splitlines()[-1])
        for f in self.files: self.assertTrue(os.path.isfile(os.path.join(self.dir, f["local"])))

    def test_present_files_are_kept_checked_and_not_fetched(self):
        self.fetcher(); self.calls = 0
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("checkpoint/model.pt: present", out); self.assertEqual(self.calls, 0)   # nothing absent: upstream is not even imported

    def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(self):
        p = os.path.join(self.dir, "checkpoint", "model.pt"); os.makedirs(os.path.dirname(p))
        with open(p, "wb") as f: f.write(b"a short transfer")
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out)
        self.assertIn("REFUSED: 1 of 3", out); self.assertIn("checkpoint/model.pt", out.splitlines()[-1])
        self.assertTrue(os.path.isfile(p)); self.assertEqual(open(p, "rb").read(), b"a short transfer")

    def test_a_file_the_downloader_does_not_produce_is_refused(self):
        files = self.files + [{"local": "checkpoint/extra.pt", "sha256": "0" * 64, "bytes": 1}]
        rc, out = self.fetch(files=files)
        self.assertEqual(rc, 1, out)
        self.assertIn("checkpoint/extra.pt not under", out); self.assertIn("no download entry", out)

    def test_a_fetch_error_is_relayed(self):
        def broken(): raise ConnectionError("network unreachable")
        rc, out = self.fetch(fetcher=broken)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAILED fetching checkpoint/model.pt, checkpoint/companion.pt, ccd_cache/components.cif: ConnectionError: network unreachable", out)
        self.assertEqual(sorted(os.listdir(self.dir)), ["ccd_cache", "checkpoint"])          # the two directories, nothing in them, nothing deleted

    # -- the checkpoints are compared with their pins BEFORE anything can load them: transferred to <name>.part, hashed, then renamed ----------
    def prefetching(self, served):
        """fetch() with the kit's own prefetch_checkpoints bound to a recording retriever serving ``served`` (name -> bytes), and a recording
        upstream fetcher that (like upstream) would load a checkpoint it had to download itself."""
        events = []

        def retrieve(url, path):
            events.append(("retrieve", url, os.path.basename(path)))
            with open(path, "wb") as fh: fh.write(served[url.rsplit("/", 1)[-1]])

        def upstream():
            missing = [k for k in self.payload if k.startswith("checkpoint/") and not os.path.isfile(os.path.join(self.dir, k))]
            events.append(("upstream", "checkpoints present" if not missing else "downloaded+loaded unverified: " + ", ".join(missing)))
            self.fetcher()

        def prefetch(root, files, out=None):
            return self.W.prefetch_checkpoints(root, files, retrieve=retrieve, url_for=lambda name: "https://upstream.example/" + name, out=out)
        out = io.StringIO()
        rc = self.W.fetch(self.dir, files=self.files, fetcher=upstream, out=out, prefetch=prefetch)
        return rc, out.getvalue(), events

    def test_checkpoints_are_transferred_and_compared_before_upstream_runs(self):
        rc, out, events = self.prefetching({"model": self.payload["checkpoint/model.pt"], "companion": self.payload["checkpoint/companion.pt"]})
        self.assertEqual(rc, 0, out)
        self.assertEqual(events, [("retrieve", "https://upstream.example/model", "model.pt.part"), ("retrieve", "https://upstream.example/companion", "companion.pt.part"),
                                  ("upstream", "checkpoints present")], events)
        for k in ("checkpoint/model.pt", "checkpoint/companion.pt"):
            p = os.path.join(self.dir, k)
            self.assertEqual(open(p, "rb").read(), self.payload[k]); self.assertFalse(os.path.exists(p + ".part"))
        self.assertIn("checkpoint/model.pt: transferring https://upstream.example/model to checkpoint/model.pt.part", out)
        self.assertLess(out.index("= the pin; in place before upstream's routine runs"), out.index("WEIGHTS OK: 3/3"))

    def test_a_transfer_off_the_pin_is_refused_before_anything_loads_it(self):
        rc, out, events = self.prefetching({"model": b"tampered bytes", "companion": self.payload["checkpoint/companion.pt"]})
        self.assertEqual(rc, 1, out)
        self.assertEqual(events, [("retrieve", "https://upstream.example/model", "model.pt.part")], events)      # upstream's routine never ran; the second file was not transferred
        p = os.path.join(self.dir, "checkpoint", "model.pt")
        self.assertFalse(os.path.exists(p)); self.assertEqual(open(p + ".part", "rb").read(), b"tampered bytes")   # never under the name anything loads; kept for inspection
        last = out.splitlines()[-1]
        self.assertIn("REFUSED: checkpoint/model.pt.part sha256 ", last); self.assertIn("is not the pin " + self.files[0]["sha256"], last); self.assertIn("never loaded", last)

    def test_present_checkpoints_are_not_transferred_again(self):
        self.fetcher(); os.remove(os.path.join(self.dir, "ccd_cache", "components.cif"))
        rc, out, events = self.prefetching({})
        self.assertEqual(rc, 0, out)
        self.assertEqual(events, [("upstream", "checkpoints present")], events)

    def test_the_real_route_prefetches_with_upstream_names(self):
        import inspect
        src = inspect.getsource(self.W.prefetch_checkpoints)
        self.assertIn("from pxdesign.utils.infer import URL", src); self.assertIn("urllib.request.urlretrieve", src); self.assertIn("os.replace(part, final)", src)
        self.assertLess(src.index("digest(part)"), src.index("os.replace(part, final)"))
        body = inspect.getsource(self.W.fetch)
        self.assertLess(body.index("prefetch(root, files, out=out)"), body.index("            fetcher()"))
        self.assertIn("prefetch = prefetch_checkpoints if prefetch is None else prefetch", body)

    def test_the_pin_table_is_the_seven_files_of_stock_pins(self):
        """pinned_files(PINS) = checkpoint/<the checkpoint and its three companions> + ccd_cache/<the three CCD files>, digests from PINS —
        the same names infer_loop.preflight requires at run time."""
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))
        table = self.W.pinned_files(pins)
        w = pins["weights"]
        expect = [("checkpoint/" + w["checkpoint"]["file"], w["checkpoint"]["sha256"])]
        expect += [("checkpoint/" + n, d) for n, d in w["required_in_dir"]["files"].items()]
        expect += [("ccd_cache/" + n, d) for n, d in pins["ccd_cache"]["files"].items()]
        self.assertEqual([(f["local"], f["sha256"]) for f in table], expect)
        self.assertEqual(len(table), 7); self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) for f in table))

    def test_pins_sources_are_upstreams_url_table(self):
        """stock/PINS.json's source URLs (what STOCK.md lists) are the URLs upstream's downloader fetches: pxdesign.utils.infer.URL, read from the
        pinned source as text (no torch import) — the checkpoint's full URL, the companions' and the CCD cache's base + file name."""
        src = open(os.path.join(TREE, "stock", "src", "PXDesign", "pxdesign", "utils", "infer.py"), encoding="utf-8").read()
        table = dict(re.findall(r'"(\w[\w.]*)":\s*"(https://[^"]+)"', src[src.index("URL = {"):src.index("}", src.index("URL = {"))]))
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8")); w = pins["weights"]
        self.assertEqual(w["checkpoint"]["source"].split()[0], table[w["model_name"]])
        self.assertEqual(os.path.basename(table[w["model_name"]]), w["checkpoint"]["file"])
        base = w["required_in_dir"]["source"].split()[0]
        for name in w["required_in_dir"]["files"]: self.assertEqual(table[name[:-len(".pt")]], base + name, name)
        base = pins["ccd_cache"]["source"].split()[0]
        ccd_urls = {os.path.basename(u): u for k, u in table.items() if k in ("ccd_components_file", "ccd_components_rdkit_mol_file", "pdb_cluster_file")}
        for name in pins["ccd_cache"]["files"]: self.assertEqual(ccd_urls.get(name), base + name, name)


class EnvironmentDefinition(unittest.TestCase):
    """environment/ is the kit's one environment definition: requirements.lock restates stock/PINS.json's pinned stack and stock pins, the
    Dockerfile builds that lock on the pinned CUDA base with the pinned interpreter and ends with `run.sh install`, apptainer.def converts it."""

    @classmethod
    def setUpClass(cls):
        cls.pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))
        cls.lock_lines = [l.strip() for l in open(os.path.join(ENV_DIR, "requirements.lock"), encoding="utf-8") if l.strip() and not l.startswith("#")]
        cls.dockerfile = open(os.path.join(ENV_DIR, "Dockerfile"), encoding="utf-8").read()

    def test_environment_holds_exactly_the_definition(self):
        self.assertEqual(sorted(os.listdir(ENV_DIR)), ["Dockerfile", "apptainer.def", "requirements.lock"])

    def test_lock_restates_the_pinned_stack(self):
        pinned = self.pins["pinned_stack"]["pinned"]
        versions = dict(l.split("==", 1) for l in self.lock_lines if "==" in l and not l.startswith("-"))
        for k in ("triton", "deepspeed", "numpy"): self.assertEqual(versions[k], pinned[k], k)
        cu = "cu" + pinned["cuda"].replace(".", ""); self.assertTrue(pinned["torch"].endswith("+" + cu), pinned["torch"])
        torch_line = [l for l in self.lock_lines if l.startswith("torch @ ")]                                       # the one line that is not PyPI's file: the PyTorch index's CUDA build, by URL and digest
        self.assertEqual(len(torch_line), 1, torch_line)
        self.assertRegex(torch_line[0], r"^torch @ https://download\.pytorch\.org/whl/%s/torch-%s-cp311-cp311-linux_x86_64\.whl#sha256=[0-9a-f]{64}$" % (cu, re.escape(pinned["torch"].replace("+", "%2B"))))
        self.assertFalse([l for l in self.lock_lines if l.startswith("--")], "no index or option lines: every line names one distribution")
        self.assertEqual(versions["nvidia-cudnn-cu12"].split(".")[:3], ["8", "9", "2"]); self.assertEqual(pinned["cudnn"], "8902")   # cuDNN 8.9.2 = torch.backends.cudnn.version() 8902
        for k, v in self.pins["pins"].items():                                        # upstream's own requirement pins the stack honours (torch, numpy, deepspeed, biotite, biopython, modelcif, protobuf)
            if k in versions and re.fullmatch(r"[0-9][\w.]*", v): self.assertEqual(versions[k].split("+")[0], v, k)
        self.assertEqual(len(self.lock_lines), len(set(self.lock_lines)))

    def test_lock_stock_lines_are_the_pinned_upstream(self):
        up = self.pins["upstream"]
        stock = [l for l in self.lock_lines if "github.com/bytedance/" in l]
        self.assertEqual(len(stock), 3, stock)
        self.assertIn(f"protenix @ git+{up['protenix']['repo']}.git@{up['protenix']['commit']}", stock)            # a regular install from the repository: it provides the `configs` and `runner` packages protenix imports
        for name in ("pxdesign", "pxdbench"):                                                                       # editable checkouts at the pinned commit (PXDesign's own `pip install -e .`)
            self.assertIn(f"-e git+{up[name]['repo']}.git@{up[name]['commit']}#egg={name}", stock)

    def test_dockerfile_builds_the_pinned_stack(self):
        pinned = self.pins["pinned_stack"]["pinned"]
        m = re.search(r"^FROM (\S+)$", self.dockerfile, re.M); self.assertIsNotNone(m)
        self.assertEqual(m.group(1), pinned["cuda_base"])                                                          # the development base: nvcc builds stock's LayerNorm extension at first use
        self.assertIn("-devel-", pinned["cuda_base"]); self.assertEqual(pinned["nvcc"], pinned["cuda"])
        self.assertIn(f"cpython-{self.pins['python']}+", self.dockerfile)
        self.assertIn("COPY pxdesign/environment/requirements.lock ", self.dockerfile)
        self.assertRegex(self.dockerfile, r"pip install --no-cache-dir --no-deps --src /opt -r /tmp/stack\.txt")
        ext = self.dockerfile.index("import os, protenix.model.layer_norm.layer_norm as m")                           # stock's LayerNorm CUDA extension compiled at build time, protenix's own way, before the kit is copied
        self.assertLess(self.dockerfile.index("-r /tmp/stack.txt"), ext); self.assertLess(ext, self.dockerfile.index("COPY pxdesign /kit/pxdesign"))
        self.assertIn("COPY common/opt_core /kit/common/opt_core", self.dockerfile); self.assertIn("COPY pxdesign /kit/pxdesign", self.dockerfile)
        self.assertIn("RUN bash run.sh install", self.dockerfile)
        env = dict(kv.split("=", 1) for line in re.findall(r"^ENV (.+)$", self.dockerfile, re.M) for kv in line.split())
        self.assertEqual(env, {"PYTHONHASHSEED": "0", "CFLAGS": "-g0"})                                             # the process environment; no weights path, no cache path, no thread count baked in
        for var in ("PXDESIGN_CKPT_DIR=", "PROTENIX_DATA_ROOT_DIR=", "TORCH_EXTENSIONS_DIR=", "LAYERNORM_TYPE="):
            self.assertNotRegex(self.dockerfile, r"(?m)^ENV .*" + var)

    def test_apptainer_converts_the_docker_image(self):
        d = open(os.path.join(ENV_DIR, "apptainer.def"), encoding="utf-8").read()
        self.assertRegex(d, r"(?m)^Bootstrap: docker-daemon$"); self.assertRegex(d, r"(?m)^From: pxdesign-kit:dev$")
        self.assertIn("export PYTHONNOUSERSITE=1", d); self.assertIn("exec bash /kit/pxdesign/run.sh \"$@\"", d)


if __name__ == "__main__":
    unittest.main()
