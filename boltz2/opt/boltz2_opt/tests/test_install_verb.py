"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), and the weights step's gates (boltz2_opt.weights.fetch with an injected route, cache gate and digest comparator; no network, no
upstream, no torch). CPU only."""
import hashlib
import io
import os
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
RUN_SH = os.path.join(TREE, "run.sh")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG (with a mark when BOLTZ2_OPT reached it); exit codes per call kind by STUB_RC_PIP / _PINS / _WEIGHTS / _PROBE
printf '%s%s\\n' "$*" "${BOLTZ2_OPT:+ [BOLTZ2_OPT=$BOLTZ2_OPT]}" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m boltz2_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="boltz2_install_verb_")
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        self.stub = os.path.join(self.bin, "python")
        with open(self.stub, "w") as f: f.write(STUB)
        os.chmod(self.stub, os.stat(self.stub).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = os.path.join(self.tmp, "calls.log")

    def run_sh(self, args, extra_env=None, **rc):
        env = {"PATH": self.bin + os.pathsep + "/usr/bin:/bin", "HOME": self.tmp, "STUB_LOG": self.log}
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()}); env.update(extra_env or {})
        r = subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env, cwd=self.tmp)
        lines = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        self.probes = [c for c in lines if c.startswith("-I -c import os,sys")]           # the installed-from-this-tree probe (one per install call)
        return r.returncode, r.stdout + r.stderr, [c for c in lines if not c.startswith("-I -c import os,sys")]

    def test_install_runs_pip_then_the_pin_check(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(calls), 2, calls)
        self.assertEqual(calls[0], f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt")
        self.assertEqual(calls[1], f"-I {TREE}/stock/check_pins.py")

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/boltz_cache"], ["install", "--weights=/data/boltz_cache"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[2:], ["-m boltz2_opt.weights /data/boltz_cache"], calls)
            self.assertEqual(len(calls), 3, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--override"], ["install", "--weights="], ["install", "--bogus"],
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

    def test_the_mode_variable_does_not_reach_the_pin_check_or_the_weights_step(self):
        """BOLTZ2_OPT set in the caller's shell: the install step is not a route — its probe, pin check and weights step run with the variable
        dropped (the .pth hook acts on it in every interpreter; importing upstream for the downloader would trip a mode's trigger); pip keeps it."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], extra_env={"BOLTZ2_OPT": "fast"})
        self.assertEqual((rc, len(calls)), (0, 3), (out, calls))
        self.assertTrue(calls[0].endswith(" [BOLTZ2_OPT=fast]") and calls[0].startswith("-m pip install"), calls)
        self.assertEqual(calls[1:], [f"-I {TREE}/stock/check_pins.py", "-m boltz2_opt.weights /w"], calls)
        self.assertEqual([p for p in self.probes if "BOLTZ2_OPT=" in p], [], self.probes)

    def test_a_tree_already_installed_skips_pip_by_name(self):
        """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check (and
        --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py", "-m boltz2_opt.weights /w"], calls)   # pin check, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)


class WeightsCacheGate(unittest.TestCase):
    """boltz2_opt.weights.fetch: fetch what is absent through the route, then the routes' cache gate and the pinned checkpoint's digest — a
    missing entry, an entry without a route, a fetch error or a digest off the pin fails by name; nothing is deleted."""

    ENTRIES = ["boltz2_conf.ckpt", "boltz2_aff.ckpt", "ccd.pkl", "mols", "mols.tar"]
    PAYLOAD = {"boltz2_conf.ckpt": b"conf-bytes", "boltz2_aff.ckpt": b"aff-bytes", "ccd.pkl": b"ccd-bytes", "mols.tar": b"tar-bytes"}

    def setUp(self):
        from boltz2_opt import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="boltz2_weights_")
        self.pin = hashlib.sha256(self.PAYLOAD["boltz2_conf.ckpt"]).hexdigest()
        self.calls = []

    def write(self, name, data=None):
        p = os.path.join(self.dir, name)
        if name == "mols": os.makedirs(p, exist_ok=True)
        elif not os.path.exists(p):
            with open(p, "wb") as f: f.write(self.PAYLOAD[name] if data is None else data)

    def route(self, name):
        """upstream's shape: download_boltz2 writes every entry the cache lacks except ccd.pkl; ccd.pkl has its own call."""
        def download_boltz2():
            self.calls.append(("download_boltz2", name))
            for n in ("mols.tar", "mols", "boltz2_conf.ckpt", "boltz2_aff.ckpt"): self.write(n)
        def ccd():
            self.calls.append(("ccd", name)); self.write("ccd.pkl")
        return ccd if name == "ccd.pkl" else download_boltz2 if name in ("mols.tar", "mols", "boltz2_conf.ckpt", "boltz2_aff.ckpt") else None

    def gate(self, d):
        """stack.cache_check's contract: [] or one refusal naming the missing entries."""
        missing = [n for n in self.ENTRIES if not os.path.exists(os.path.join(d, n))]
        return [f"BOLTZ_CACHE={d} lacks {missing}"] if missing else []

    def matcher(self, d):
        """stack.weights_status's contract: {status pinned|unknown|absent, sha256, pinned_sha256, memo, memo_note}."""
        p = os.path.join(d, "boltz2_conf.ckpt")
        digest = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else None
        return {"status": "absent" if digest is None else "pinned" if digest == self.pin else "unknown", "sha256": digest, "pinned_sha256": self.pin,
                "memo": "written", "memo_note": ""}

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.W.fetch(self.dir, entries=kw.pop("entries", list(self.ENTRIES)), route=kw.pop("route", self.route), matcher=kw.pop("matcher", self.matcher),
                          gate=kw.pop("gate", self.gate), pinned_file="boltz2_conf.ckpt", out=out, **kw)
        return rc, out.getvalue()

    def test_all_fetched_present_and_pinned(self):
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 5/5", out); self.assertIn(f"export BOLTZ_CACHE={self.dir}", out)
        self.assertEqual(self.calls, [("download_boltz2", "boltz2_conf.ckpt"), ("ccd", "ccd.pkl")])   # one upstream batch call, then the CCD file; the rest were present by then
        for n in self.ENTRIES: self.assertTrue(os.path.exists(os.path.join(self.dir, n)), n)

    def test_present_entries_are_kept_and_checked_without_a_fetch(self):
        for n in self.ENTRIES: self.write(n)
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.calls, []); self.assertEqual(out.count(": present"), 5, out)

    def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(self):
        self.write("boltz2_conf.ckpt", b"other bytes")
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out)
        last = out.splitlines()[-1]
        self.assertIn("REFUSED: boltz2_conf.ckpt", last); self.assertIn("not the pin's", last); self.assertIn("left in place", last)
        self.assertEqual(open(os.path.join(self.dir, "boltz2_conf.ckpt"), "rb").read(), b"other bytes")

    def test_an_entry_without_an_upstream_source_is_refused(self):
        rc, out = self.fetch(route=lambda name: None if name == "ccd.pkl" else self.route(name))
        self.assertEqual(rc, 1, out)
        self.assertIn("no download source for ccd.pkl", out)

    def test_a_fetch_error_is_relayed_and_nothing_is_deleted(self):
        def broken(name):
            def call(): raise ConnectionError("network unreachable")
            return call
        rc, out = self.fetch(route=broken)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAILED fetching boltz2_conf.ckpt: ConnectionError: network unreachable", out)
        self.assertEqual(os.listdir(self.dir), [])

    def test_an_entry_still_missing_after_the_fetch_is_refused_by_the_routes_gate(self):
        """A route that returns without writing (upstream skipped it): the missing entry is named — by the post-fetch check or the cache gate."""
        rc, out = self.fetch(route=lambda name: (lambda: None))
        self.assertEqual(rc, 1, out)
        self.assertIn("boltz2_conf.ckpt is not at", out)

    # -- the further pinned entries (boltz2_aff.ckpt, ccd.pkl, mols.tar): hashed after the fetch, refused by name off the pin ----------------
    def others(self, **override):
        pins = {n: hashlib.sha256(self.PAYLOAD[n]).hexdigest() for n in ("boltz2_aff.ckpt", "ccd.pkl", "mols.tar")}
        pins.update(override); return pins

    def test_further_pinned_entries_are_hashed_and_pass_at_their_pins(self):
        rc, out = self.fetch(others=self.others())
        self.assertEqual(rc, 0, out)
        for n in ("boltz2_aff.ckpt", "ccd.pkl", "mols.tar"): self.assertIn(f"{n}: sha256 {hashlib.sha256(self.PAYLOAD[n]).hexdigest()[:16]}… = the pin", out)
        self.assertIn("3 further entries at their pins (boltz2_aff.ckpt, ccd.pkl, mols.tar)", out.splitlines()[-1]); self.assertIn("WEIGHTS OK: 5/5", out.splitlines()[-1])

    def test_a_further_entry_off_its_pin_is_refused_by_name_and_left_in_place(self):
        self.write("ccd.pkl", b"tampered pickle")
        rc, out = self.fetch(others=self.others())
        self.assertEqual(rc, 1, out)
        last = out.splitlines()[-1]
        self.assertIn("REFUSED: 1 of 3 further pinned entries", last); self.assertIn("ccd.pkl", last); self.assertNotIn("export BOLTZ_CACHE", out)
        self.assertIn(f"ccd.pkl: sha256 {hashlib.sha256(b'tampered pickle').hexdigest()} is not the pin", out)
        self.assertEqual(open(os.path.join(self.dir, "ccd.pkl"), "rb").read(), b"tampered pickle")                      # never deleted

    def test_stock_pins_carry_a_digest_for_every_file_the_step_fetches(self):
        """stock/PINS.json weights.files: the structure checkpoint (the routes' WEIGHTS line) and the three further files each carry a 64-hex
        sha256 — no open entry; other_pins reads exactly the three; mols/ is the archive unpacked and carries none of its own."""
        import json, re
        from boltz2_opt import stack
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))
        files = pins["weights"]["files"]
        for n in ("boltz2_conf.ckpt", "boltz2_aff.ckpt", "ccd.pkl", "mols.tar"):
            self.assertRegex(files[n]["sha256"], r"^[0-9a-f]{64}$", n)
        self.assertEqual(self.W.other_pins(pins, stack.WEIGHTS_FILE), {n: files[n]["sha256"] for n in ("boltz2_aff.ckpt", "ccd.pkl", "mols.tar")})
        self.assertEqual([n for n, e in files.items() if not re.fullmatch(r"[0-9a-f]{64}", str(e.get("sha256", "")))], [])   # no open entry

    def test_entries_default_to_the_routes_cache_files_and_every_one_has_an_upstream_source(self):
        """The set fetched is the set the routes' cache gate requires (stack.CACHE_FILES), the pinned file is the routes' (stack.WEIGHTS_FILE), and
        upstream_route — on a stand-in `boltz.main` — has a source for every entry: download_boltz2 for the four it writes, CCD_URL for ccd.pkl."""
        import sys, types
        from boltz2_opt import stack
        self.assertEqual(list(stack.CACHE_FILES), self.ENTRIES); self.assertEqual(stack.WEIGHTS_FILE, "boltz2_conf.ckpt")
        rec = []
        main = types.ModuleType("boltz.main"); main.CCD_URL = "https://upstream.example/ccd.pkl"
        main.download_boltz2 = lambda cache: rec.append(("download_boltz2", str(cache)))
        pkg = types.ModuleType("boltz"); pkg.main = main
        saved = {k: sys.modules.get(k) for k in ("boltz", "boltz.main")}; saved_retrieve = self.W.urllib.request.urlretrieve
        sys.modules.update({"boltz": pkg, "boltz.main": main}); self.W.urllib.request.urlretrieve = lambda url, path: rec.append(("urlretrieve", url, path))
        try:
            route = self.W.upstream_route(self.dir)
            for n in stack.CACHE_FILES: self.assertIsNotNone(route(n), n)
            route("mols.tar")(); route("ccd.pkl")()
            self.assertIsNone(route("boltz1_conf.ckpt"))
        finally:
            self.W.urllib.request.urlretrieve = saved_retrieve
            for k, v in saved.items():
                if v is None: sys.modules.pop(k, None)
                else: sys.modules[k] = v
        self.assertEqual(rec, [("download_boltz2", self.dir), ("urlretrieve", main.CCD_URL, os.path.join(self.dir, "ccd.pkl"))])

    def test_upstream_sources_exist_in_the_pinned_tree(self):
        """The two upstream names the step relies on are in the pinned source (stock/src/boltz/main.py, read as text — no torch import)."""
        src = open(os.path.join(TREE, "stock", "src", "boltz", "main.py"), encoding="utf-8").read()
        self.assertIn("def download_boltz2(cache: Path) -> None:", src); self.assertRegex(src, r'(?m)^CCD_URL = "https://[^"]+/ccd\.pkl"')
        for name in ("mols.tar", "boltz2_conf.ckpt", "boltz2_aff.ckpt"): self.assertIn(f'cache / "{name}"', src.split("def download_boltz2", 1)[1].split("def get_cache_path", 1)[0])


if __name__ == "__main__":
    unittest.main()
