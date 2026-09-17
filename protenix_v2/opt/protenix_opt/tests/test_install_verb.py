"""`run.sh install [--weights DIR]` and `environment/`: the verb's argument handling, the weights step (upstream's downloader bound after
PROTENIX_ROOT_DIR, every file checked against stock/PINS.json, refusals by name, nothing deleted), the pin check's shape, and the environment
definition restating the pins (Dockerfile base / interpreter / lock / wheel / install step; the lock's stack entries == PINS pinned_stack)."""
import hashlib
import json
import os
import re
import subprocess
import sys
import types

import pytest

from protenix_opt import _frozen, manifest, weights

OPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # <tree>/opt/protenix_opt
TREE = os.path.dirname(os.path.dirname(OPT))                                # <tree>
PINS = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fake_pins(payloads):
    """PINS-shaped dict whose checkpoint + data_caches digests are those of ``payloads`` {relpath: bytes} (checkpoint = manifest.CHECKPOINT_RELPATH)."""
    files = {rel: {"sha256": _sha(b), "bytes": len(b)} for rel, b in payloads.items() if rel != manifest.CHECKPOINT_RELPATH}
    return {"checkpoint": {"file": manifest.CHECKPOINT_RELPATH, "sha256": _sha(payloads[manifest.CHECKPOINT_RELPATH]), "root_env": "PROTENIX_ROOT_DIR"},
            "data_caches": {"files": files}}


PAYLOADS = {manifest.CHECKPOINT_RELPATH: b"ckpt-bytes", "common/components.cif": b"ccd", "common/release_date_cache.json": b"{}"}


def writer(root, payloads):
    """A stand-in for upstream's downloader: writes every absent payload under root (present files untouched, as upstream does)."""
    calls = []
    def route():
        calls.append(1)
        for rel, b in payloads.items():
            p = os.path.join(root, rel)
            if not os.path.isfile(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as fh: fh.write(b)
    return route, calls


@pytest.fixture()
def memo(tmp_path, monkeypatch):
    monkeypatch.setenv(manifest.CACHE_DIR_ENV, str(tmp_path / "memo"))     # the digest memo of weights_status, kept out of ~/.cache
    return tmp_path


def test_wanted_is_every_file_the_stock_cli_downloads():
    """PINS checkpoint + data_caches == the frozen-inputs list for the pinned checkpoint with templates on (_frozen.relpaths) — one file set."""
    assert set(weights.wanted(PINS)) == set(_frozen.relpaths(manifest.CHECKPOINT_NAME, use_template=True))
    assert weights.wanted(PINS)[0] == PINS["checkpoint"]["file"] == manifest.CHECKPOINT_RELPATH
    assert len(weights.wanted(PINS)) == 7 and all(len(e["sha256"]) == 64 and e["bytes"] > 0 for e in PINS["data_caches"]["files"].values())


def test_empty_dir_fetched_then_ok(memo):
    root = str(memo / "w"); pins = fake_pins(PAYLOADS); route, calls = writer(root, PAYLOADS); lines = []
    assert weights.main([root], route=route, log=lines.append) == weights.EXIT_OK
    assert calls == [1] and all(os.path.isfile(os.path.join(root, r)) for r in weights.wanted(pins) if r in PAYLOADS)
    assert any(l.endswith(f"export PROTENIX_ROOT_DIR={root}") and "WEIGHTS OK: 3/3" in l for l in lines), lines
    assert any("WEIGHTS pinned" in l for l in lines)                          # the checkpoint line is manifest.weights_status's own


def test_present_files_are_kept_and_upstream_is_not_called(memo):
    root = str(memo / "w"); route, calls = writer(root, PAYLOADS); route(); calls.clear()
    def refuse(): raise AssertionError("upstream must not run when nothing is absent")
    lines = []
    assert weights.main([root], route=refuse, log=lines.append) == weights.EXIT_OK
    assert sum("present — kept" in l for l in lines) == 3


def test_a_file_off_its_pin_is_refused_by_name_and_left_in_place(memo):
    root = str(memo / "w"); route, _ = writer(root, {**PAYLOADS, "common/components.cif": b"tampered"}); lines = []
    assert weights.main([root], route=route, log=lines.append) == weights.EXIT_FAIL
    assert any("common/components.cif: sha256 " in l and "NOT the pin" in l for l in lines), lines
    assert any(l.startswith(f"{weights.PREFIX} REFUSED: 1 of 3") for l in lines)
    assert open(os.path.join(root, "common/components.cif"), "rb").read() == b"tampered"   # never deleted


def test_checkpoint_off_its_pin_is_refused(memo):
    root = str(memo / "w"); route, _ = writer(root, {**PAYLOADS, manifest.CHECKPOINT_RELPATH: b"other-model"}); lines = []
    assert weights.main([root], route=route, log=lines.append) == weights.EXIT_FAIL
    assert any("WEIGHTS unknown" in l for l in lines) and any("REFUSED: 1 of 3" in l and manifest.CHECKPOINT_RELPATH in l for l in lines), lines


def test_a_failed_transfer_names_what_is_absent(memo):
    root = str(memo / "w"); lines = []
    def broken(): raise OSError("connection reset")
    assert weights.main([root], route=broken, log=lines.append) == weights.EXIT_FAIL
    assert any("upstream's download FAILED: OSError: connection reset" in l for l in lines)
    assert any("FAILED: 3 of 3 files absent" in l for l in lines)


def test_a_partial_transfer_checks_what_arrived_and_names_the_rest(memo):
    """The checkpoint's server refuses (upstream raises after writing the caches): the caches are checked and reported pinned, the checkpoint is
    named absent, exit 1; the re-run with the file placed by hand passes without a transfer."""
    root = str(memo / "w"); lines = []
    caches = {rel: b for rel, b in PAYLOADS.items() if rel != manifest.CHECKPOINT_RELPATH}
    write_caches, _ = writer(root, caches)
    def partial():
        write_caches(); raise OSError("HTTP Error 403: Forbidden")
    assert weights.main([root], route=partial, log=lines.append) == weights.EXIT_FAIL
    assert any("data caches pinned: 2/2" in l for l in lines), lines
    assert any("FAILED: 1 of 3 files absent" in l and manifest.CHECKPOINT_RELPATH in l for l in lines), lines
    assert not any(l.startswith(weights.PREFIX + " REFUSED") for l in lines)
    os.makedirs(os.path.join(root, os.path.dirname(manifest.CHECKPOINT_RELPATH)), exist_ok=True)
    with open(os.path.join(root, manifest.CHECKPOINT_RELPATH), "wb") as fh: fh.write(PAYLOADS[manifest.CHECKPOINT_RELPATH])
    lines2 = []
    def refuse(): raise AssertionError("upstream must not be called when every file is present")
    assert weights.main([root], route=refuse, log=lines2.append) == weights.EXIT_OK
    assert any("WEIGHTS OK: 3/3" in l for l in lines2)


# -- the checkpoint is compared with its pin BEFORE anything can load it: transferred to <name>.part, hashed, then renamed ------------------------
def _prefetching(root, served: bytes):
    """main() with the kit's own prefetch_checkpoint bound to a recording retriever serving ``served``, and a recording upstream route that (like
    upstream) would load a checkpoint it had to download itself."""
    events, lines = [], []
    upstream, _ = writer(root, PAYLOADS)

    def retrieve(url, path):
        events.append(("retrieve", url, os.path.basename(path)))
        with open(path, "wb") as fh: fh.write(served)

    def route():
        ck = os.path.join(root, manifest.CHECKPOINT_RELPATH)
        events.append(("upstream", "checkpoint present" if os.path.isfile(ck) else "checkpoint downloaded+loaded unverified")); upstream()

    def prefetch(root_, pins, log=print):
        return weights.prefetch_checkpoint(root_, pins, retrieve=retrieve, url_for=lambda name: "https://upstream.example/" + name + ".pt", log=log)
    rc = weights.main([root], route=route, log=lines.append, prefetch=prefetch)
    return rc, lines, events


def test_the_checkpoint_is_transferred_and_compared_before_upstream_runs(memo):
    root = str(memo / "w")
    rc, lines, events = _prefetching(root, PAYLOADS[manifest.CHECKPOINT_RELPATH])
    assert rc == weights.EXIT_OK, lines
    assert events == [("retrieve", f"https://upstream.example/{manifest.CHECKPOINT_NAME}.pt", os.path.basename(manifest.CHECKPOINT_RELPATH) + ".part"), ("upstream", "checkpoint present")], events
    ck = os.path.join(root, manifest.CHECKPOINT_RELPATH)
    assert open(ck, "rb").read() == PAYLOADS[manifest.CHECKPOINT_RELPATH] and not os.path.exists(ck + ".part")
    i = next(n for n, l in enumerate(lines) if "= the pin; in place before upstream's routine runs" in l)
    j = next(n for n, l in enumerate(lines) if "WEIGHTS OK: 3/3" in l)
    assert i < j, lines


def test_a_transfer_off_the_pin_is_refused_before_anything_loads_it(memo):
    root = str(memo / "w")
    rc, lines, events = _prefetching(root, b"tampered bytes")
    assert rc == weights.EXIT_FAIL, lines
    assert events == [("retrieve", f"https://upstream.example/{manifest.CHECKPOINT_NAME}.pt", os.path.basename(manifest.CHECKPOINT_RELPATH) + ".part")], events   # upstream's routine never ran
    ck = os.path.join(root, manifest.CHECKPOINT_RELPATH)
    assert not os.path.exists(ck) and open(ck + ".part", "rb").read() == b"tampered bytes"          # never under the name anything loads; kept for inspection
    refused = [l for l in lines if l.startswith(f"{weights.PREFIX} REFUSED: {manifest.CHECKPOINT_RELPATH}.part sha256 ")]
    assert len(refused) == 1 and "is not the pin " + _sha(PAYLOADS[manifest.CHECKPOINT_RELPATH]) in refused[0] and "never loaded" in refused[0], lines
    assert any("FAILED: 3 of 3 files absent" in l for l in lines), lines


def test_a_present_checkpoint_is_not_transferred_again(memo):
    root = str(memo / "w"); p = os.path.join(root, manifest.CHECKPOINT_RELPATH); os.makedirs(os.path.dirname(p))
    with open(p, "wb") as fh: fh.write(PAYLOADS[manifest.CHECKPOINT_RELPATH])
    rc, lines, events = _prefetching(root, b"unused")
    assert rc == weights.EXIT_OK, lines
    assert events == [("upstream", "checkpoint present")], events


def test_a_checkpoint_transfer_the_server_refuses_still_fetches_the_caches(memo):
    """The checkpoint's server refuses this step's own transfer (HTTP 403, README 'Setup'): named, the data caches fetched by upstream's routine
    regardless, the checkpoint reported absent, exit 1 — no REFUSED line (nothing arrived to compare)."""
    root = str(memo / "w"); lines, events = [], []
    caches = {rel: b for rel, b in PAYLOADS.items() if rel != manifest.CHECKPOINT_RELPATH}
    write_caches, calls = writer(root, caches)
    def route():
        events.append("upstream"); write_caches()
    def prefetch(root_, pins, log=print):
        events.append("prefetch"); raise OSError("HTTP Error 403: Forbidden")
    assert weights.main([root], route=route, log=lines.append, prefetch=prefetch) == weights.EXIT_FAIL
    assert events == ["prefetch", "upstream"], events
    assert any(f"{manifest.CHECKPOINT_RELPATH}: the transfer did not complete (OSError: HTTP Error 403: Forbidden)" in l for l in lines), lines
    assert any("data caches pinned: 2/2" in l for l in lines) and any("FAILED: 1 of 3 files absent" in l and manifest.CHECKPOINT_RELPATH in l for l in lines), lines
    assert not any(l.startswith(weights.PREFIX + " REFUSED") for l in lines)


def test_upstreams_routine_is_handed_a_stand_in_checkpoint_directory():
    """upstream_route points upstream's ``load_checkpoint_dir`` at a private directory already holding the checkpoint's name, so upstream's
    routine transfers the data caches only (source check; the routine itself needs the pinned stack)."""
    import inspect
    src = inspect.getsource(weights.upstream_route)
    assert 'tempfile.mkdtemp(prefix="protenix-opt-ckptdir-")' in src and 'CHECKPOINT_NAME + ".pt"' in src and '"load_checkpoint_dir": standin' in src
    assert 'inference_configs["load_checkpoint_dir"]' not in src


def test_the_real_route_prefetches_with_upstream_names():
    import inspect
    src = inspect.getsource(weights.prefetch_checkpoint)
    assert "from protenix.web_service.dependency_url import URL" in src and "url_for(CHECKPOINT_NAME)" in src
    assert "urllib.request.urlretrieve" in src and src.index("sha256_file(part)") < src.index("os.replace(part, final)")
    body = inspect.getsource(weights.fetch)
    assert body.index("prefetch(root, pins, log=log)") < body.index("            route()")
    assert "prefetch = prefetch_checkpoint if prefetch is None else prefetch" in body


def test_usage():
    assert weights.main([]) == weights.EXIT_USAGE and weights.main(["--weights"]) == weights.EXIT_USAGE and weights.main(["a", "b"]) == weights.EXIT_USAGE


@pytest.fixture(autouse=True)
def _pins(monkeypatch, request):
    """The weights tests read fake pins for the fake payloads; the real PINS stay for the tests that name them."""
    if "memo" in request.fixturenames:
        from protenix_opt import stack
        monkeypatch.setattr(stack, "pins", lambda: fake_pins(PAYLOADS))


def test_upstream_route_binds_after_the_root_and_refuses_a_stale_import(tmp_path, monkeypatch):
    """upstream_route sets PROTENIX_ROOT_DIR, then reads upstream's config modules; modules that resolved another root earlier are refused by name;
    the configs it hands upstream's downloader name the pinned checkpoint, templates on, and the root's checkpoint dir."""
    rec = {}
    def install(root_seen):
        cd = types.ModuleType("configs.configs_data"); cd.PROTENIX_ROOT_DIR = root_seen; cd.data_configs = {"ccd_components_file": os.path.join(root_seen, "common/components.cif"), "template": {}}
        ci = types.ModuleType("configs.configs_inference"); ci.PROTENIX_ROOT_DIR = root_seen; ci.inference_configs = {"load_checkpoint_dir": os.path.join(root_seen, "checkpoint")}
        pkg = types.ModuleType("configs"); pkg.configs_data = cd; pkg.configs_inference = ci
        ri = types.ModuleType("runner.inference"); ri.download_inference_cache = lambda configs: rec.setdefault("configs", configs)
        rpkg = types.ModuleType("runner"); rpkg.inference = ri
        for name, mod in {"configs": pkg, "configs.configs_data": cd, "configs.configs_inference": ci, "runner": rpkg, "runner.inference": ri}.items():
            monkeypatch.setitem(sys.modules, name, mod)
    pytest.importorskip("ml_collections")
    root = str(tmp_path / "w")
    monkeypatch.delenv("PROTENIX_ROOT_DIR", raising=False)
    install("/elsewhere")
    with pytest.raises(RuntimeError, match="resolved PROTENIX_ROOT_DIR='/elsewhere' before"):
        weights.upstream_route(root)
    install(root)
    weights.upstream_route(root)()
    assert os.environ["PROTENIX_ROOT_DIR"] == root
    c = rec["configs"]; assert c.model_name == manifest.CHECKPOINT_NAME and c.use_template is True and c["data"]["ccd_components_file"].startswith(root)
    assert c.load_checkpoint_dir != os.path.join(root, "checkpoint") and os.path.isfile(os.path.join(c.load_checkpoint_dir, manifest.CHECKPOINT_NAME + ".pt"))   # the stand-in: upstream transfers the caches only
    assert os.path.getsize(os.path.join(c.load_checkpoint_dir, manifest.CHECKPOINT_NAME + ".pt")) == 0 and not os.path.exists(os.path.join(root, "checkpoint", manifest.CHECKPOINT_NAME + ".pt"))


def _run_sh(*args, env=None):
    return subprocess.run(["bash", os.path.join(TREE, "run.sh"), *args], capture_output=True, text=True, cwd=TREE, env=env, timeout=120)


def test_run_sh_install_usage_errors():
    r = _run_sh("install", "--weights"); assert r.returncode == 2 and "install --weights takes a directory" in r.stderr
    r = _run_sh("install", "--weights", "--verbose"); assert r.returncode == 2
    r = _run_sh("install", "--weights="); assert r.returncode == 2 and "takes a directory" in r.stderr
    r = _run_sh("install", "--config", "h100"); assert r.returncode == 2 and "install takes no --config / --mode" in r.stderr


def test_run_sh_install_is_pip_then_pins_then_weights():
    text = open(os.path.join(TREE, "run.sh"), encoding="utf-8").read()
    block = text[text.index('if [ "$CMD" = install ]'):text.index("\nfi\n", text.index('if [ "$CMD" = install ]'))]
    i_pip, i_pins, i_w = block.index('python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt"'), block.index('python -I "$HERE/stock/check_pins.py"'), block.index("python -m protenix_opt.weights")
    assert i_pip < i_pins < i_w
    assert '${PIPARGS[@]+"${PIPARGS[@]}"}' in block                        # extra arguments still go to pip
    assert re.search(r"(?m)^case \"\$CMD\" in pred\|check\|warm\|install\) ;; \*\) usage ;; esac$", text)


def test_check_pins_is_stdlib_and_reads_the_pinned_wheel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ptx_check_pins", os.path.join(TREE, "stock", "check_pins.py")); cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
    src = open(os.path.join(TREE, "stock", "check_pins.py"), encoding="utf-8").read()
    assert not re.search(r"(?m)^(from|import) (torch|protenix|opt_core|protenix_opt)\b", src)   # standard library only
    rec = cp.wheel_record(os.path.join(TREE, PINS["wheel"]["file"]))
    assert len(rec) > 100 and all(k.endswith(".py") and len(v) == 64 for k, v in rec.items()) and "runner/inference.py" in rec
    assert cp.stack_line([]) is None and cp.stack_line([("torch", "2.9.0+cu128", "2.13.0+cu130")]).startswith("STACK not pinned: torch 2.9.0+cu128 (pinned 2.13.0+cu130)")
    assert set(cp.STACK_DISTS) <= set(PINS["pinned_stack"])


def _lock():
    """{name: version} over the lock's `name==version` lines and its `name @ <url>#sha256=` lines (version read from the wheel filename)."""
    from urllib.parse import unquote
    lock = {}
    for line in open(os.path.join(TREE, "environment", "requirements.lock"), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"): continue
        if " @ " in line:
            k, url = line.split(" @ ", 1); assert re.search(r"#sha256=[0-9a-f]{64}$", url), line
            lock[k.lower().replace("_", "-")] = unquote(url.rsplit("/", 1)[1]).split("-")[1]
        elif "==" in line:
            k, v = line.split("==", 1); lock[k.lower().replace("_", "-")] = v
    return lock


def test_lock_restates_the_pinned_stack():
    """environment/requirements.lock is the stack's one package list; its stack entries equal stock/PINS.json pinned_stack and protenix_version."""
    lock, st = _lock(), PINS["pinned_stack"]
    for k in ("torch", "triton", "cuequivariance-torch", "cuequivariance-ops-torch-cu13"): assert lock[k] == st[k], k
    assert st["torch"].endswith("+cu" + st["cuda"].replace(".", "")) and lock["nvidia-cuda-runtime"].startswith(st["cuda"] + ".")
    assert lock["protenix"] == PINS["protenix_version"] and st["python"] == PINS["python"]
    assert "protenix-opt" not in lock and "opt-core" not in lock                # the kit and its core install editable on top, never from the lock
    assert not [l for l in open(os.path.join(TREE, "environment", "requirements.lock"), encoding="utf-8") if l.startswith("-")]   # no index / option lines: files off PyPI are pinned by URL + sha256
    urls = [l.split(" @ ", 1) for l in open(os.path.join(TREE, "environment", "requirements.lock"), encoding="utf-8") if " @ " in l and not l.startswith("#")]
    assert sorted(k for k, _ in urls) == ["torch", "torchvision", "triton"] and all(u.startswith("https://download.pytorch.org/whl/") for _, u in urls)


def test_environment_dockerfile_builds_the_pinned_stack():
    """environment/Dockerfile: the CUDA base tag carries PINS cuda, the python-build-standalone tarball PINS python, the lock and the stock wheel are the
    files it installs, `run.sh install` is its kit step; apptainer.def converts that image."""
    df = open(os.path.join(TREE, "environment", "Dockerfile"), encoding="utf-8").read(); st = PINS["pinned_stack"]
    m = re.search(r"^FROM (\S+)$", df, re.M); assert m and m.group(1).startswith(f"nvidia/cuda:{st['cuda']}.") and "-devel-ubuntu24.04" in m.group(1), m and m.group(1)
    assert f"cpython-{PINS['python']}+" in df
    assert "COPY protenix_v2/environment/requirements.lock " in df and f"COPY protenix_v2/{PINS['wheel']['file']} " in df
    assert "grep -E '^pip==' /tmp/requirements.lock | xargs python -m pip install" in df and "pip install --no-cache-dir --no-deps -r /tmp/stack.txt" in df and "index-url" not in df and 'python -c "import protenix.model.layer_norm"' in df
    assert "RUN bash run.sh install" in df and not re.search(r"(?m)^ENV .*PROTENIX_ROOT_DIR", df) and "NUM_THREADS" not in df
    assert re.search(r"(?m)^From: protenix_v2-kit:", open(os.path.join(TREE, "environment", "apptainer.def"), encoding="utf-8").read())
