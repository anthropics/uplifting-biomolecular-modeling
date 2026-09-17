"""opt_core.jax_design.pcc: the persistent compilation cache + autotune pin — key, place, XLA_FLAGS append rule, the two routes, the
never-re-apply rule, equality key, evidence fields. No jax is installed in the core's test interpreter; a live jax is a stub object."""
import hashlib
import importlib
import os
import sys
import types

import pytest

import opt_core
from opt_core.jax_design import pcc


def _populate(cache_dir, payload=b"tuned"):
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, pcc.AUTOTUNE_FILENAME), "wb") as fh:
        fh.write(payload)


# --------------------------------------------------------------------------------------------------------------------- hygiene

def test_package_imports_no_jax():
    """The package imports no jax BY ITSELF — checked in a fresh interpreter, whatever this session imported before it."""
    import subprocess
    import opt_core.jax_design as jd                                       # docstring-only package: submodules are imported by name
    assert jd.pcc is pcc
    core_dir = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))
    code = ("import sys; import opt_core.jax_design; from opt_core.jax_design import pcc; "
            "bad = sorted(m for m in ('jax', 'jaxlib') if m in sys.modules); print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": core_dir, "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 0, "importing opt_core.jax_design imported: " + r.stdout + r.stderr[-500:]


# ------------------------------------------------------------------------------------------------------------------ key / place

def test_gpu_slug_plugin_label_and_key_form():
    assert pcc.gpu_slug("NVIDIA H100 80GB HBM3") == "nvidia-h100-80gb-hbm3"
    assert pcc.gpu_slug("  Tesla V100-SXM2-16GB ") == "tesla-v100-sxm2-16gb"
    assert pcc.plugin_label("jax-cuda12-plugin") == "cuda12plugin"
    assert pcc.plugin_label("jax_rocm60_plugin") == "rocm60plugin"
    assert pcc.key("0.4.30", "0.4.30", "0.4.30", "NVIDIA H100 80GB HBM3") == "jax0.4.30-jaxlib0.4.30-cuda12plugin0.4.30-nvidia-h100-80gb-hbm3"
    assert pcc.key("0.5.2+cuda12", "0.5.1", "0.5.1", "NVIDIA H100 PCIe", plugin_dist="jax-cuda13-plugin") == \
        "jax0.5.2-jaxlib0.5.1-cuda13plugin0.5.1-nvidia-h100-pcie"


def test_key_refuses_by_name_for_every_missing_part(monkeypatch):
    monkeypatch.setattr(pcc.gates, "dist_version", lambda name: None)
    monkeypatch.setattr(pcc.gates, "nvidia_smi_probe", lambda **kw: {"name": None})
    with pytest.raises(pcc.PccError, match="no installed 'jax'"):
        pcc.key(gpu_name="NVIDIA H100 80GB HBM3")
    with pytest.raises(pcc.PccError, match="no installed 'jaxlib'"):
        pcc.key("0.4.30", gpu_name="NVIDIA H100 80GB HBM3")
    with pytest.raises(pcc.PccError, match="no installed 'jax-cuda12-plugin'"):
        pcc.key("0.4.30", "0.4.30", gpu_name="NVIDIA H100 80GB HBM3")
    with pytest.raises(pcc.PccError, match="no GPU name") as ex:
        pcc.key("0.4.30", "0.4.30", "0.4.30")
    from opt_core import gates, jit_cache
    assert gates.is_cannot_run(ex.value)                                    # the environment's refusal: the kit refuses the mode by name
    with pytest.raises(pcc.PccError) as usage:
        pcc.autotune_mode("/nowhere/a.pb", "sometimes")
    assert not gates.is_cannot_run(usage.value)                             # a usage error is not a cannot-run event
    # exact=False (a tolerance-class line): never a shared `unknown` bucket — the ISOLATED per-process part, and key_facts names it
    tok = jit_cache.isolation_token()
    assert len(tok) == 8 and tok == jit_cache.isolation_token()             # one token per process
    assert pcc.key("0.4.30", "0.4.30", "0.4.30", exact=False) == f"jax0.4.30-jaxlib0.4.30-cuda12plugin0.4.30-unknown{tok}"
    f = pcc.key_facts("0.4.30", None, None, "NVIDIA H100 80GB HBM3")
    assert f["key"] == f"jax0.4.30-jaxlibunknown{tok}-cuda12pluginunknown{tok}-nvidia-h100-80gb-hbm3"
    assert f["unknown"] == ["jaxlib", "plugin"] and f["word"] == "cache_key=unknown(jaxlib,plugin)" and f["parts"]["jax"] == "0.4.30"
    full = pcc.key_facts("0.4.30", "0.4.30", "0.4.30", "NVIDIA H100 80GB HBM3")
    assert full["key"] == pcc.key("0.4.30", "0.4.30", "0.4.30", "NVIDIA H100 80GB HBM3") and full["unknown"] == [] and full["word"] is None
    live = types.SimpleNamespace(__version__="0.5.3.dev1")                   # a live module's own version string is READ, not unknown
    monkeypatch.setitem(sys.modules, "jax", live)
    assert pcc.key_facts(None, "0.5.3", "0.5.3", "H100")["key"] == "jax0.5.3.dev1-jaxlib0.5.3-cuda12plugin0.5.3-h100"
    with pytest.raises(pcc.PccError, match="no installed 'jax'"):           # exact still reads distribution metadata only and refuses
        pcc.key(None, "0.5.3", "0.5.3", "H100")


def test_cache_dir_layout():
    assert pcc.cache_dir("/jitcache", "k") == "/jitcache/k/xla"
    assert pcc.cache_dir("/jitcache", "k", "engine", "len169") == "/jitcache/k/engine/len169"
    assert pcc.autotune_file_of("/c") == "/c/" + pcc.AUTOTUNE_FILENAME
    assert pcc.autotune_file_of("/c", "/elsewhere/a.pb") == "/elsewhere/a.pb"


# --------------------------------------------------------------------------------------------------------------------- XLA_FLAGS

def test_xla_flags_append_never_drops_or_duplicates():
    base = "--xla_gpu_deterministic_ops=true --xla_gpu_enable_triton_gemm=false"
    out = pcc.xla_flags_append(base, "--xla_gpu_load_autotune_results_from=/c/a.pb")
    assert out == base + " --xla_gpu_load_autotune_results_from=/c/a.pb"
    assert pcc.xla_flags_append(out, "--xla_gpu_load_autotune_results_from=/other.pb") == out      # same NAME: first writer wins
    assert pcc.xla_flags_append(None, "--a=1", "", "--b=2", "--a=3") == "--a=1 --b=2"
    assert pcc.xla_flags_autotune(out) == {"mode": "load", "file": "/c/a.pb"}
    assert pcc.xla_flags_autotune(base) is None
    assert pcc.xla_flags_autotune("--xla_gpu_dump_autotune_results_to=/c/a.pb") == {"mode": "dump", "file": "/c/a.pb"}


def test_autotune_mode_is_load_or_refuse_never_an_implicit_dump(tmp_path):
    f = tmp_path / pcc.AUTOTUNE_FILENAME
    with pytest.raises(pcc.PccError, match="not populated") as ex:
        pcc.autotune_mode(str(f), "auto")                                  # exact (default): a cold process never dumps implicitly (warm once, then fleet)
    from opt_core import gates
    assert gates.is_cannot_run(ex.value)
    with pytest.raises(pcc.PccError, match="not populated"):
        pcc.autotune_mode(str(f), "load")
    with pytest.raises(pcc.PccError, match="not populated"):
        pcc.autotune_mode(str(f), "auto", exact=True)
    assert pcc.autotune_mode(str(f), "auto", exact=False) == "cold"       # a tolerance-class line proceeds cold, named
    assert pcc.autotune_mode(str(f), "load", exact=False) == "cold"
    assert pcc.autotune_mode(str(f), "dump") == "dump"                    # the warm verb's explicit request
    assert pcc.autotune_mode(str(f), "dump", exact=False) == "dump"
    f.write_bytes(b"results")
    assert pcc.autotune_mode(str(f), "auto") == "load" == pcc.autotune_mode(str(f), "auto", exact=False)   # populated: the tier changes nothing
    assert pcc.autotune_mode(str(f), "off") == "off"
    with pytest.raises(pcc.PccError):
        pcc.autotune_mode(str(f), "sometimes")
    assert pcc.autotune_flag("load", "/c/a.pb") == pcc.FLAG_LOAD + "=/c/a.pb"
    assert pcc.autotune_flag("dump", "/c/a.pb") == pcc.FLAG_DUMP + "=/c/a.pb"
    assert pcc.autotune_flag("off", "/c/a.pb") is None and pcc.autotune_flag("cold", "/c/a.pb") is None


# ----------------------------------------------------------------------------------------------------------- the environment route

def test_plan_and_env_are_pure_and_append(tmp_path):
    c = str(tmp_path / "xla")
    environ = {"XLA_FLAGS": "--xla_gpu_deterministic_ops=true", "KEEP": "1"}
    warm = pcc.plan(c, autotune="dump", environ=environ)                  # the warm verb's plan (populating process)
    assert warm["autotune"] == "dump" and warm["autotune_file"] == os.path.join(c, pcc.AUTOTUNE_FILENAME)
    assert warm["exports"] == {"JAX_COMPILATION_CACHE_DIR": c,
                               "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
                               "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
                               "XLA_FLAGS": f"--xla_gpu_deterministic_ops=true {pcc.FLAG_DUMP}={c}/{pcc.AUTOTUNE_FILENAME}"}
    assert environ == {"XLA_FLAGS": "--xla_gpu_deterministic_ops=true", "KEEP": "1"} and not os.path.exists(c)   # untouched, nothing created
    assert warm["exact"] is True and warm["words"] == []
    with pytest.raises(pcc.PccError, match="not populated"):
        pcc.env(c, environ={})                                             # the exact fleet's row on a cold cache refuses by name
    cold = pcc.plan(c, environ={"XLA_FLAGS": "--xla_gpu_deterministic_ops=true"}, exact=False)   # a tolerance-class row on a cold cache: proceeds, named
    assert cold["autotune"] == "cold" and cold["words"] == ["autotune=cold"] and cold["exact"] is False
    assert cold["exports"] == {"JAX_COMPILATION_CACHE_DIR": c, "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
                               "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0", "XLA_FLAGS": "--xla_gpu_deterministic_ops=true"}   # the cache keyed, NO autotune flag
    assert pcc.env(c, environ={}, exact=False) == {k: v for k, v in cold["exports"].items() if k != "XLA_FLAGS"}
    assert pcc.evidence_fields(cold)["autotune"] == "cold"
    _populate(c)
    assert pcc.env(c, environ={})["XLA_FLAGS"] == f"{pcc.FLAG_LOAD}={c}/{pcc.AUTOTUNE_FILENAME}"
    assert pcc.plan(c, environ={}, exact=False)["autotune"] == "load"        # populated: both tiers load
    off = pcc.env(c, autotune="off", environ={})
    assert "XLA_FLAGS" not in off and off["JAX_COMPILATION_CACHE_DIR"] == c
    assert pcc.env(c, autotune="off", environ={"XLA_FLAGS": "--k=v"})["XLA_FLAGS"] == "--k=v"


# ------------------------------------------------------------------------------------------------------------- the in-process route

class _Config:
    def __init__(self):
        self.values = {"jax_compilation_cache_dir": None}

    def update(self, name, value):
        self.values[name] = value

    @property
    def jax_compilation_cache_dir(self):
        return self.values.get("jax_compilation_cache_dir")


def _stub_jax():
    return types.SimpleNamespace(config=_Config())


def _bridge(monkeypatch, initialized):
    xb = types.SimpleNamespace(backends_are_initialized=lambda: initialized) if initialized is not None else types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "jax._src.xla_bridge", xb)


def test_enable_without_jax_writes_environ_and_is_idempotent(tmp_path):
    if sys.modules.get("jax") is not None:                                   # jax_module=None reads the LIVE jax when one is imported: this is the no-jax route
        pytest.skip("holds the no-jax route (jax_module=None reads the live jax when one is imported); this interpreter has jax %s — the live "
                    "routes are test_enable_mirrors_into_a_live_uninitialised_jax / test_enable_refuses_when_backends_initialised_and_names_an_unknown_state"
                    % getattr(sys.modules["jax"], "__version__", "?"))
    c = str(tmp_path / "k" / "xla")
    environ = {"XLA_FLAGS": "--xla_gpu_deterministic_ops=true"}
    rec = pcc.enable(c, autotune="dump", environ=environ, jax_module=None)      # a resident warm: dump, explicitly
    assert rec["applied_via"] == "environ" and rec["backend_state"] == "no_jax" and rec["created"] is True and os.path.isdir(c)
    assert rec["autotune"] == "dump" and environ["JAX_COMPILATION_CACHE_DIR"] == c
    assert environ["XLA_FLAGS"].startswith("--xla_gpu_deterministic_ops=true ") and pcc.FLAG_DUMP in environ["XLA_FLAGS"]
    again = pcc.enable(c, autotune="dump", environ=environ, jax_module=None)    # same settings in force: recorded no-op
    assert again["applied_via"] == "already" and again["created"] is False
    with pytest.raises(pcc.PccError, match="conflicting"):                      # another directory: refused by name
        pcc.enable(str(tmp_path / "other"), autotune="dump", environ=environ, jax_module=None)
    _populate(c)
    with pytest.raises(pcc.PccError, match="conflicting"):                      # dump pin in force, load requested: refused by name
        pcc.enable(c, environ=environ, jax_module=None)
    assert "autotune dump" in pcc.already_enabled(environ, jax_module=None)


def test_enable_mirrors_into_a_live_uninitialised_jax(tmp_path, monkeypatch):
    c = str(tmp_path / "xla")
    _populate(c, b"populated")
    jax = _stub_jax()
    _bridge(monkeypatch, initialized=False)
    environ = {}
    rec = pcc.enable(c, environ=environ, jax_module=jax)
    assert rec["applied_via"] == "environ+jax.config" and rec["autotune"] == "load" and rec["backend_state"] == "uninitialized"
    assert jax.config.values == {"jax_compilation_cache_dir": c, "jax_persistent_cache_min_compile_time_secs": 0.0,
                                 "jax_persistent_cache_min_entry_size_bytes": 0}
    assert environ["XLA_FLAGS"] == f"{pcc.FLAG_LOAD}={c}/{pcc.AUTOTUNE_FILENAME}"
    assert "jax.config" in pcc.already_enabled({}, jax_module=jax)
    assert pcc.enable(c, environ=environ, jax_module=jax)["applied_via"] == "already"


def test_enable_refuses_when_backends_initialised_and_names_an_unknown_state(tmp_path, monkeypatch):
    c = str(tmp_path / "xla")
    _populate(c)
    jax = _stub_jax()
    _bridge(monkeypatch, initialized=True)
    assert pcc.backend_state(jax) == "initialized"
    with pytest.raises(pcc.PccError, match="backend state is initialized") as ex:
        pcc.enable(c, environ={}, jax_module=jax)
    from opt_core import gates
    assert gates.is_cannot_run(ex.value)                                        # the pin cannot apply: the mode refuses by name
    _bridge(monkeypatch, initialized=None)                                      # this jax cannot say: apply the settings and NAME it
    assert pcc.backend_state(jax) == "unknown"
    environ = {}
    rec = pcc.enable(c, environ=environ, jax_module=jax)
    assert rec["applied_via"] == "environ+jax.config" and rec["backend_state"] == "unknown" and rec["words"] == ["backend_state=unknown"]
    assert environ["JAX_COMPILATION_CACHE_DIR"] == c and jax.config.values["jax_compilation_cache_dir"] == c
    with pytest.raises(pcc.PccError, match="conflicting") as cx:                # a conflicting directory in force: refused (another cache dir would serve)
        pcc.enable(str(tmp_path / "other"), autotune="dump", environ=dict(environ), jax_module=_stub_jax())
    assert gates.is_cannot_run(cx.value)
    live = sys.modules.get("jax")                                              # None = "the live jax": no_jax without one, else that jax's own state
    monkeypatch.delitem(sys.modules, "jax._src.xla_bridge", raising=False) if live is None else monkeypatch.setitem(sys.modules, "jax._src.xla_bridge", importlib.import_module("jax._src.xla_bridge"))
    assert pcc.backend_state(None) == ("no_jax" if live is None else pcc.backend_state(live))
    assert pcc.backend_initialized(None) is (None if live is None else pcc.backend_initialized(live))


# ---------------------------------------------------------------------------------------------------------------------- evidence

def test_identity_key_and_evidence_fields(tmp_path):
    c = tmp_path / "xla"
    assert pcc.identity_key(str(c)) == {"autotune_sha256": None, "cache_listing_sha256": None, "n_cache_entries": 0}
    c.mkdir()
    (c / "exec-aaa").write_bytes(b"1")
    (c / "exec-bbb").write_bytes(b"2")
    (c / pcc.AUTOTUNE_FILENAME).write_bytes(b"tuned")
    ident = pcc.identity_key(str(c))
    assert ident["autotune_sha256"] == hashlib.sha256(b"tuned").hexdigest()
    assert ident["n_cache_entries"] == 2                                        # the autotune file is not a cache entry
    assert ident["cache_listing_sha256"] == hashlib.sha256(b"exec-aaa\nexec-bbb").hexdigest()
    rec = pcc.plan(str(c), environ={})
    assert pcc.evidence_fields(rec) == {"jax_cache": str(c), "autotune": "load", "via": "exports",
                                        "autotune_sha256": ident["autotune_sha256"][:12], "cache_entries": 2}
    assert pcc.evidence_fields(None) == {"jax_cache": "off", "autotune": "off", "via": None, "autotune_sha256": None, "cache_entries": 0}
    assert opt_core.report.kv(**pcc.evidence_fields(None)) == "jax_cache=off autotune=off via=none autotune_sha256=none cache_entries=0"
