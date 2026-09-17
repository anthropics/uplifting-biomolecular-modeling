"""The autoload finder against stub trigger packages: nothing installed without the variable, fires once after the trigger's body,
removes itself, passes the variant, honours the family predicate, exits on a refused activation."""
import os
import sys
import textwrap

import pytest

from opt_core import autoload

CALLS = []
RAISE = None                                           # set by a test: the exception the stub's enable() raises


class _Refused(RuntimeError):
    pass


@pytest.fixture
def stub_tree(tmp_path, monkeypatch):
    """A kit package `acme_opt` (records enable() calls) and a trigger package `vendor.models.acme` on a temporary sys.path."""
    kit = tmp_path / "acme_opt"
    kit.mkdir()
    (kit / "__init__.py").write_text(textwrap.dedent("""
        import tests.test_autoload as t
        class ActivationError(RuntimeError):
            pass
        def enable(mode, variant=None, *, strict=False, trigger=None):
            t.CALLS.append({"mode": mode, "variant": variant, "strict": strict, "trigger": trigger})
            if mode == "refuse":
                raise ActivationError("refused by the stub")
            if t.RAISE is not None:                    # a defect inside enable(): a stale core lacking a name, a broken import …
                raise t.RAISE
            return {"active": True, "words": ["stack=drift(torch:2.13.0!=2.12.1)", "card=uncertified(NVIDIA_L4,22731MiB)"]}
    """))
    vendor = tmp_path / "vendor" / "models" / "acme"
    vendor.mkdir(parents=True)
    (tmp_path / "vendor" / "__init__.py").write_text("")
    (tmp_path / "vendor" / "models" / "__init__.py").write_text("")
    (vendor / "__init__.py").write_text("import tests.test_autoload as t\nt.CALLS.append('trigger body')\nVALUE = 1\n")
    other = tmp_path / "runner"
    other.mkdir()
    (other / "__init__.py").write_text("GENERIC = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    CALLS.clear()
    global RAISE
    RAISE = None
    saved = list(sys.meta_path)
    yield tmp_path
    sys.meta_path[:] = saved
    for m in list(sys.modules):
        if m == "acme_opt" or m.startswith("vendor") or m == "runner":
            del sys.modules[m]


def spec(**kw):
    base = dict(env="ACME_OPT", package="acme_opt", tag="acme-opt", triggers=("vendor.models.acme",), modes=("exact", "fast", "refuse", "off"))
    base.update(kw)
    return autoload.AutoloadSpec(**base)


def test_spec_validation():
    with pytest.raises(ValueError):
        spec(modes=("exact", "fast"))
    with pytest.raises(ValueError):
        spec(triggers=())


def test_nothing_installed_without_the_variable(stub_tree):
    assert autoload.install(spec(), environ={}) is None
    assert autoload.install(spec(), environ={"ACME_OPT": "off"}) is None
    assert autoload.install(spec(), environ={"ACME_OPT": "  OFF "}) is None
    assert autoload.installed(spec()) is None


def test_unknown_mode_refuses_at_the_trigger_by_default(stub_tree, capsys):
    """A mistyped selection is a refusal, never a note: the trigger refuses with the kit's own line and exit code (the default)."""
    f = autoload.install(spec(), environ={"ACME_OPT": "turbo"})
    assert f is not None and f.refuse and capsys.readouterr().err == ""
    with pytest.raises(SystemExit) as e:
        import vendor.models.acme  # noqa: F401
    assert e.value.code == 3 and CALLS == ["trigger body"]
    assert capsys.readouterr().err.strip() == "[acme-opt] NOT ACTIVE: unknown ACME_OPT='turbo' (expected exact|fast|refuse|off)"


def test_unknown_mode_can_refuse_at_once_with_the_kits_line(stub_tree, capsys):
    """on_unknown='exit': the kit's own line bytes (line_of), the kit's exit code, nothing installed."""
    with pytest.raises(SystemExit) as e:
        autoload.install(spec(on_unknown="exit"), environ={"ACME_OPT": "turbo"})
    assert e.value.code == 3 and autoload.installed(spec()) is None
    assert capsys.readouterr().err.strip() == "[acme-opt] NOT ACTIVE: unknown ACME_OPT='turbo' (expected exact|fast|refuse|off)"
    with pytest.raises(SystemExit) as e:
        autoload.install(spec(on_unknown="exit", exit_not_active=5, line_of=lambda what, value: f"[acme-opt] NOT ACTIVE: {what} {value} is not a line name"),
                         environ={"ACME_OPT": "S3"})
    assert e.value.code == 5 and capsys.readouterr().err.strip() == "[acme-opt] NOT ACTIVE: mode S3 is not a line name"


def test_unknown_selection_is_matched_verbatim(stub_tree, capsys):
    """Case-folding only onto a DECLARED mode, and only when the spec folds; a variant is verbatim against the declared names."""
    f = autoload.install(spec(), environ={"ACME_OPT": " FAST "})             # folds onto the declared 'fast'
    assert f is not None and f.mode == "fast"
    autoload.disarm(spec())
    with pytest.raises(SystemExit):
        autoload.install(spec(fold_mode=False, on_unknown="exit"), environ={"ACME_OPT": "FAST"})   # the kit matches verbatim: FAST is not declared
    assert "unknown ACME_OPT='FAST'" in capsys.readouterr().err
    with pytest.raises(SystemExit):                                          # declared variants: 'S3' is not 's3'
        autoload.install(spec(variant_env="ACME_LINE", variants=("s3", "s4"), on_unknown="exit"), environ={"ACME_OPT": "fast", "ACME_LINE": "S3"})
    assert capsys.readouterr().err.strip() == "[acme-opt] NOT ACTIVE: unknown ACME_LINE='S3' (expected s3|s4)"
    assert autoload.installed(spec()) is None
    f = autoload.install(spec(variant_env="ACME_LINE", variants=("s3", "s4")), environ={"ACME_OPT": "fast", "ACME_LINE": "S3"})
    assert f is not None and f.refuse == "[acme-opt] NOT ACTIVE: unknown ACME_LINE='S3' (expected s3|s4)"   # default: refuses at the trigger
    autoload.disarm(spec())
    f = autoload.install(spec(variant_env="ACME_LINE", variants=("s3", "s4")), environ={"ACME_OPT": "fast", "ACME_LINE": " s3 "})
    assert f is not None and f.selection == {"mode": "fast", "variant": "s3"}


def test_unknown_selection_can_refuse_at_the_trigger(stub_tree, capsys):
    """on_unknown='refuse_at_trigger': the finder is armed; a probe does not fire it; the real import refuses with the kit's line."""
    import importlib.util
    f = autoload.install(spec(on_unknown="refuse_at_trigger"), environ={"ACME_OPT": "turbo"})
    assert f is not None and f.refuse and f is sys.meta_path[0] and capsys.readouterr().err == ""
    assert importlib.util.find_spec("vendor.models.acme") is not None and f.armed and f.fired is None
    with pytest.raises(SystemExit) as e:
        import vendor.models.acme  # noqa: F401
    assert e.value.code == 3 and CALLS == ["trigger body"]                    # the trigger's body ran; the kit's enable() never did
    assert capsys.readouterr().err.strip() == "[acme-opt] NOT ACTIVE: unknown ACME_OPT='turbo' (expected exact|fast|refuse|off)"
    assert f not in sys.meta_path


def test_note_and_continue_is_an_explicit_opt_in(stub_tree, capsys):
    assert autoload.install(spec(on_unknown="note"), environ={"ACME_OPT": "turbo"}) is None
    assert capsys.readouterr().err.strip() == "[acme-opt] NOT ACTIVE: unknown ACME_OPT='turbo' (expected exact|fast|refuse|off)"
    with pytest.raises(ValueError):
        spec(on_unknown="ignore")
    with pytest.raises(ValueError):
        spec(variants=("a",))                                                 # variants need variant_env


def test_finder_survives_a_find_spec_probe(stub_tree):
    """A bare importlib.util.find_spec on the trigger (a probe) leaves the finder armed; the import that follows is still hooked."""
    import importlib.util
    f = autoload.install(spec(), environ={"ACME_OPT": "fast"})
    for _ in range(2):
        probe = importlib.util.find_spec("vendor.models.acme")
        assert probe is not None and f.armed and f.fired is None and f in sys.meta_path
    assert CALLS == [] and "vendor.models.acme" not in sys.modules
    import vendor.models.acme  # noqa: F401
    assert CALLS == ["trigger body", {"mode": "fast", "variant": None, "strict": True, "trigger": "vendor.models.acme"}]
    assert f.fired == "vendor.models.acme" and f not in sys.meta_path


def test_fires_once_even_when_the_trigger_is_reloaded(stub_tree):
    import importlib
    autoload.install(spec(), environ={"ACME_OPT": "fast"})
    import vendor.models.acme as trig
    importlib.reload(trig)
    assert CALLS.count("trigger body") == 2 and sum(isinstance(c, dict) for c in CALLS) == 1


def test_loader_is_wrapped_not_patched(stub_tree):
    """The trigger's own loader object is left untouched (a shared loader — a zip importer — must never be patched in place)."""
    import importlib.machinery
    import importlib.util
    autoload.install(spec(), environ={"ACME_OPT": "fast"})
    wrapped = importlib.util.find_spec("vendor.models.acme")
    inner = wrapped.loader._loader
    assert isinstance(wrapped.loader, autoload._Loader) and isinstance(inner, importlib.machinery.SourceFileLoader)
    assert "exec_module" not in vars(inner) and wrapped.loader.get_filename() == inner.get_filename()


def test_fires_once_after_the_trigger_body_and_removes_itself(stub_tree):
    f = autoload.install(spec(), environ={"ACME_OPT": "Exact"})
    assert f is sys.meta_path[0] and f.armed and f.fired is None
    assert autoload.install(spec(), environ={"ACME_OPT": "exact"}) is f      # idempotent
    import vendor.models.acme as trig                                       # noqa: F401
    assert CALLS == ["trigger body", {"mode": "exact", "variant": None, "strict": True, "trigger": "vendor.models.acme"}]
    assert f.fired == "vendor.models.acme" and f not in sys.meta_path
    assert "acme_opt" in sys.modules


def test_variant_from_the_variant_variable(stub_tree):
    """An undeclared variant set is passed verbatim (stripped) to the kit's enable(), which judges it."""
    autoload.install(spec(variant_env="ACME_VARIANT"), environ={"ACME_OPT": "fast", "ACME_VARIANT": " Full_MSA "})
    import vendor.models.acme  # noqa: F401
    assert CALLS[-1] == {"mode": "fast", "variant": "Full_MSA", "strict": True, "trigger": "vendor.models.acme"}


def test_family_predicate_refuses_a_generic_trigger(stub_tree):
    accepted = []

    def accept(fullname, found_spec):
        accepted.append((fullname, os.path.basename(os.path.dirname(found_spec.origin))))
        return False

    f = autoload.install(spec(triggers=("runner",), accept=accept), environ={"ACME_OPT": "exact"})
    import runner  # noqa: F401
    assert accepted == [("runner", "runner")] and f.armed and f in sys.meta_path and CALLS == []


def test_refused_activation_exits_the_process(stub_tree):
    autoload.install(spec(exit_not_active=7), environ={"ACME_OPT": "refuse"})
    with pytest.raises(SystemExit) as e:
        import vendor.models.acme  # noqa: F401
    assert e.value.code == 7 and CALLS[-1]["mode"] == "refuse"


def test_disarm_before_the_stock_route(stub_tree):
    f = autoload.install(spec(), environ={"ACME_OPT": "exact"})
    assert autoload.disarm(spec()) is True and f not in sys.meta_path and not f.armed
    assert autoload.disarm(spec()) is False
    import vendor.models.acme  # noqa: F401
    assert CALLS == ["trigger body"]


def test_module_imports_only_os_and_sys():
    import ast
    src = open(autoload.__file__).read()
    names = [n.names[0].name.split(".")[0] for n in ast.parse(src).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert names == ["os", "sys"]


def test_exit_now_refuses_with_the_kits_code_from_a_pth(tmp_path):
    """on_unknown='exit_now' at .pth time: the kit's line, the kit's exit code (not the interpreter's init status 1), no stock run."""
    import subprocess
    site = tmp_path / "site"
    site.mkdir()
    (site / "acme_opt").mkdir()
    (site / "acme_opt" / "__init__.py").write_text(textwrap.dedent("""
        import sys
        class ActivationError(RuntimeError):
            pass
        def enable(mode, variant=None, *, strict=False, trigger=None):
            sys.stderr.write(f"[acme-opt] ACTIVE mode={mode} trigger={trigger}\\n")
    """))
    (site / "acme_opt" / "_autoload.py").write_text(textwrap.dedent("""
        import os, sys
        from opt_core.autoload import AutoloadSpec, install
        FINDER = install(AutoloadSpec(env="ACME_OPT", package="acme_opt", tag="acme-opt", triggers=("json.tool",), modes=("exact", "fast", "off"),
                                      exit_not_active=3, on_unknown=os.environ.get("ACME_ON_UNKNOWN", "exit_now")))
    """))
    (site / "acme_opt_autoload.pth").write_text("import acme_opt._autoload\n")
    core_dir = os.path.dirname(os.path.dirname(os.path.abspath(autoload.__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(site), core_dir]), ACME_OPT="turbo", PYTHONDONTWRITEBYTECODE="1")
    code = "import site, sys; site.addsitedir(%r); import json.tool; print('RAN ANYWAY')" % str(site)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "RAN ANYWAY" not in r.stdout, (r.returncode, r.stdout, r.stderr)
    assert r.stderr.strip().splitlines()[-1] == "[acme-opt] NOT ACTIVE: unknown ACME_OPT='turbo' (expected exact|fast|off)"
    r = subprocess.run([sys.executable, "-c", code], env=dict(env, ACME_ON_UNKNOWN="refuse_at_trigger"), capture_output=True, text=True)
    assert r.returncode == 3 and "RAN ANYWAY" not in r.stdout and "unknown ACME_OPT='turbo'" in r.stderr
    r = subprocess.run([sys.executable, "-c", code], env=dict(env, ACME_OPT="fast"), capture_output=True, text=True)
    assert r.returncode == 0 and "RAN ANYWAY" in r.stdout and "[acme-opt] ACTIVE mode=fast trigger=json.tool" in r.stderr


def test_exit_now_in_process_calls_os_exit_after_flushing(stub_tree, capsys, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "_exit", lambda code: calls.append(code))
    autoload.install(spec(on_unknown="exit_now"), environ={"ACME_OPT": "turbo"})
    assert calls == [3] and capsys.readouterr().err.strip() == "[acme-opt] NOT ACTIVE: unknown ACME_OPT='turbo' (expected exact|fast|refuse|off)"


# ------------------------------------------------------------------------------------------------------------ pre-body hook


def test_pre_body_runs_once_before_the_trigger_body_with_the_selection_enable_receives(stub_tree):
    seen = []
    hook = lambda trigger, sel: (seen.append((trigger, dict(sel))), CALLS.append("pre_body"))  # noqa: E731
    f = autoload.install(spec(variant_env="ACME_VARIANT", pre_body=hook), environ={"ACME_OPT": "fast", "ACME_VARIANT": "v2"})
    import vendor.models.acme as trig                                       # noqa: F401
    assert CALLS == ["pre_body", "trigger body", {"mode": "fast", "variant": "v2", "strict": True, "trigger": "vendor.models.acme"}]
    assert seen == [("vendor.models.acme", {"mode": "fast", "variant": "v2"})] and f.pre_fired == "vendor.models.acme"
    import importlib
    importlib.reload(trig)                                                  # a reload: the body runs again, neither the hook nor enable does
    assert CALLS.count("pre_body") == 1 and sum(1 for c in CALLS if isinstance(c, dict)) == 1


def test_pre_body_is_skipped_off_the_activation_path(stub_tree, capsys):
    ran = []
    hook = lambda trigger, sel: ran.append(trigger)  # noqa: E731
    assert autoload.install(spec(pre_body=hook), environ={}) is None        # unset: nothing installed, nothing runs
    f = autoload.install(spec(pre_body=hook), environ={"ACME_OPT": "bogus"})  # unknown mode: the trigger refuses; the hook never runs
    with pytest.raises(SystemExit) as ex:
        import vendor.models.acme  # noqa: F401
    assert ex.value.code == 3 and ran == [] and "NOT ACTIVE: unknown ACME_OPT='bogus'" in capsys.readouterr().err
    f.remove()


def test_pre_body_is_skipped_after_disarm(stub_tree):
    ran = []
    autoload.install(spec(pre_body=lambda t, s: ran.append(t)), environ={"ACME_OPT": "exact"})
    assert autoload.disarm(spec()) is True
    import vendor.models.acme  # noqa: F401
    assert ran == [] and CALLS == ["trigger body"]


def test_a_failing_pre_body_refuses_with_the_kits_line_and_code(stub_tree, capsys):
    def hook(trigger, sel):
        raise KeyError("RFX")
    autoload.install(spec(pre_body=hook), environ={"ACME_OPT": "exact"})
    with pytest.raises(SystemExit) as ex:
        import vendor.models.acme  # noqa: F401
    assert ex.value.code == 3
    assert "[acme-opt] NOT ACTIVE: pre-body hook failed before vendor.models.acme: KeyError('RFX')" in capsys.readouterr().err
    assert CALLS == []                                                      # neither the body nor enable ran


def test_pre_body_must_be_callable():
    with pytest.raises(ValueError):
        spec(pre_body="not callable")


# ------------------------------------------------------------------------------------------------------------ patch_attr_at_import


@pytest.fixture
def stock_mod(tmp_path, monkeypatch):
    """A stock module `stockmod` (function f, class C with method m) on a temporary sys.path; patches and modules cleaned up."""
    (tmp_path / "stockmod.py").write_text("def f(x):\n    return x + 1\n\nclass C:\n    def m(self, x):\n        return x * 2\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    saved = list(sys.meta_path)
    yield tmp_path
    sys.meta_path[:] = saved
    autoload._PATCHES.clear()
    sys.modules.pop("stockmod", None)


def _wrap10(orig):
    def w(*a, **k):
        return orig(*a, **k) * 10
    return w


def test_patch_armed_then_installed_at_import(stock_mod):
    p = autoload.patch_attr_at_import("stockmod", "C.m", _wrap10, tag="acme-opt", name="tenfold")
    assert p.state == "armed" and p in sys.meta_path and p.pairs() == [("patch", "armed"), ("site", "stockmod:C.m")]
    import stockmod
    assert p.state == "installed" and p not in sys.meta_path and stockmod.C().m(2) == 40 and p.original(stockmod.C(), 2) == 4
    assert autoload.patch_attr_at_import("stockmod", "C.m", _wrap10, tag="acme-opt") is p       # same site, same factory: no-op
    assert p.record() == {"name": "tenfold", "target": "stockmod", "attr": "C.m", "state": "installed", "error": None}


def test_patch_now_when_already_imported_and_repatch_wraps_the_original(stock_mod):
    import stockmod
    p = autoload.patch_attr_at_import("stockmod", "f", _wrap10, tag="acme-opt")
    assert p.state == "installed" and stockmod.f(1) == 20
    p2 = autoload.patch_attr_at_import("stockmod", "f", lambda orig: (lambda x: orig(x) + 100), tag="acme-opt")
    assert p2 is p and stockmod.f(1) == 102                                  # 100 + orig(1)=2: the original was re-wrapped, not the x10 wrapper
    p.restore()
    assert stockmod.f(1) == 2 and p.state == "disarmed"


def test_missing_attribute_is_a_named_refusal_now_or_an_exit_at_import(stock_mod, capsys):
    import stockmod  # noqa: F401
    with pytest.raises(autoload.PatchError, match=r"\[acme-opt\] NOT ACTIVE: lever needs stockmod\.C\.gone, not found"):
        autoload.patch_attr_at_import("stockmod", "C.gone", _wrap10, tag="acme-opt", name="lever")
    sys.modules.pop("stockmod"); autoload._PATCHES.clear()
    p = autoload.patch_attr_at_import("stockmod", "nothere", _wrap10, tag="acme-opt", name="lever", exit_not_active=3)
    assert p.state == "armed"
    with pytest.raises(SystemExit) as ex:
        import stockmod  # noqa: F401, F811
    assert ex.value.code == 3 and "[acme-opt] NOT ACTIVE: lever needs stockmod.nothere, not found" in capsys.readouterr().err


def test_a_raising_factory_is_a_patch_error(stock_mod):
    import stockmod  # noqa: F401
    def bad(orig):
        raise KeyError("cells")
    with pytest.raises(autoload.PatchError, match="could not be built"):
        autoload.patch_attr_at_import("stockmod", "f", bad, tag="acme-opt")


def test_disarm_before_import_leaves_stock(stock_mod):
    p = autoload.patch_attr_at_import("stockmod", "f", _wrap10, tag="acme-opt")
    assert p.disarm() is True and p.state == "disarmed" and p not in sys.meta_path
    import stockmod
    assert stockmod.f(1) == 2
    p2 = autoload.patch_attr_at_import("stockmod", "f", _wrap10, tag="acme-opt")                # re-planned after a disarm: patches now
    assert p2 is p and p.state == "installed" and stockmod.f(1) == 20


def test_two_patches_armed_on_one_unimported_module_both_install_at_its_import(stock_mod):
    # several armed hooks on the same not-yet-imported module recursed without end in _real_spec; the first hook serves the
    # import and fires the others
    p1 = autoload.patch_attr_at_import("stockmod", "C.m", _wrap10, tag="acme-opt", name="tenfold")
    p2 = autoload.patch_attr_at_import("stockmod", "f", lambda orig: (lambda x: orig(x) + 100), tag="acme-opt", name="plus100")
    p3 = autoload.patch_attr_at_import("stockmod", "nothere_ok", _wrap10, tag="acme-opt", name="third")
    p3.disarm()                                                              # a disarmed hook on the same module is not fired
    assert p1.state == p2.state == "armed" and p1 in sys.meta_path and p2 in sys.meta_path
    import importlib.util
    assert importlib.util.find_spec("stockmod") is not None                  # a probe recursed here before; both stay armed after it
    assert p1.state == p2.state == "armed"
    import stockmod
    assert (p1.state, p2.state, p3.state) == ("installed", "installed", "disarmed")
    assert stockmod.C().m(2) == 40 and stockmod.f(1) == 102
    assert p1 not in sys.meta_path and p2 not in sys.meta_path


@pytest.mark.parametrize("exc", [ImportError("cannot import name 'apply' from 'opt_core.mem'"), AttributeError("module 'opt_core.mem' has no attribute 'compose_big'"),
                                 RuntimeError("engine refused")], ids=["ImportError", "AttributeError", "RuntimeError"])
def test_any_exception_from_enable_is_a_named_not_active_exit_never_a_traceback(stub_tree, capsys, exc):
    global RAISE
    RAISE = exc
    autoload.install(spec(exit_not_active=7), environ={"ACME_OPT": "exact"})
    with pytest.raises(SystemExit) as e:
        import vendor.models.acme  # noqa: F401
    assert e.value.code == 7 and CALLS[-1]["mode"] == "exact"
    err = capsys.readouterr().err.strip().splitlines()
    assert err == [f"[acme-opt] NOT ACTIVE: enable() raised at vendor.models.acme: {type(exc).__name__}: {exc}"], err


def test_an_activation_under_named_uncertainty_returns_and_the_program_continues(stub_tree, capsys):
    """P12: environment words (drifted stack, uncertified card) are the kit's ACTIVE line's business; enable() RETURNS under strict=True and
    the finder lets the program run — no exit, nothing printed by the finder."""
    autoload.install(spec(exit_not_active=7), environ={"ACME_OPT": "fast"})
    import vendor.models.acme  # noqa: F401
    assert vendor.models.acme.VALUE == 1 and CALLS == ["trigger body", {"mode": "fast", "variant": None, "strict": True, "trigger": "vendor.models.acme"}]
    assert capsys.readouterr().err == ""


def test_a_cannot_run_event_from_enable_refuses_the_mode_by_name(stub_tree, capsys):
    """opt_core.gates outcome (b): an exception carrying cannot_run=True is the MODE's refusal, named as such, the kit's exit code."""
    from opt_core import gates
    global RAISE
    RAISE = gates.cannot_run(RuntimeError("fpf_trimul_v4: none:CompilationError"))
    autoload.install(spec(exit_not_active=7), environ={"ACME_OPT": "fast"})
    with pytest.raises(SystemExit) as e:
        import vendor.models.acme  # noqa: F401
    assert e.value.code == 7
    err = capsys.readouterr().err.strip().splitlines()
    assert err == ["[acme-opt] NOT ACTIVE: mode fast refused at vendor.models.acme: RuntimeError: fpf_trimul_v4: none:CompilationError"], err


def test_a_kit_package_that_cannot_be_imported_is_a_named_not_active_exit(stub_tree, capsys):
    (stub_tree / "acme_opt" / "__init__.py").write_text("import no_such_core_module_acme  # noqa\n")
    sys.modules.pop("acme_opt", None)
    autoload.install(spec(exit_not_active=7), environ={"ACME_OPT": "exact"})
    with pytest.raises(SystemExit) as e:
        import vendor.models.acme  # noqa: F401
    assert e.value.code == 7
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("[acme-opt] NOT ACTIVE: acme_opt could not be imported at vendor.models.acme: ModuleNotFoundError: "), err
