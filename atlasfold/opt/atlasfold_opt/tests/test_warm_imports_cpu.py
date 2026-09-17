"""warm_imports — the one start-up call of opt_core.warm_imports() in stack.activate and its NOTE line (CPU: the core call is stubbed)."""
from atlasfold_opt import stack as S


def test_note_line_reports_the_core_words(monkeypatch, capsys):
    import opt_core
    calls = []
    monkeypatch.setattr(opt_core, "warm_imports", lambda **k: calls.append(k) or {"cuequivariance_ops_torch": 1.234, "cuequivariance_torch": "present"}, raising=False)
    rep = S.warm_imports("[atlasfold-opt]")
    err = capsys.readouterr().err
    assert calls == [{"libraries": ("torch", "cuequivariance_ops_torch", "cuequivariance_torch"), "origin": "atlasfold_opt.activate"}]   # torch first (load order)
    assert S.WARM_LIBRARIES[0] == "torch"
    assert rep == {"cuequivariance_ops_torch": 1.234, "cuequivariance_torch": "present"}
    assert "[atlasfold-opt] NOTE warm_imports=cuequivariance_ops_torch:1.234,cuequivariance_torch:present origin=activate" in err


def test_an_older_core_is_named_never_fatal(monkeypatch, capsys):
    import opt_core
    def boom(**k): raise AttributeError("no warm_imports")
    monkeypatch.setattr(opt_core, "warm_imports", boom, raising=False)
    assert S.warm_imports("[atlasfold-opt]") == {}
    assert "NOTE warm_imports=unavailable:AttributeError origin=activate" in capsys.readouterr().err


def test_the_real_call_is_idempotent_and_returns_words():
    import opt_core
    a = dict(opt_core.warm_imports(origin="test")); b = dict(opt_core.warm_imports(origin="test"))
    assert a == b                                             # decided once per process; CPU hosts read absent / present words
