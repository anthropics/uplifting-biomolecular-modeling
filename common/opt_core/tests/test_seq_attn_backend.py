"""opt_core.seq.attn_backend: the pin probes metadata only, names every problem, and prints one fragment."""
from __future__ import annotations

from opt_core.seq import attn_backend as ab


def pin(**over):
    kw = dict(engine_tag="kit", backend="flash_attn2_varlen", versions={"flash_attn": "2.7.4.post1", "torch": None},
              modules=("json", "importlib.machinery"), forbidden=("xformers",), arch={"sm90"})
    kw.update(over)
    return ab.declare(**kw)


def test_probe_ok_and_line():
    p = pin()
    g = p.probe(gpu={"sm": "sm90"}, found={"flash_attn": "2.7.4.post1", "torch": "2.8.0", "xformers": None})
    assert g.ok, g.reason
    assert g.name == ab.GATE_NAME == "attn_backend"
    assert g.details["modules"] == {"json": True, "importlib.machinery": True}
    line = p.line(gpu={"cc": "9.0"}, found={"flash_attn": "2.7.4.post1", "torch": "2.8.0"})
    assert line == "attn_backend=flash_attn2_varlen flash_attn=2.7.4.post1 torch=2.8.0 sm=sm90", line
    assert p.line_fields(gpu=None, found={}) == {"attn_backend": "flash_attn2_varlen", "flash_attn": "none", "torch": "none", "sm": "none"}


def test_every_problem_is_named():
    p = pin(modules=("json", "no_such_module_xyz.sub"))
    g = p.probe(gpu={"sm": "sm100"}, found={"flash_attn": "2.8.1", "torch": None, "xformers": "0.0.35"})
    assert not g.ok
    r = g.reason
    assert r.startswith("attention backend — ")
    assert "flash_attn==2.8.1, the certificate holds for flash_attn==2.7.4.post1" in r
    assert "torch not installed" in r
    assert "xformers==0.0.35 is installed" in r
    assert "module no_such_module_xyz.sub not importable" in r
    assert "GPU is sm100; the flash_attn2_varlen path is certified on sm90" in r
    assert len(g.details["problems"]) == 5


def test_arch_empty_means_any_card_and_sm_unknown_named():
    assert pin(arch=()).probe(gpu=None, found={"flash_attn": "2.7.4.post1", "torch": "x", "xformers": None}).ok
    g = pin().probe(gpu={}, found={"flash_attn": "2.7.4.post1", "torch": "x", "xformers": None})
    assert not g.ok and "GPU sm unknown" in g.reason


def test_module_present_never_imports(monkeypatch):
    import sys
    sys.modules.pop("this_is_not_installed_anywhere", None)
    assert ab.module_present("json")
    assert ab.module_present("importlib.util")
    assert not ab.module_present("this_is_not_installed_anywhere")
    assert not ab.module_present("json.no_such_child")
    assert "email.mime" not in sys.modules or True   # resolution goes through PathFinder, not import
    before = set(sys.modules)
    ab.module_present("email.mime.text")
    assert not ({"email.mime.text"} & (set(sys.modules) - before))
