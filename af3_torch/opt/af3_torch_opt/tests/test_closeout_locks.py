"""Close-out locks: the KERNELS line's arch fields, the kit's last bucket travelling on the model process's argv (never a literal in
forward.py), the pin table's bare form, the unexpected-fallback classification for triangle-attention's serve dtype, the .pth is one line
and pyproject ships this package, and featurise.py's bare-interpreter bucket literal agrees with the kit's own row."""
import os
import re

from af3_torch_opt import report

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))
HOME = os.path.dirname(OPT)


def _read(*rel):
    with open(os.path.join(HOME, *rel), encoding="utf-8") as f:
        return f.read()


def test_arch_fields_from_the_census():
    assert report.arch_fields(None) == {}
    assert report.arch_fields({"arch": "not-a-dict"}) == {}
    tuned = {"arch": {"cc": "10.0", "notices": []}}
    assert report.arch_fields(tuned) == {"arch": "10.0", "cells": "tuned"}
    builtin = {"arch": {"cc": "9.0", "notices": ["flash_triattn: FALLBACK CONFIG (no tuned cell for cc 9.0)"]}}
    assert report.arch_fields(builtin) == {"arch": "9.0", "cells": "builtin"}
    text = report.kernels_line({"fpf_trimul_v4": {"imported_from": "/c/k/fpf_trimul_v4/__init__.py", "core_copy": "/c/k", "version": "4.1.0", "ok": True}}, ["fpf_trimul_v4"], builtin)
    assert text.endswith("arch=9.0 cells=builtin"), text
    assert " ok=1 " in text


def test_pin_table_is_bare():
    """[tool.opt_core] carries exactly path / version, no comment on the header or on a value line (the core's readers
    and the release's re-pin touch these two lines only)."""
    text = _read("opt", "pyproject.toml")
    i = text.index("[tool.opt_core]")
    block = text[i:].split("\n\n")[0].splitlines()
    assert block[0] == "[tool.opt_core]"
    keys = []
    for line in block[1:]:
        assert "#" not in line, line
        k, _, v = line.partition(" = ")
        assert v.startswith('"') and v.endswith('"'), line
        keys.append(k)
    assert keys == ["path", "version"], keys
    assert re.search(r'^requires = \["setuptools>=64", "wheel"\]$', text, re.M)


def test_at05_triattn_served_dtype_fallback_is_unexpected():
    """AT-05: the eval census must show the triangle-attention kernel SERVED in 16-bit (served_fp32 == 0). The carried adapter serves
    flash_triattn only under bf16 autocast and otherwise takes the stock path counted as `fallback:no-bf16-autocast`
    (opt/forward/af3t/kernels/af3_kernels.py) — that counter is NOT in the kernels' declared coverage (registry.EXPECTED_FALLBACKS), so one
    such event splits as UNEXPECTED and the pred is PARTIAL (cli.cmd_pred: an unexpected fallback is an `events` entry, exit 1), never a
    silent fp32 run under the mode's name. The same holds for trimul's `no-bf16-autocast` and for any kernel_error."""
    import os, re
    from af3_torch_opt import registry, stack
    for lever in ("triattn", "trimul"):
        assert "fallback:no-bf16-autocast" not in registry.EXPECTED_FALLBACKS.get(lever, ()), lever
        exp, unexp = registry.fallback_split({lever: {"fallback:no-bf16-autocast": 1}})
        assert exp == {} and unexp == {lever: {"fallback:no-bf16-autocast": 1}}, (exp, unexp)
    exp, unexp = registry.fallback_split({"triattn": {"kernel_error:RuntimeError": 1}, "trimul": {"fallback:c=64": 40}})
    assert exp == {"trimul": {"fallback:c=64": 40}} and unexp == {"triattn": {"kernel_error:RuntimeError": 1}}, (exp, unexp)
    assert registry.EXPECTED_FALLBACKS == {"trimul": ("fallback:c=64", "fallback:N<{lt}", "fallback:stock_row:{row}"), "tmpl_trimul": ("fallback:stock_row:{row}",),
                                           "triattn": ("fallback:N<{lt}", "fallback:refused:{row}:stock_row"),
                                           "transition": ("fallback:c=384,stock-row", "fallback:c=128,stock-row", "fallback:c=64,stock-row", "fallback:c=64,no_cell"),
                                           "glu_proj": ("fallback:c=384", "fallback:c=128,rows<16384", "fallback:c=64,rows<16384", "fallback:c=64,card-off", "fallback:superseded:transition"),
                                           "trimul_exact": ("fallback:unvouched", "fallback:superseded:trimul", "fallback:stock_row:{row}"),
                                           "opm": ("fallback:arch=sm_{sm}",)}   # the enumerated declared coverage: a new expected word is a change to this line
    src = open(os.path.join(stack.forward_dir(), "af3t", "kernels", "af3_kernels.py"), encoding="utf-8").read()   # the counter's spelling is the adapter's
    assert len(re.findall(r'_fallback\(lever, "no-bf16-autocast"\)', src)) >= 2, "af3_kernels.py no longer counts no-bf16-autocast the way registry.EXPECTED_FALLBACKS judges it"
    cli_src = open(os.path.join(os.path.dirname(registry.__file__), "cli.py"), encoding="utf-8").read()
    assert 'if unexpected or rep["dead"]:' in cli_src and '"kind": "fallback"' in cli_src        # the gate that turns it into PARTIAL


# ---- the .pth is one line and pyproject ships this package; featurise.py's bare-interpreter bucket literal agrees with the kit's own row ----

def test_pth_and_pyproject():
    pth = open(os.path.join(HOME, "opt", "af3_torch_opt_autoload.pth")).read()
    import importlib.util as _ilu                                                       # the .pth is the build backend's output: the plain import line (template
    spec = _ilu.spec_from_file_location("_bb", os.path.join(HOME, "opt", "_build_backend.py")); bb = _ilu.module_from_spec(spec); spec.loader.exec_module(bb)   # without pth_text) or the guarded form
    if hasattr(bb, "pth_fields"):
        assert bb.pth_fields(pth) == ("af3_torch_opt", "AF3_TORCH_OPT", "af3-torch-opt", 3), pth
    else:
        assert pth.strip().splitlines() == ["import af3_torch_opt._autoload"], pth
    py = open(os.path.join(HOME, "opt", "pyproject.toml"), encoding="utf-8").read()
    assert 'name = "af3_torch_opt"' in py and 'packages = ["af3_torch_opt", "af3_torch_opt.tests"]' in py


def test_featurise_default_buckets_is_no_padding():
    """featurise.py runs on the image's JAX interpreter with no package on its path, so its --buckets default is its own literal: `none`
    (xfold as shipped: featurise_input(buckets=None)); the modes' rows reach it from cli.bucket_row (none | tile:<k>) — held here."""
    src = open(os.path.join(HOME, "opt", "af3_torch_opt", "featurise.py"), encoding="utf-8").read()
    assert re.search(r'^NO_BUCKETS = "none"', src, re.M) and 'add_argument("--buckets", default=NO_BUCKETS' in src
    assert "DEFAULT_BUCKETS" not in src and re.search(r'^TILE_BUCKETS = "tile:"', src, re.M)


def test_e14_size_gate_words_are_the_kernels_spellings():
    """registry.EXPECTED_FALLBACKS judges the size gates by the census WORD the kernels print (`fallback:N<{lt}`: the threshold is read
    from the word, registry.size_gate_threshold). Lock the two spellings at their source so a respelling fails here, not as a
    REFUSED:fallback on every small item: the shared core's fused trimul raises `TrimulUnsupported("N<%d" % nmin, …)` under `if N < nmin`,
    where `nmin` is `N_MIN` or the caller's per-call floor (opt_core/kernels/fpf_trimul_v4/generic.py — the adapter counts e.reason as
    `fallback:<reason>`), and the carried triangle-attention
    adapter counts `_fallback(lever, "N<16")` under `if N < 16` (opt/forward/af3t/kernels/af3_kernels.py)."""
    import os
    import opt_core
    from af3_torch_opt import registry, stack
    gen = open(os.path.join(os.path.dirname(opt_core.__file__), "kernels", "fpf_trimul_v4", "generic.py"), encoding="utf-8").read()
    assert 'nmin = N_MIN if n_min is None else int(n_min)' in gen and 'if N < nmin:' in gen and 'raise TrimulUnsupported("N<%d" % nmin, "N=%d" % N)' in gen, "fpf_trimul_v4 generic.py no longer spells its small-N gate N<%d — registry.EXPECTED_FALLBACKS trimul size gate"
    assert 'N_MIN = int(os.environ.get("FPF_TRIMUL_V4_NMIN", "101"))' in gen
    ak = open(os.path.join(stack.forward_dir(), "af3t", "kernels", "af3_kernels.py"), encoding="utf-8").read()
    assert 'if N < 16:\n        _fallback(lever, "N<16")' in ak, "af3_kernels.py no longer spells triattn's small-N gate N<16 — registry.EXPECTED_FALLBACKS triattn size gate"
    assert 'sel = _trimul_select(lever_c, word, N, C, direction, prec)' in ak                      # the trimul adapter asks the provider per size: the v4 row's floor is the provider's cell boundary now (the key stays declared)
    assert registry.size_gate_threshold("trimul", "fallback:N<101") == 101 and registry.size_gate_threshold("triattn", "fallback:N<16") == 16
