"""CPU contract tests (no GPU, no weights): mode table, literal MODE_NAMES line, byte-identical kit_template copies, generated .pth text, PINS schema."""
import ast, hashlib, json, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); OPT = os.path.dirname(PKG); KIT = os.path.dirname(OPT)
CORE = os.path.normpath(os.path.join(OPT, "..", "..", "common", "opt_core"))

def _sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def test_mode_table():
    from atlasfold_opt import modes, registry
    assert tuple(sorted(modes.MODES)) == tuple(sorted(modes.MODE_NAMES))
    assert modes.MODE_NAMES[0] == "off"
    for m, levers in modes.MODES.items():
        for l in levers: assert l in registry.LEVERS, (m, l)
    assert modes.MODE_NAMES == ("off", "exact", "fast", "big")   # the kit ships four modes
    ex, fa, bg = (set(modes.MODES[m]) for m in ("exact", "fast", "big"))
    assert ex - fa == {"trimul_exact"} and fa - bg == {"denoiser_graph", "graph_reuse"} and bg <= fa                       # exact ⊂ fast (its TriMul lever is trimul_v4 there) ⊃ big: big = fast minus the CUDA-graph pool levers; no big-only lever
    assert (ex - {"trimul_exact"}) <= fa and "pair_transition_chunk" in ex and "pair_transition_chunk" in fa and "pair_transition_chunk" in bg   # every memory lever rides in all three rows (the chunked pair Transition: exact class, in exact / fast / big)

def test_unknown_mode_word_is_refused():
    """A mode word the kit does not ship (here `zzz_notamode`) is refused as unknown on every route, naming the word and the shipped modes: the CLI
    (argparse invalid choice, exit 2), modes.levers_of (ValueError naming MODE_NAMES) and the Python API (`NOT ACTIVE: unknown_mode:<word> (known: off,exact,fast)`, exit 3)."""
    import io, contextlib, pytest
    from atlasfold_opt import cli, modes, stack
    err = io.StringIO()
    with contextlib.redirect_stderr(err), pytest.raises(SystemExit) as x:
        cli.main(["pred", "--mode", "zzz_notamode", "--", "multimer", "--input-fasta", "x.fasta", "--out-dir", "o"])
    assert x.value.code == 2 and "invalid choice: 'zzz_notamode'" in err.getvalue()
    with pytest.raises(ValueError, match="mode must be one of"):
        modes.levers_of("zzz_notamode")
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(err), pytest.raises(SystemExit) as x:
        stack._REPORT = None; stack.activate("zzz_notamode")
    assert x.value.code == 3 and "unknown_mode:zzz_notamode" in err.getvalue()

def test_mode_names_literal_one_line():
    src = open(os.path.join(PKG, "modes.py")).read()
    assert len([l for l in src.splitlines() if l.startswith("MODE_NAMES")]) == 1

def test_kit_template_byte_identical():
    if not os.path.isdir(CORE): return
    assert _sha(os.path.join(CORE, "kit_template", "_core_gate.py")) == _sha(os.path.join(PKG, "_core_gate.py"))
    assert _sha(os.path.join(CORE, "kit_template", "_build_backend.py")) == _sha(os.path.join(OPT, "_build_backend.py"))

def test_pth_text_generated():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_bb", os.path.join(OPT, "_build_backend.py")); bb = importlib.util.module_from_spec(spec); spec.loader.exec_module(bb)
    assert open(os.path.join(OPT, "atlasfold_opt_autoload.pth")).read() == bb.pth_text("atlasfold_opt", "ATLASFOLD_OPT", "atlasfold-opt")

def test_pins_schema():
    pins = json.load(open(os.path.join(KIT, "stock", "PINS.json")))
    assert pins["upstream"]["commit"].startswith("992067e")
    pth = [w for w in pins["weights"] if w["file"].endswith(".pth")]
    assert len(pth) == 3 and all(len(w["sha256"]) == 64 and w["revision"] and w["url"].startswith("https://huggingface.co/") for w in pth)
    assert "ATLASFOLD_OPT" in pins["stock_environment"]["must_be_absent_prefixes"]

def test_core_pin_declared():
    s = open(os.path.join(OPT, "pyproject.toml")).read()
    assert re.search(r'^\[tool\.opt_core\]', s, re.M) and 'path = "../../common/opt_core"' in s and re.search(r'^version = "0\.5\.\d+\.\d+"', s, re.M)

def test_trimul_weight_vocabulary():
    import torch
    from atlasfold.model.network.primitives.triangle_update import TriangleMultiplicationOutgoing
    from atlasfold_opt.hooks.trimul import weights_of
    m = TriangleMultiplicationOutgoing(128); w = weights_of(m)
    assert set(w) == {"ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp", "ln_out_w", "ln_out_b", "w_o", "w_og"}
    assert tuple(w["w_ap"].shape) == (128, 128) and tuple(w["w_bg"].shape) == (128, 128) and tuple(w["w_og"].shape) == (128, 128)
    # a|b order == stock torch path (linear_in(z).chunk(2): a = rows [0:C])
    z = torch.randn(3, 3, 128); ab = m.linear_in(z); a, b = ab.chunk(2, dim=-1)
    assert torch.allclose(a, z @ w["w_ap"].T) and torch.allclose(b, z @ w["w_bp"].T)

def test_lm_sdpa_keeps_logits_cpu():
    import torch, os
    os.environ['AFO_LM_SDPA_MIN_TOKENS'] = '0'
    import importlib, atlasfold_opt.hooks.lm as _lm; importlib.reload(_lm)
    from atlasfold_opt.hooks import lm as H
    ins = H.install("fast", "atlasfold-opt", {})
    assert ins.applied
    from atlaslm.layers.attention import MultiHeadAttention
    m = MultiHeadAttention(64, 4).eval(); x = torch.randn(2, 9, 64); seq = torch.tensor([[0]*5+[1]*4, [0]*9]); pos = torch.arange(9).repeat(2, 1)
    with torch.no_grad():
        o1, a1 = m(x, seq, pos, False, True); o0, a0 = MultiHeadAttention.forward.__wrapped_stock__(m, x, seq, pos, False, True)
    assert torch.equal(torch.nan_to_num(a1, neginf=-1e9), torch.nan_to_num(a0, neginf=-1e9)) and float((o1 - o0).abs().max()) < 1e-5
    MultiHeadAttention.forward = MultiHeadAttention.forward.__wrapped_stock__


def test_every_mode_lever_has_an_installer_and_a_registry_row():
    from atlasfold_opt.modes import MODES
    from atlasfold_opt.hooks import installers
    from atlasfold_opt.registry import LEVERS
    table = installers()
    for mode, levers in MODES.items():
        for lever in levers:
            assert lever in LEVERS, (mode, lever)
            assert lever == "lever_report" or lever in table, (mode, lever, "no installer: activation would be partial -> NOT ACTIVE")


def test_size_gated_levers_do_not_refuse_when_every_call_is_below_the_floor():
    import os, importlib, torch
    os.environ['AFO_LM_SDPA_MIN_TOKENS'] = '1000'
    import atlasfold_opt.hooks.lm as H; importlib.reload(H)
    ins = H.install("fast", "atlasfold-opt", {})
    from atlaslm.layers.attention import MultiHeadAttention
    try:
        m = MultiHeadAttention(64, 4).eval(); x = torch.randn(1, 9, 64); seq = torch.zeros(1, 9, dtype=torch.long); pos = torch.arange(9).unsqueeze(0)
        with torch.no_grad():
            m(x, seq, pos, False, True)                      # 9 tokens < floor -> expected fallback below_min_tokens
        g = ins.gates[0]()
        assert g.ok, g
    finally:
        MultiHeadAttention.forward = MultiHeadAttention.forward.__wrapped_stock__
