"""The mode table is locked to the kit's own LEVER_SETS (read from the file, live); `exact` keeps the bitwise levers on the stock-kernels base; overrides are labelled."""
import ast
import json
import os
import shutil

import pytest

from af3_torch_opt import big, modes, registry, stack


def test_mode_table():
    assert modes.MODES == ("off", "exact", "fast", "big") and modes.DEFAULT_MODE == "fast"     # the rule: ../README.md (always fast)
    assert modes.REFUSED_MODES == {}                                                               # no standard mode name is refused: exact meets its guarantee on the stock-kernels base
    assert set(modes.MODE_SETS) == set(modes.MODES) == set(modes.MODE_DTK) == set(modes.MODE_PACKAGE_LEVERS)
    assert modes.MODE_SETS == {"off": "eager", "exact": "fastest", "fast": "fastest", "big": "fastest"} and modes.MODE_DTK == {"off": False, "exact": False, "fast": True, "big": True}
    assert modes.MODE_PADDING == {"off": "none", "exact": "none", "fast": "kernel_tile", "big": "kernel_tile"} and set(modes.MODE_PADDING) == set(modes.MODES)
    assert modes.MODE_PACKAGE_LEVERS == {"off": (), "exact": ("template_dedupe", "dev_scalars", "tri_layout", "ln_rows", "attn_layout", "gate_fuse", "castcache", "prefetch", "write_behind", "autotune_cache", "feat_par"), "fast": ("template_dedupe", "dev_scalars", "tri_layout", "ln_rows", "attn_layout", "gate_fuse", "canonical_noise", "prefetch", "write_behind", "autotune_cache", "feat_par"), "big": ("template_dedupe", "dev_scalars", "tri_layout", "ln_rows", "attn_layout", "gate_fuse", "canonical_noise", "prefetch", "write_behind", "autotune_cache", "feat_par")}
    assert modes.MODE_FILTER == {"exact": "EXACT"} and modes.FASTNN_MODES == ("off", "exact") and modes.STOCK_NUM_DIFFUSION_SAMPLES == 5
    assert all(("canonical_noise" in modes.MODE_PACKAGE_LEVERS[m]) == (modes.MODE_PADDING[m] == "kernel_tile") for m in modes.MODES)   # canonical noise rides the kernel_tile padding
    assert set(registry.PACKAGE_LEVERS) <= set(registry.LEVERS) and all(set(v) <= set(registry.PACKAGE_LEVERS) for v in modes.MODE_PACKAGE_LEVERS.values())


def test_kit_lever_sets_are_the_kits():
    sets = modes.kit_lever_sets()
    assert sets["eager"] == ()
    assert set(sets) == {"eager", "fastest"} and sets["eager"] == ()
    assert "compile" in sets["fastest"] and "stepgraph" in sets["fastest"] and "graph_pairformer" not in sets["fastest"]
    # the same values by a second, independent read of the kit file
    src = open(os.path.join(stack.kit_home(), modes.API_RELPATH), encoding="utf-8").read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "LEVER_SETS")
    assert {k: tuple(v) for k, v in ast.literal_eval(node.value).items()} == sets


def test_resolve_off_and_fast():
    off, fast = modes.resolve("off"), modes.resolve("fast")
    sets = modes.kit_lever_sets()
    assert off == {"mode": "off", "lever_set": "eager", "levers": (), "dtk": False, "big": None, "package_levers": (), "padding": "none", "fastnn": True, "levers_off": ()}   # levers_off: modes.ENV_LEVERS_OFF dropped nothing (test_levers_off.py)
    assert fast["lever_set"] == "fastest" and fast["levers"] == tuple(sets["fastest"]) and "graph_pairformer" not in fast["levers"] and fast["dtk"] is True and fast["package_levers"] == ("template_dedupe", "dev_scalars", "tri_layout", "ln_rows", "attn_layout", "gate_fuse", "canonical_noise", "prefetch", "write_behind", "autotune_cache", "feat_par") and "overridden" not in fast


def test_exact_composition_on_every_route(box, capsys, monkeypatch):
    """`exact` = the stock-kernels base (xfold's shipped fastnn kernels: the stock CLI's --fastnn default, modes.FASTNN_MODES) + registry.EXACT (the kit
    levers byte-equal to that base, in `fastest`'s order) + the package levers template_dedupe, dev_scalars, tri_layout, ln_rows, attn_layout, gate_fuse and castcache; no DTK; stock AF3 bucket padding. Every
    route composes exactly that and refuses nothing legal: `modes.resolve`, `check --mode exact`, `pred --mode exact`, `AF3_TORCH_OPT=exact`."""
    from af3_torch_opt import cli
    from .conftest import stub_calls
    r = modes.resolve("exact")
    assert (r["lever_set"], r["levers"], r["dtk"], r["package_levers"], r["padding"], r["fastnn"]) == ("exact", registry.EXACT, False, ("template_dedupe", "dev_scalars", "tri_layout", "ln_rows", "attn_layout", "gate_fuse", "castcache", "prefetch", "write_behind", "autotune_cache", "feat_par"), "none", True)
    assert registry.EXACT == ("stepgraph", "glu_proj", "trimul_exact") and all(registry.BITWISE_EVIDENCE[l]["bitwise_vs_stock_kernels"] and registry.BITWISE_EVIDENCE[l]["kept"] for l in registry.EXACT + ("hoist", "template_dedupe", "dev_scalars", "tri_layout", "ln_rows", "attn_layout", "gate_fuse", "castcache", "prefetch", "write_behind", "autotune_cache", "feat_par")[1:]) and registry.BITWISE_EVIDENCE["hoist"]["bitwise_vs_stock_kernels"]
    assert registry.BITWISE_EVIDENCE["bf16w"]["bitwise_vs_stock_kernels"] is False and "bf16w" not in r["levers"]          # bitwise vs the eager port only (xfold's fastnn gated-linear-unit kernel reads the fp32 weight itself)
    assert set(registry.EXACT).isdisjoint(registry.NOT_BITWISE) and "compile" not in r["levers"] and "graph_pairformer" not in r["levers"]
    assert modes.fastnn_refusal("exact", True) is None and modes.fastnn_refusal("exact", None) is None and modes.fastnn_refusal("exact", False) and modes.fastnn_refusal("fast", True)   # the stock kernels ARE exact's base; --nofastnn is off's alone; fast refuses an explicit --fastnn by name
    assert cli.main(["check", "--mode", "exact"]) == 0
    err = capsys.readouterr().err
    assert " ACTIVE mode=exact lever_set=exact levers=stepgraph+glu_proj+trimul_exact dtk=0 " in err and err.rstrip().endswith(" package=template_dedupe,dev_scalars,tri_layout,ln_rows,attn_layout,gate_fuse,castcache,prefetch,write_behind,autotune_cache,feat_par padding=none compile=off:mode"), err
    inp = os.path.join(str(box["tmp"]), "in"); os.makedirs(inp, exist_ok=True); src = "tiny.json"
    with open(os.path.join(inp, src), "w") as fh: json.dump({"name": "tiny", "sequences": [], "modelSeeds": [1]}, fh)
    out = str(box["tmp"] / "oute")
    assert cli.main(["pred", "--mode", "exact", "--json_path", os.path.join(inp, src), "--output_dir", out]) == 0
    err = capsys.readouterr().err
    assert " ACTIVE mode=exact lever_set=exact levers=stepgraph+glu_proj+trimul_exact dtk=0 " in err and "[af3-torch-opt] SETTINGS num_recycles=None num_diffusion_samples=5 diffusion_steps=None fastnn=1 confidence=once_per_sample" in err, err
    fwd = [c for c in stub_calls(box) if c and os.path.basename(c[1]) == "forward.py"][-1]
    assert fwd[fwd.index("--levers") + 1] == "stepgraph,glu_proj,trimul_exact" and fwd[fwd.index("--dtk") + 1] == "0" and fwd[fwd.index("--fastnn") + 1] == "1"
    assert fwd[fwd.index("--package-levers") + 1] == "template_dedupe,dev_scalars,tri_layout,ln_rows,attn_layout,gate_fuse,castcache,prefetch,write_behind,autotune_cache" and fwd[fwd.index("--padding") + 1] == "none"
    feat = [c for c in stub_calls(box) if c and os.path.basename(c[1]) == "featurise.py"][-1]; assert feat[feat.index("--buckets") + 1] == "none" == cli.bucket_row("none")   # xfold as shipped: no token padding
    man = cli.last_run(); a_ = man["activation"]
    assert (a_["mode"], a_["lever_set"], a_["settings"]["fastnn"], "settings_preset" in a_) == ("exact", "exact", 1, False) and man["partial"] is None
    assert cli.main(["pred", "--mode", "exact", "--num_recycles", "1", "--num_diffusion_samples", "1", "--diffusion_steps", "2", "--json_path", os.path.join(inp, src), "--output_dir", str(box["tmp"] / "oute2")]) == 0   # the knobs pass through; the mode keeps its kernels
    err = capsys.readouterr().err; assert "[af3-torch-opt] SETTINGS num_recycles=1 num_diffusion_samples=1 diffusion_steps=2 fastnn=1 confidence=once_per_sample" in err, err
    monkeypatch.setenv(modes.ENV_MODE, "exact")
    assert cli.main(["check"]) == 0 and " ACTIVE mode=exact " in capsys.readouterr().err
def test_bitwise_evidence_table():
    """Every lever measured bitwise against the eager `off` carries its record; a bitwise lever not in force says why; the tier-2 levers are named;
    every lever of fast's kit set is classified one way or the other."""
    sets = modes.kit_lever_sets()
    for l, ev in registry.BITWISE_EVIDENCE.items():
        assert ev["bitwise_vs_off"] is True and (ev["kept"] or ev.get("why")), l
    assert set(registry.BITWISE_EVIDENCE).isdisjoint(registry.NOT_BITWISE) and set(registry.NOT_BITWISE) < set(registry.LEVERS)
    assert set(sets["fastest"]) <= set(registry.BITWISE_EVIDENCE) | set(registry.NOT_BITWISE)          # every lever of fast is classified
    assert set(registry.DETERMINISM) == set(modes.MODES)

def test_unknown_refused():
    with pytest.raises(modes.UnsupportedMode, match="unknown mode"):
        modes.resolve("turbo")
    with pytest.raises(TypeError):                                                  # the mode is the whole surface: resolve takes no lever / dtk / package-lever override
        modes.resolve("fast", levers="fastest")


def test_lever_sets_read_live_from_the_kit(tmp_path, monkeypatch):
    """Editing the kit's LEVER_SETS changes the resolved levers: the table is read, never transcribed."""
    kit = tmp_path / "kit"; shutil.copytree(stack.kit_home(), kit)
    api = kit / modes.API_RELPATH
    src = api.read_text(encoding="utf-8").replace('"fastest": ("bf16w",', '"fastest": ("dummy_lever", "bf16w",')
    api.write_text(src, encoding="utf-8")
    monkeypatch.setenv("AF3_TORCH_OPT_KIT", str(kit))
    assert modes.resolve("fast")["levers"][0] == "dummy_lever"


def test_registry_covers_every_lever_of_the_line():
    sets = modes.kit_lever_sets()
    assert set(sets["fastest"]) | {"dtk"} <= set(registry.LEVERS)
    for name, d in registry.LEVERS.items():
        assert {"name", "kind", "numerics", "switch", "touches"} <= set(d), name
        for rel in d["touches"]:
            assert os.path.exists(stack.touch_path(rel).rstrip("/")), (name, rel)          # under opt/forward/, or the core's package for a routed kernel's `core:` path


def test_mode_from_env(monkeypatch):
    monkeypatch.delenv("AF3_TORCH_OPT", raising=False)
    assert modes.mode_from_env() == "fast" == modes.DEFAULT_MODE
    monkeypatch.setenv("AF3_TORCH_OPT", "off")
    assert modes.mode_from_env() == "off"


def test_big_recomposes_fastest():
    """big = fastest with the graph levers dropped (stepgraph -> hoist kept) + DTK + every memory lever in LEVER_ORDER: ONE composition,
    no toggles (a caller-set AF3_TORCH_BIG_* name is refused by the undeclared-variable gate, test_stack_gates)."""
    from af3_torch_opt import big
    fastest = modes.resolve("fast")["levers"]                                  # big recomposes fast's table (the kit's fastest set minus registry.FAST_EXCLUDED)
    r = modes.resolve("big")
    assert r["lever_set"] == "big" and r["dtk"] is True and "overridden" not in r
    assert r["levers"] == tuple(l for l in big.build_levers(fastest, big.LEVER_ORDER)) == tuple("hoist" if l == "stepgraph" else l for l in fastest) and "stepgraph" not in r["levers"]   # NO step graph in big: its hoist kept
    assert r["big"] == big.selection() == {"levers": list(big.LEVER_ORDER), "settings": {"graph_drop_min_tokens": big.GRAPH_DROP_MIN_TOKENS, "paircond_rows": big.PAIRCOND_CHUNK_ROWS}} and "buckets" not in big.LEVER_ORDER
    assert big.GRAPH_DROP_MIN_TOKENS == 0                                                   # graph_drop: the graph levers dropped from the build (no CUDA graph in the memory line at any size)
    argv = big.forward_argv(r["big"])
    assert argv[:2] == ["--big", ",".join(big.LEVER_ORDER)] and "--graph-drop-tokens" not in argv
    assert big.build_levers(fastest, big.LEVER_ORDER, {"graph_drop_min_tokens": 832}) == tuple(fastest)   # a positive gate = the per-item size gate: the graph stays built
    assert big.graph_dropped_for(832, 832) and not big.graph_dropped_for(768, 832) and not big.graph_dropped_for(448, None)
    assert not hasattr(big, "bucket_row") and not hasattr(big, "BUCKETS_MAX")            # one padding rule for the kit line: big pads on cli.bucket_row like fast
    src = open(big.__file__, encoding="utf-8").read()
    assert "AF3_TORCH_BIG_" not in src and "environ" not in src                 # no environment word selects a composition


def test_padding_policies():
    """off / exact do not pad (`none`: each input at its own token count, xfold as shipped); fast / big pad to the shared core's kernel
    tile through opt_core.shape_policy — the one policy function; an unknown policy is refused by name; a core without shape_policy refuses
    kernel_tile by name (stack.padding_gate), never another row."""
    from af3_torch_opt import cli
    assert cli.bucket_row("none") == "none" and modes.MODE_PADDING == {"off": "none", "exact": "none", "fast": "kernel_tile", "big": "kernel_tile"}
    with pytest.raises(ValueError, match="unknown padding policy"):
        cli.bucket_row("tile64")
    assert stack.padding_gate("none") is None and stack.padding_gate(None) is None
    try:
        from opt_core import shape_policy as sp
    except ImportError:
        assert "opt_core.shape_policy" in stack.padding_gate("kernel_tile")
        return
    assert cli.bucket_row("kernel_tile") == f"tile:{sp.KERNEL_TILE}"                          # fast: the open-ended row — every length pads to the next kernel tile, none is refused or left unpadded
    from af3_torch_opt import featurise
    import bisect
    rng = featurise.parse_buckets(cli.bucket_row("kernel_tile"))
    assert isinstance(rng, range) and rng.step == sp.KERNEL_TILE and rng[0] == sp.KERNEL_TILE and len(rng) == featurise.TILE_ROW_LEN
    for n in list(range(1, 4200)) + [6143, 6144, 6145, 20000, 100001]:                         # the featuriser bisects the row exactly as alphafold3 calculate_bucket_size does: == the core's padded_len at every size
        assert rng[bisect.bisect_left(rng, n)] == sp.padded_len(n, "kernel_tile"), n
    assert featurise.buckets_word(rng) == cli.bucket_row("kernel_tile") and featurise.parse_buckets("none") is None and featurise.parse_buckets("256,512") == (256, 512)
    assert cli.bucket_row(modes.MODE_PADDING["big"]) == cli.bucket_row(modes.MODE_PADDING["fast"]) == featurise.buckets_word(rng)   # one grid for both padded modes


# ---- card classes (B10): every lever declared once in opt_core.arch; H100/H200 (sm90) and A100 (sm80) tested for every lever of every
#      mode (registry.ARCH_TESTED); B200/B300 untested; nothing below sm80 (the verdict words are the core's) -----

def test_every_lever_is_declared_once():
    assert set(registry.ARCH) == set(registry.LEVERS) | set(big.LEVER_ORDER) | set(registry.N_GPU_LEVER)
    sup = registry.declare_arch()
    assert set(sup) == set(registry.ARCH)
    assert registry.declare_arch().keys() == sup.keys()                       # idempotent (same content twice is a no-op in opt_core.arch)
    from opt_core import arch
    for lever in registry.ARCH:
        assert arch.declared(registry.arch_lever_id(lever)) is not None


def test_card_table_words():
    t = registry.card_table(["sm80", "sm90", "sm100", "sm103"])
    assert registry.ARCH_TESTED == ("sm80", "sm90") and "graph_pairformer" not in registry.LEVERS
    for lever in list(registry.LEVERS) + list(big.LEVER_ORDER) + list(registry.N_GPU_LEVER):   # every mode's levers: tested on sm90 (H100) and sm80 (A100)
        assert t[lever]["sm90"] == t[lever]["sm80"] == "supported", (lever, t[lever])
        assert all(t[lever][sm].startswith("uncertified") for sm in ("sm100", "sm103")), (lever, t[lever])
