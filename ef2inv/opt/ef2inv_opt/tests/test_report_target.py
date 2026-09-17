"""The report line grammar (report.py), and the cookbook's own target rule (target_name / target_sequence)."""
import os

import pytest

from .. import modes, report as R, stock_design as SD
from . import _paths as P


def test_line_grammar():
    m = modes.MODES["fast"]                                                   # fast's own row
    a = R.active_line(m, {"esm": "3.4.0@d0207ea3", "torch": "2.11.0+cu128", "device": "NVIDIA H100 80GB HBM3", "kit_switch_var": "EF2_FAST_KIT", "compile": "stock"})
    assert a.startswith("ACTIVE mode=fast route=subprocess tier=2 esm=3.4.0@d0207ea3 torch=2.11.0+cu128 gpu=NVIDIA H100 80GB HBM3 kit_switch=EF2_FAST_KIT=agk3 cache=unset compile=stock line=") and " alias=" not in a
    # the ACTIVE grammar's one optional token after mode=: alias=<word> when the run was invoked by an alias of its mode (modes.ALIASES, empty as shipped), nothing under the mode's own name
    assert modes.alias_word("fast") is None and R.active_line(m, {"compile": "stock", "alias": modes.alias_word("fast")}) == R.active_line(m, {"compile": "stock"})
    b = R.active_line(m, {"esm": "3.4.0@d0207ea3", "torch": "2.11.0+cu128", "device": "NVIDIA H100 80GB HBM3", "kit_switch_var": "EF2_FAST_KIT", "compile": "stock", "alias": "quick"})   # the token's grammar, were an alias defined
    assert b.startswith("ACTIVE mode=fast alias=quick route=subprocess tier=2 ") and b.replace(" alias=quick", "", 1) == a
    g = R.active_line(modes.MODES["big"], {"compile": "stock"})
    assert g.startswith("ACTIVE mode=big route=subprocess tier=2 ") and "kit_switch=EF2_FAST_KIT=big" in g and "memory plan pinned to its floor" in g
    assert R.active_line(modes.MODES["off"], {}).startswith("ACTIVE mode=off ") and "kit_switch=none" in R.active_line(modes.MODES["off"], {}) and "settings=" not in a
    assert R.not_active_line("pins") == "NOT ACTIVE reason=pins"
    assert R.run_line("off", "pdl1_binder", 115, 80, 195, 0, 150) == "RUN mode=off target=pdl1_binder target_len=115 binder_len=80 tokens=195 seed=0 steps=150"
    assert R.run_line("off", "pd-l1", "built-in", 80, None, 0, 150) == "RUN mode=off target=pd-l1 target_len=built-in binder_len=80 tokens=None seed=0 steps=150"
    assert R.ready_line(31.2, None, 16.7) == "ready t=31.2 first_call=nan params_load=16.7"
    ln = R.run_summary_line("pdl1_binder", 0, 195, 150, {"step_time_steady_median_s": 0.7668, "first_calls_s": [29.14, 12.0], "design_s": 189.18}, 1.234, 0.5, "ab" * 32)
    assert ln == "[run] target=pdl1_binder seed=0: tokens 195 steps 150 steady 0.767 s/step first_calls 29.1,12.0 total 189.2 s loss[-1]=1.2340 iptm=0.5000 design.pdb sha256=abababababababab"
    assert R.evidence_line({"applied": True, "fallback": [], "refusals": [], "events": ["x"]}, "fast") == "EVIDENCE levers=applied fallback=none missing=none refusals=0 events=1"
    assert R.evidence_line({"applied": False, "fallback": ["F1 pPPL graph disabled itself: x"], "refusals": ["F1 pPPL graph disabled itself: x"], "events": []}, "exact").startswith("EVIDENCE levers=partial fallback=F1 ")
    assert R.exit_line(0, m, "applied", "/o") == "EXIT rc=0 mode=fast kit_switch=agk3 evidence=applied cache=unset out=/o"
    assert R.exit_line(0, modes.MODES["big"], "applied", "/o", R.cache_word("per-box")) == "EXIT rc=0 mode=big kit_switch=big evidence=applied cache=per-box out=/o"
    assert R.cache_word("shared", explicit=True) == "shared+explicit"
    assert "cache=per-box" in R.active_line(m, {"cache": "per-box"}) and "cache=unset" in R.active_line(m, {})


BUILT_IN = {"cd45": "GSPGEPQIIF", "pd-l1": "AFTVTVPKDL"}                                       # the shape of the cookbook's TARGET_SEQUENCES (name -> sequence)


def test_the_cookbooks_own_target_rule():
    """resolve_target = the cookbook's design_binder rule (l.997-1005), applied before anything loads: a built-in ``target_name`` takes NO
    sequence (its own), any other name takes the explicit ``target_sequence``; the two combinations the cookbook itself raises on are refused
    with its own words. Returns (target_name, design()'s target_sequence keyword, route, the effective sequence)."""
    assert SD.resolve_target("pd-l1", None, BUILT_IN) == ("pd-l1", None, "built-in", "AFTVTVPKDL")                            # a built-in: its own sequence, target_sequence=None to design()
    assert SD.resolve_target("il7ra", "MKWVTFISLL", BUILT_IN) == ("il7ra", "MKWVTFISLL", "sequence", "MKWVTFISLL")            # a name + its sequence
    assert SD.resolve_target("cd45_binder", "QIIFCRSEAAHQGV", BUILT_IN) == ("cd45_binder", "QIIFCRSEAAHQGV", "sequence", "QIIFCRSEAAHQGV")   # the built-in cd45 construct under its item name
    with pytest.raises(ValueError, match="'cd45' is a preset target; omit target_sequence."):                                  # the cookbook's l.1000, verbatim
        SD.resolve_target("cd45", "QIIFCRSEAAHQGV", BUILT_IN)
    with pytest.raises(ValueError, match="'x' is not a preset target; provide target_sequence."):                             # l.1005, verbatim
        SD.resolve_target("x", None, BUILT_IN)
    with pytest.raises(ValueError, match="provide target_sequence"):
        SD.resolve_target("x", "", BUILT_IN)
    real = P.stock_literals({"TARGET_SEQUENCES"})["TARGET_SEQUENCES"]                                                    # the file's own table (l.159-171)
    assert {"pd-l1", "cd45"} <= set(real) and SD.resolve_target("pd-l1", None, real) == ("pd-l1", None, "built-in", real["pd-l1"])
    stock = open(P.STOCK_FILE).read().splitlines()
    assert [ln.strip() for ln in stock[996:1000]] == ["if target_name in TARGET_SEQUENCES:", "if target_sequence is not None:", "raise ValueError(",
                                                      'f"{target_name!r} is a preset target; omit target_sequence."']            # l.997-1000: the rule this mirrors
    assert stock[1004].strip() == 'f"{target_name!r} is not a preset target; provide target_sequence."'
    src = open(SD.__file__).read()
    for gone in ("read_pdb_chains", "read_fasta_one", "fasta_record_count", '"--target-fasta"', '"--chain"', '"--case-id"', '"--tag"', "EXPLICIT_TARGET"):
        assert gone not in src, gone                                                                                            # no file reader, no case / tag labels: the cookbook's two target parameters only


def test_main_resolves_the_target_through_the_cookbooks_rule_and_records_it():
    """stock_design.main (CUDA-only past this point; its contract read from source, as these tests read main's other contracts): the target goes
    through resolve_target before anything loads (a usage error exits 3 by name), run.json `case` records target_name / route / length /
    sequence digest, design() receives exactly the resolved pair, and design.fasta / the [run] line / the PEAK lines are named by target_name."""
    import inspect
    src = inspect.getsource(SD.main)
    assert "target_name, target_sequence, target_route, target_seq = resolve_target(a.target_name, a.target_sequence, BD.TARGET_SEQUENCES)" in src
    assert '"target_name": target_name, "target_route": target_route, "target_len": len(target_seq)' in src
    assert "app.design(target_name=target_name, target_sequence=target_sequence," in src
    assert 'O.write_fasta(os.path.join(a.out, "design.fasta"), a.target_name, target_seq, binder_seq)' in src and "R.peak_lines(a.target_name, a.seed" in src
    assert "R.run_summary_line(a.target_name, a.seed, n_tokens" in src


def test_peak_lines_grammar_and_no_cuda():
    """`PEAK item=<bare name> alloc_gib=<%.2f> reserved_gib=<%.2f>` — exactly three tokens, item first (a fixed grammar) — then
    `PEAK-NOTE item=<id> seed=<s>`; without torch.cuda NO PEAK line, one `PEAK-NOTE item=<id> seed=<s> cuda=absent`."""
    import re, types
    from .. import report as R
    assert R.peak_lines("pdl1_binder", 0, 20113674240, 21474836480) == ["PEAK item=pdl1_binder alloc_gib=18.73 reserved_gib=20.00", "PEAK-NOTE item=pdl1_binder seed=0"]
    assert re.fullmatch(r"PEAK item=\S+ alloc_gib=\d+\.\d+ reserved_gib=\d+\.\d+", R.peak_lines("x", 1, 2 ** 30, 3 * 2 ** 29)[0])
    no_cuda = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    assert R.cuda_peaks(no_cuda) == (None, None) and R.cuda_peaks(types.SimpleNamespace()) == (None, None)
    assert R.peak_lines("cd45_binder", 3, *R.cuda_peaks(no_cuda)) == ["PEAK-NOTE item=cd45_binder seed=3 cuda=absent"]             # no PEAK line without a device
    gpu = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, max_memory_allocated=lambda: 2 ** 30, max_memory_reserved=lambda: 3 * 2 ** 29))
    assert R.peak_lines("x", 1, *R.cuda_peaks(gpu)) == ["PEAK item=x alloc_gib=1.00 reserved_gib=1.50", "PEAK-NOTE item=x seed=1"]
    assert f"{R.PREFIX} {R.peak_lines('x', 1, 2 ** 30, 2 ** 30)[0]}" == "[ef2inv-opt] PEAK item=x alloc_gib=1.00 reserved_gib=1.00"      # as logged


def test_the_rows_model_inputs_are_unchanged_by_the_target_surface():
    """A caller that named the target by FILE (`--target target.pdb --chain A`, the sequence read off the chain's CA records) now names
    it the cookbook's way (`--target-name {name} --target-sequence <that sequence>`): given the same residues, the design() target keyword is
    byte-for-byte that sequence on the `sequence` route (an item name is never a built-in name), so every model input — target_sequence, the binder
    factory / length, seed, hotspots, the pass-through knobs — is what it was; target_name is a label the cookbook only uses to look up a built-in
    (design_binder l.997-1005) and it names design.fasta's records."""
    import inspect
    from .. import launch
    seq = "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPY"   # pd-l1's 115 residues, as an item's PDB chain / target.fasta carries them
    for name in ("pdl1_binder", "1jz7A_N200", "cd45_binder"):
        assert SD.resolve_target(name, seq, {"pd-l1": seq, "cd45": "X"}) == (name, seq, "sequence", seq)          # the explicit sequence verbatim, under the item's name — even when it equals a built-in's
    argv = launch.argv_for(modes.MODES["off"], P.ROOT, cookbook_stock="/s/binder_design.py", target_name="pdl1_binder", target_sequence=seq, binder_len=80, seed=0, out="/o")
    i = argv.index("--target-name"); assert argv[i:i + 4] == ["--target-name", "pdl1_binder", "--target-sequence", seq] and "--target-fasta" not in argv and "--case-id" not in argv and "--tag" not in argv
    built_in = launch.argv_for(modes.MODES["off"], P.ROOT, cookbook_stock="/s/binder_design.py", target_name="pd-l1", binder_len=80, seed=0, out="/o")
    assert "--target-sequence" not in built_in and [t for t in argv if t not in ("pdl1_binder", "--target-sequence", seq)] == [t for t in built_in if t != "pd-l1"]   # a built-in alone: the same argv minus the sequence pair
    src = inspect.getsource(SD.main)
    assert "app.design(target_name=target_name, target_sequence=target_sequence, binder_name=binder_name, binder_sequence=st.binder_sequence, seed=a.seed," in src or "app.design(target_name=target_name, target_sequence=target_sequence," in src
