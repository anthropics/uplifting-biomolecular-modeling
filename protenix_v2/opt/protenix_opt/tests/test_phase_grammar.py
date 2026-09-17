"""The PHASE line grammar: `PHASE item=<plan item name> seed=<s> lm_s=… trunk_s=… sampler_s=… conf_s=… fwd_s=… total_s=…`
— item is the runner's bare sample_name (never the runner's `<name>[seed:N]` decoration), the seed its own key=value token, every token key=value,
fwd_s (the model forward alone) immediately before total_s (device copy-in + forward)."""
import re

from protenix_opt import phase_timing


def tokens(line):
    head, *rest = line.split()
    assert head == phase_timing.PREFIX == "PHASE"
    return dict(t.split("=", 1) for t in rest)


def test_line_grammar():
    ln = phase_timing.phase_line("n0200_43ls__0w", 101, "-", {"trunk": 1.5, "sampler": 2.25, "conf": 0.125, "fwd": 3.9375}, 4.0)
    assert ln == "PHASE item=n0200_43ls__0w seed=101 lm_s=- trunk_s=1.500 sampler_s=2.250 conf_s=0.125 fwd_s=3.938 total_s=4.000"
    kv = tokens(ln)
    assert kv["item"] == "n0200_43ls__0w" and kv["seed"] == "101" and "[" not in kv["item"]
    assert all(re.fullmatch(r"[0-9]+\.[0-9]+", kv[k]) for k in ("trunk_s", "sampler_s", "conf_s", "fwd_s", "total_s"))
    keys = [t.split("=", 1)[0] for t in ln.split()[1:]]
    assert keys == ["item", "seed", "lm_s", "trunk_s", "sampler_s", "conf_s", "fwd_s", "total_s"]          # fwd_s immediately before total_s


def test_forward_site_is_the_model_forward():
    assert phase_timing.FWD_SITE == "forward" and phase_timing.FWD == "fwd" and phase_timing.ACCUMULATORS == ("trunk", "sampler", "conf", "fwd")
    assert phase_timing.SITES == {"trunk": "get_pairformer_output", "sampler": "sample_diffusion", "conf": "run_confidence_head"}


def test_item_is_the_bare_sample_name_and_seed_its_own_token():
    assert phase_timing.item_name({"sample_name": "7abc_A"}) == "7abc_A"
    assert phase_timing.item_name(None) == "unknown"
    assert phase_timing.item_seed(7) == 7


def test_guard_note_is_a_second_line_not_a_token():
    ln = phase_timing.phase_line("x", 1, "NA", {"trunk": 3.0, "sampler": 3.0, "conf": 3.0, "fwd": 4.0}, 10.0)
    first, note = ln.split("\n")
    assert tokens(first)["lm_s"] == "NA" and note == "PHASE-NOTE guard violated on x: trunk_s+sampler_s+conf_s=9.000 > fwd_s=4.000"


def test_guard_holds_the_forward_inside_predict():
    """trunk_s + sampler_s + conf_s <= fwd_s <= total_s: each violated inequality is its own PHASE-NOTE line, never hidden, never a token."""
    ok = phase_timing.phase_line("x", 1, "-", {"trunk": 1.0, "sampler": 1.0, "conf": 1.0, "fwd": 3.5}, 3.6)
    assert "\n" not in ok
    inner = phase_timing.phase_line("x", 1, "-", {"trunk": 2.0, "sampler": 2.0, "conf": 2.0, "fwd": 3.5}, 3.6).split("\n")
    assert len(inner) == 2 and inner[1] == "PHASE-NOTE guard violated on x: trunk_s+sampler_s+conf_s=6.000 > fwd_s=3.500"
    outer = phase_timing.phase_line("x", 1, "-", {"trunk": 1.0, "sampler": 1.0, "conf": 1.0, "fwd": 4.0}, 3.6).split("\n")
    assert len(outer) == 2 and outer[1] == "PHASE-NOTE guard violated on x: fwd_s=4.000 > total_s=3.600"
    unset = phase_timing.phase_line("x", 1, "-", {"trunk": 1.0, "sampler": 1.0, "conf": 1.0}, 3.6).split("\n")    # a route whose forward site never ran prints fwd_s=0.000 AND the guard note
    assert tokens(unset[0])["fwd_s"] == "0.000" and unset[1].startswith("PHASE-NOTE guard violated on x: trunk_s+sampler_s+conf_s=3.000 > fwd_s=0.000")


def test_peak_lines_grammar_without_cuda():
    """The reader's torch grammar is `PEAK item=<id> alloc_gib=<f> reserved_gib=<f>` (item first, exactly those tokens); on a box without CUDA
    NO PEAK line prints — one `PEAK-NOTE item=<id> seed=<s> cuda=absent` (never `alloc_gib=NA`)."""
    import re
    from protenix_opt import phase_timing as pt
    lines = pt.peak_lines("7xyz_A", 101)
    assert lines == ["PEAK-NOTE item=7xyz_A seed=101 cuda=absent"] or (
        len(lines) == 2 and re.fullmatch(r"PEAK item=7xyz_A alloc_gib=\d+\.\d\d reserved_gib=\d+\.\d\d", lines[0]) and lines[1] == "PEAK-NOTE item=7xyz_A seed=101")
    assert pt.peak_lines("x", None)[-1].startswith("PEAK-NOTE item=x seed=-")
    pt.peak_reset()                                                          # no CUDA: a no-op, never raises
