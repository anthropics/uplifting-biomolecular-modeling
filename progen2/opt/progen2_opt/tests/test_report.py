"""report.py's lines: the package's printed-line contract."""
from progen2_opt import report


def test_report_lines():
    rep = {"active": True, "mode": "exact", "variant": "small", "route": "score", "kit_line": "score=forward:v0_score_r3_1+v0_ew", "stack": {"torch": "2.8.0+cu128", "transformers": "4.16.2", "tokenizers": "0.10.3"},
           "gpu": {"name": "NVIDIA H100 80GB HBM3", "sm": "sm90"}, "card_class": "h100", "optimizations": {"sample": [], "score": ["rotary_tables"]}, "applied": "configured",
           "package_home": "/t/progen2/opt", "kit_dirs": {"scoring": "/t/progen2/opt/forward/engines/progen2/kits/v0_score_r3_1", "serving": "/t/progen2/opt/serving/pipeline_v0_4"}}
    line = report.activation_line(rep)
    assert line == "[progen2-opt] ACTIVE mode=exact variant=small route=score kit=opt/forward/engines/progen2/kits/v0_score_r3_1 on=rotary_tables", line
    assert report.stack_line(rep).startswith("[progen2-opt] stack torch=2.8.0+cu128 transformers=4.16.2 tokenizers=0.10.3") and "card=h100" in report.stack_line(rep)
    rep2 = dict(rep, route="sample", optimizations={"sample": ["oneread_mmap", "static_kv"], "score": []})
    assert report.activation_line(rep2) == "[progen2-opt] ACTIVE mode=exact variant=small route=sample kit=opt/serving/pipeline_v0_4 on=oneread_mmap,static_kv"
    assert report.activation_line({"active": False, "mode": "exact", "reason": "why"}) == "[progen2-opt] NOT ACTIVE: why (mode=exact variant=None); " + report.REFUSED_ESCAPE
    assert report.activation_line({"active": False, "mode": "off", "variant": "small", "reason": "mode off: stock in a clean subprocess"}) == "[progen2-opt] NOT ACTIVE: mode off: stock in a clean subprocess (mode=off variant=small)"
    assert report.ready_line("small", 3.14159) == "[progen2-opt] ready variant=small t=3.14s"
    assert report.load_line("xlarge", 24.514, [64], 512) == "[progen2-opt] load variant=xlarge t=24.51s slots=64 max_length=512"
    assert report.load_line("small", 9.8, [1, 2, 4], 256) == "[progen2-opt] load variant=small t=9.80s slots=1,2,4 max_length=256" and report.load_line("small", 9.8, [], 256).endswith("slots=- max_length=256")
    assert report.KIT_MODULES == ("engines.progen2.kits.v0_score_r3_1", "engines.progen2.kits.v0_ew", "progen2_decode", "sampler_exact", "oneread_loader", "kit_t1")   # the EXIT line's kit_modules= census: both kits' modules, in this process
    assert report.item_line("ctx1_L256", 1.23456) == "[progen2-opt] item ctx1_L256 1.235s"
    assert report.exit_line("off", "sample", 3, []) == "[progen2-opt] EXIT mode=off route=sample items=3 kit_modules=none"
    assert report.score_ok_line(2, 4, {"calls": 4, "eager": 4}) == "[progen2-opt] score path ok units=2 forwards=4 counters=calls=4,eager=4"
    assert report.exit_tally_line().startswith("[progen2-opt] EXIT mode=")
    # the partial-exit line of score (the family grammar: the fixed parts byte-literal, the detail the engine's own — findings first, then the verb's reason; no escape word: not waivable)
    line = report.partial_line(["rotary tables: 3/1024 positions differ from the stock function (first [5, 9])", "counted path: counters off the counted path"], "the outputs stay", "exact", "small", "score")
    assert line == ("[progen2-opt] NOT ACTIVE: partial activation — rotary tables: 3/1024 positions differ from the stock function (first [5, 9]); counted path: counters off the counted path; "
                    "the outputs stay (mode=exact variant=small route=score); exit 3")
    assert line.startswith("[progen2-opt] NOT ACTIVE: partial activation — ") and line.endswith("(mode=exact variant=small route=score); exit 3") and "allow" not in line
    assert not hasattr(report, "partial_allowed_line")                                             # `PARTIAL allowed` is not a line of the package any more
    assert report.refused_line("the kit refused: drift") == "[progen2-opt] NOT ACTIVE: the kit refused: drift; " + report.REFUSED_ESCAPE
    assert report.REFUSED_ESCAPE == "exit 3 (`--mode off` runs the stock route without the kit)" and report.refused_line("x").endswith("--mode off` runs the stock route without the kit)")   # a refusal names its exit code and the one-flag escape on the same line
    assert report.refused_line("KitRefused: drift", "exact", "small", "score") == "[progen2-opt] NOT ACTIVE: KitRefused: drift (mode=exact variant=small route=score); " + report.REFUSED_ESCAPE
    assert report.refused_line("stock files differ from the pinned commit (sample.py: 1234abcd != 6451430c) — this is not the stock the modes are defined against", "off", "small", "score", escape="exit 3 (every mode refuses it: …)") == (
        "[progen2-opt] NOT ACTIVE: stock files differ from the pinned commit (sample.py: 1234abcd != 6451430c) — this is not the stock the modes are defined against (mode=off variant=small route=score); exit 3 (every mode refuses it: …)")


def test_engagement_lines():
    """The lever policy's words: the stack line NAMES the uncertainties (notes=), the ACTIVE line the mode's levers (on=, all of them) and — only
    when non-empty — what the kits found untested and engaged on (notes=); a lever that cannot run is the mode's refusal by name (refused_line over
    cannot_run_words); log_activation prints the stack line at activation, log_active the ACTIVE line once the evidence is in."""
    import io
    rep = {"active": True, "mode": "exact", "variant": "small", "route": "score", "kit_line": "score=forward:v0_score_r3_1+v0_ew", "stack": {"torch": "2.9.0+cu128", "transformers": "4.16.2", "tokenizers": "0.10.3", "python": "3.9.23"},
           "gpu": {"name": "NVIDIA RTX 6000 Ada", "sm": "sm89"}, "card_class": "wrong_card", "stack_key": "torch2.9.0-cu128-sm89", "applied": "deferred",
           "optimizations": {"sample": [], "score": ["rotary_tables", "one_forward_per_direction", "ew:gelu", "ew:ln"]},
           "notes": ["stack differs from the pinned one (torch have 2.9.0+cu128 want 2.8.0+cu128): the levers were tested on the pinned stack", "settings outside the tested defaults: fp16=false"],
           "package_home": "/t/progen2/opt", "kit_dirs": {"scoring": "/t/progen2/opt/forward/engines/progen2/kits/v0_score_r3_1", "serving": "/t/progen2/opt/serving/pipeline_v0_4"}}
    err = io.StringIO()
    assert report.log_activation(rep, stream=err) == ("[progen2-opt] stack torch=2.9.0+cu128 transformers=4.16.2 tokenizers=0.10.3 python=3.9.23 gpu=NVIDIA RTX 6000 Ada(sm89) card=wrong_card stack_key=torch2.9.0-cu128-sm89 "
                                                       "applied=deferred notes=stack differs from the pinned one (torch have 2.9.0+cu128 want 2.8.0+cu128): the levers were tested on the pinned stack; settings outside the tested defaults: fp16=false")
    line = report.log_active(rep, ["rotary_tables", "one_forward_per_direction", "ew:gelu", "ew:ln"],
                             notes=["ew kernels JIT-compiled from PTX (compute_90) on sm_120: no prebuilt image for this card in libew_progen2.so"], stream=err)
    assert line == ("[progen2-opt] ACTIVE mode=exact variant=small route=score kit=opt/forward/engines/progen2/kits/v0_score_r3_1 on=rotary_tables,one_forward_per_direction,ew:gelu,ew:ln "
                    "notes=ew kernels JIT-compiled from PTX (compute_90) on sm_120: no prebuilt image for this card in libew_progen2.so")
    assert report.activation_line(rep) == line and rep["on"] == ["rotary_tables", "one_forward_per_direction", "ew:gelu", "ew:ln"] and rep["apply_notes"] == ["ew kernels JIT-compiled from PTX (compute_90) on sm_120: no prebuilt image for this card in libew_progen2.so"]
    report.log_active(rep, ["x"], stream=err)                                                       # once per process: a second call prints nothing
    assert err.getvalue().splitlines() == [report.stack_line(rep), line]
    assert report.active_line("exact", "small", "sample", "opt/serving/pipeline_v0_4", ["oneread_mmap", "sampler_exact"]) == "[progen2-opt] ACTIVE mode=exact variant=small route=sample kit=opt/serving/pipeline_v0_4 on=oneread_mmap,sampler_exact"   # no notes= segment when empty
    assert " off=" not in line and not hasattr(report, "off_words") and not hasattr(report, "levers_off_line")   # a mode never runs with a subset: no off= segment, no LEVERS OFF line
    words = report.cannot_run_words([("oneread_mmap", "loader import failed (stock loader)"), ("static_kv", "no decode kit dir")])
    assert words == "oneread_mmap cannot run: loader import failed (stock loader); static_kv cannot run: no decode kit dir"
    assert report.refused_line(f"{words} — nothing generated", "exact", "small", "sample") == (
        "[progen2-opt] NOT ACTIVE: oneread_mmap cannot run: loader import failed (stock loader); static_kv cannot run: no decode kit dir — nothing generated "
        "(mode=exact variant=small route=sample); " + report.REFUSED_ESCAPE)
    cpu = {"active": False, "mode": "exact", "variant": "small", "route": "score", "reason": "--device cpu: the levers run on CUDA", "escape": "exit 3 (`--mode off --device cpu` runs the stock on the cpu)"}
    assert report.activation_line(cpu) == "[progen2-opt] NOT ACTIVE: --device cpu: the levers run on CUDA (mode=exact variant=small route=score); exit 3 (`--mode off --device cpu` runs the stock on the cpu)"
    assert report.activation_line(dict(cpu, dry_run=True, kit_line="k", optimizations={"score": []}, would_refuse=cpu["reason"])).endswith("would_refuse='--device cpu: the levers run on CUDA'")   # check names it as would_refuse

def test_peak_line():
    """The sample item's allocator line: the family grammar `PEAK item=<name> alloc_gib=<f.2> reserved_gib=<f.2>` + the pid of the process holding the model (this one)."""
    assert report.peak_line("gen-007", 43.129, 45.9, 4242) == "[progen2-opt] PEAK item=gen-007 alloc_gib=43.13 reserved_gib=45.90 pid=4242"
