"""The line grammar (report.py) is a contract: every line the package prints matches its pattern, field by field."""
from gpnstar_opt import report

REP = {"gpu": {"name": "NVIDIA H100 80GB HBM3", "sm": "sm90", "mib": 81559}, "card": "h100", "model_label": "gpn-star-hg38-v100-200m@0c949f13",
       "gpn_commit8": "6f28c81b", "torch": "2.13.0", "transformers": "5.15.0", "tf32": True, "words": ["weights=ok(v100-200m)"]}


def test_active_line_fields_in_order():
    s = report.active_line(REP)
    assert s == ("[gpnstar-opt] ACTIVE mode=exact levers=devconst+constcache+srcgather+unifiedkv+dedup+colattn+fusedattn+graph gpu=NVIDIA_H100_80GB_HBM3(sm90) card=h100 mib=81559 "
                 "model=gpn-star-hg38-v100-200m@0c949f13 gpn=6f28c81b torch=2.13.0 transformers=5.15.0 tf32=on weights=ok(v100-200m)")
    m = report.RE_ACTIVE.match(s)
    assert m and m.group("mode") == "exact" and m.group("card") == "h100" and m.group("tf32") == "on" and m.group("words") == " weights=ok(v100-200m)"


def test_active_line_names_uncertainty_never_hides_it():
    rep = dict(REP, gpu={"name": "NVIDIA L4", "sm": "sm89", "mib": 22731}, card=None, tf32=False, torch="2.9.0",
               words=["card=untested(NVIDIA_L4,22731MiB,sm89)", "stack=drift(torch:2.9.0!=2.13.0)"])
    s = report.active_line(rep)
    assert s.endswith("tf32=off card=untested(NVIDIA_L4,22731MiB,sm89) stack=drift(torch:2.9.0!=2.13.0)")
    assert " card=untested mib=22731 " in s and report.RE_ACTIVE.match(s)
    assert report.active_line(dict(REP, tf32=None)).split(" tf32=")[1].startswith("unknown")     # torch answered through neither interface: said, not guessed


def test_lever_lines_one_per_lever_all_on():
    ls = report.lever_lines()
    assert [report.RE_LEVER.match(x).group("name") for x in ls] == list(report.LEVERS)
    assert all(report.RE_LEVER.match(x).group("state") == "on" for x in ls)
    assert ls[0] == "[gpnstar-opt] LEVER name=devconst state=on impl=gpnstar_opt.accel.patches origin=kit"


def test_kv_line_and_reasons():
    s = report.kv_line("dedup", 1, 0, "8x128", first_ms=812.4, reason="kvcheck", compile=False)
    assert s == "[gpnstar-opt] KV route=dedup pairs=1 rejects=0 shape=8x128 first_ms=812 reason=kvcheck compile=off"
    m = report.RE_KV.match(s)
    assert m and m.group("route") == "dedup" and m.group("first_ms") == "812" and m.group("reason") == "kvcheck" and m.group("compile") == "off"
    assert report.kv_line("unifiedkv", 0, 0, "1x128", first_ms=95, reason="small_batch", compile=True).endswith("shape=1x128 first_ms=95 reason=small_batch compile=on")
    assert report.kv_line("stock", 1, 0, "512x128", first_ms=3000, reason="memory").endswith("route=stock pairs=1 rejects=0 shape=512x128 first_ms=3000 reason=memory compile=off")
    x = report.kv_line("dedup", 2, 0)                                                       # the exit tally: shape=all, no first forward of its own
    assert x == "[gpnstar-opt] KV route=dedup pairs=2 rejects=0 shape=all first_ms=none reason=none compile=off" and report.RE_KV.match(x)
    assert report.RE_KV.match(report.kv_line("bogus", 0, 0, reason="whatever")).group("route") == "none"     # unknown words collapse to none, the grammar holds


def test_dry_run_not_active_compile_removed_error():
    d = report.dry_run_line(dict(REP, words=["weights=ok(v100-200m)"], would_refuse=None))
    assert d.startswith("[gpnstar-opt] DRY-RUN mode=exact levers=devconst+constcache+srcgather+unifiedkv+dedup+colattn+fusedattn+graph gpu=NVIDIA_H100_80GB_HBM3(sm90) card=h100 ")
    assert " tf32=" not in d and d.endswith(" weights=ok(v100-200m) would_refuse=none") and report.RE_DRY_RUN.match(d).group("would_refuse") == "none"
    assert report.RE_DRY_RUN.match(report.dry_run_line({"gpu": None, "words": [], "would_refuse": "no CUDA device is visible"})).group("would_refuse") == "no CUDA device is visible"
    n = report.not_active_line("transformers is 5.15.1, upstream pins 5.15.0 (stock/PINS.json pins.transformers)")
    assert n == "[gpnstar-opt] NOT ACTIVE: transformers is 5.15.1, upstream pins 5.15.0 (stock/PINS.json pins.transformers)" and report.RE_NOT_ACTIVE.match(n)
    assert report.not_active_line("the model is on cpu", "mode=exact") == "[gpnstar-opt] NOT ACTIVE: the model is on cpu (mode=exact)"
    c = report.compile_line("compiled")
    assert c == "[gpnstar-opt] COMPILE levers=eager(kit-disabled) rest=compiled" and report.RE_COMPILE.match(c)
    assert report.RE_COMPILE.match(report.compile_line("nonsense")).group("rest") == "eager"
    r = report.removed_line(3, 2, 1)
    assert r == "[gpnstar-opt] REMOVED arm=3 hooks=2 models=1 levers=kept" and report.RE_REMOVED.match(r)
    assert report.error_line("out of device memory at batch shape 64x128") == "[gpnstar-opt] ERROR: out of device memory at batch shape 64x128"


def test_the_package_surface_is_one_mode():
    import gpnstar_opt
    assert gpnstar_opt.PREFIX == "[gpnstar-opt]" and set(gpnstar_opt.__all__) >= {"enable", "disable", "apply", "check", "status", "ActivationError"}
    for word in ("fast", "off", "exact-no-dedup", "big"):
        rep = gpnstar_opt.enable(word)
        assert rep == {"active": False, "reason": f"unknown mode '{word}': this kit has one mode, exact"}
        try:
            gpnstar_opt.check(word)
        except gpnstar_opt.ActivationError as e:
            assert "one mode" in str(e)
        else:
            raise AssertionError(word)
    assert gpnstar_opt.status() == {"active": False, "reason": "enable() has not run in this process"}
