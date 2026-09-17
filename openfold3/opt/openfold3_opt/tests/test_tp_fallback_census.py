"""A tp rank names a per-call fallback once, in ONE grammar (`[rowpair rK] FALLBACK <name>=<word>: <why>`, tp_rowpair.core.log_fallback): counted in
the rank's own EXIT census (report.FALLBACK_SOURCES label `tp`) and in the launcher's, which sums the ranks' transcript lines (tp.parse_rank_log)."""
import sys, types


def test_a_named_rank_fallback_reaches_the_transcript_and_both_exit_censuses(tmp_path, monkeypatch):
    from openfold3_opt import report, tp
    from openfold3_opt.tp_rowpair import core as C
    lines = []
    monkeypatch.setattr(C, "comm", lambda: types.SimpleNamespace(log=lines.append, rank=1))
    monkeypatch.setattr(C, "_FALLBACKS", {})
    C.log_fallback("sample_loop_pocket", "one_pass_forced", "why")
    C.log_fallback("sample_loop_pocket", "one_pass_forced", "why again")
    assert lines == ["FALLBACK sample_loop_pocket=one_pass_forced: why", "FALLBACK sample_loop_pocket=one_pass_forced: why again"]
    assert C.fallbacks() == {"sample_loop_pocket=one_pass_forced": 2}
    assert report.fallback_census().get("tp:sample_loop_pocket=one_pass_forced") == 2          # the rank process's EXIT census reads its own record
    log = tmp_path / "rank1.log"                                                              # the launcher's side: the rank transcript's lines, in the comm's spelling
    log.write_text("".join(f"[rowpair r1] {l}\n" for l in lines))
    assert tp.parse_rank_log(str(log))["fallbacks"] == {"sample_loop_pocket=one_pass_forced": 2}
    assert [s[1] for s in report.FALLBACK_SOURCES if s[0] in ("openfold3_opt.tp", "openfold3_opt.tp_rowpair.core")] == ["tp", "tp"]
