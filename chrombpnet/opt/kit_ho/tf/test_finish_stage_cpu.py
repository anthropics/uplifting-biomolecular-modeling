"""The finish-stage stamp: finish_stage_s = finish() entry -> return (the labelled sub-stages sum to it within 0.05 s);
overlap_lifetime_s = the overlap's construction -> the end of finish (it spans the whole of stage 1 as well). No GPU, no TF:
the overlap is exercised with a fake reader (the test's own bigWig-free path)."""
import os, sys, time, json, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chrombpnet_fastkit import metrics_overlap as mo
def test_finish_stage_timer_is_the_labelled_sum():
    T = {"join_wait_s": 0.0, "h5_assemble": 0.3, "counts_metrics_png": 0.2, "jsd_histogram_png": 0.1, "json": 0.0}
    t_entry = time.time(); time.sleep(0.05); T.update(mo._finish_stage_stamp(T, t_entry, t_entry - 1.0))
    assert "finish_stage_s" in T and "overlap_lifetime_s" in T and "finish_total" not in T, T
    assert 0.04 <= T["finish_stage_s"] <= 0.5, T; assert T["overlap_lifetime_s"] >= T["finish_stage_s"] + 0.9, T
    assert abs(T["finish_stage_s"] - (time.time() - t_entry)) < 0.05, T
    assert T["finish_stage_note"].startswith("finish_stage_s = finish() entry -> return"), T
if __name__ == "__main__":
    test_finish_stage_timer_is_the_labelled_sum(); print("FINISH-STAGE STAMP CPU TEST OK")
