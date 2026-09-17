"""CPU tests of the kit_ho tree (no GPU, no forward): run `python tf/test_spawned_helpers_cpu.py` from opt/kit_ho on the kit's stack
(the stock chrombpnet installed for the PNG form; the writer test needs only numpy).
1. PNG form: the counts-PNG / jsd-PNG helpers spawned at process start (png_workers) write the same bytes as the serial stock calls (counts PNG
   judged), a helper error raises, stop() ends them — skipped when the stock package is not importable.
2. The pre-spawned writer: two items in sequence through the one child interpreter write the same bigWig + stats bytes as the kit's own
   `_writer_proc` in this process; bind() while an item is open refuses; stop() ends the child."""
import os, sys, json, hashlib, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_png_helper_form(n=2048):
    """The PNG tasks on the helpers SPAWNED at process start (png_workers: fresh interpreters, no fork of the main process): the helper's counts PNG bytes and
    its three numbers must equal the serial stock form's in this process (the jsd PNG: present, unjudged); a helper error surfaces as the main process's
    RuntimeError (no fallback); stop() ends both helpers."""
    try:
        import chrombpnet.training.metrics as metrics
    except Exception as e:  # noqa: BLE001
        return "skipped: the stock package is not importable here (%s)" % type(e).__name__
    from chrombpnet_fastkit import png_workers as pw
    w = pw.start()
    rng = np.random.default_rng(0); labels = rng.random(n) * 5; preds = labels + rng.random(n) * 0.5; jsd_pw = rng.random(n) * 0.6; jsd_rnd = rng.random(n) * 0.8
    d = tempfile.mkdtemp(); ps, ph = os.path.join(d, "serial"), os.path.join(d, "helper")
    w["counts"].submit((labels, preds, ph, "All regions provided")); w["jsd"].submit((jsd_pw, jsd_rnd, ph, "All regions provided"))
    sp, pe, mse = metrics.counts_metrics(labels, preds, ps, "All regions provided"); metrics.plot_histogram(jsd_pw, jsd_rnd, ps, "All regions provided")
    vals = w["counts"].result(300); w["jsd"].result(300)
    assert sha(ps + ".counts_pearsonr.png") == sha(ph + ".counts_pearsonr.png"), "counts PNG bytes differ between the serial and helper form"
    assert [float(sp), float(pe), float(mse)] == [float(v) for v in vals], "counts metrics differ"
    assert os.path.getsize(ph + ".profile_jsd.png") > 0
    w["jsd"].submit((None, None, ph, "x"))
    try:
        w["jsd"].result(60); raise AssertionError("a helper error did not raise")
    except RuntimeError as e:
        assert "helper failed" in str(e), e
    pw.stop(); assert all(h.proc.poll() is not None for h in w.values())
    return {"counts_png_sha256_16": sha(ps + ".counts_pearsonr.png")[:16], "n": n, "helpers": "spawned; stopped"}


def test_prespawned_writer_form(n=64, L=1000):
    """The pre-spawned writer (one child interpreter per process, items bound in sequence) writes the same bigWig and stats bytes as the kit's
    own `_writer_proc` run in this process on the same chunk stream — for two items in a row (the resident state inert between items) — and
    stop() ends it; bind() while an item is open is refused."""
    import queue as _queue
    from chrombpnet_fastkit import _writer_proc, prespawn_writer, stop_prespawned_writer
    gs = [("chr1", 5000000)]; regions = [("chr1", 10000 + 2000 * i, 10000 + 2000 * i + L, 10000 + 2000 * i + L // 2) for i in range(n)]
    rng = np.random.default_rng(1); prof = (rng.random((n, L)) * 3).astype(np.float32)
    d = tempfile.mkdtemp(); ps = os.path.join(d, "serial.bw")
    q = _queue.Queue(); q.put((0, prof[:32])); q.put((32, prof[32:])); q.put(None)
    _writer_proc(q, ps, gs, regions, 10, n, ps + ".writer_stats.json", "numpy", "stock", ps + ".stats")
    w = prespawn_writer(); ready = w.wait_ready(300)
    out = {"serial_bw_sha256_16": sha(ps)[:16], "ready": ready, "items": []}
    for k in (1, 2):
        pp = os.path.join(d, "pre%d.bw" % k)
        q2, wp = w.bind(pp, gs, regions, 10, n, pp + ".writer_stats.json", "numpy", "stock", pp + ".stats")
        try:
            w.bind(pp, gs, regions, 10, n, pp + ".writer_stats.json", "numpy", "stock", pp + ".stats"); raise AssertionError("bind() while an item is open did not refuse")
        except RuntimeError as e:
            assert "item is open" in str(e), e
        q2.put((0, prof[:32])); q2.put((32, prof[32:])); q2.put(None); wp.join(300)
        assert sha(pp) == sha(ps), "bigWig bytes differ between the pre-spawned writer (item %d) and the in-process writer" % k
        assert sha(pp + ".stats") == sha(ps + ".stats"), "stats bytes differ (item %d)" % k
        out["items"].append({"item": k, "bw_equal": True, "stats_equal": True})
    assert w.items_done == 2
    stop_prespawned_writer(); assert not w.is_alive()
    out["stopped"] = True
    return out


if __name__ == "__main__":
    res = {"png_helper_form": test_png_helper_form(), "prespawned_writer_form": test_prespawned_writer_form()}
    print(json.dumps(res, indent=1)); print("OK")
