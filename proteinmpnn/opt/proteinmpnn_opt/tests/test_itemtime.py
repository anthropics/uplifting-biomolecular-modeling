"""The per-item time line with its PEAK line, the STACK line and the OUTPUTS_WRITTEN line (stage.ITEM_LINE_FMT / ITEM_LINE_RE, PEAK_LINE_FMT / PEAK_LINE_RE,
STACK_LINE_FMT / STACK_LINE_RE, OUTPUTS_WRITTEN_FMT / OUTPUTS_WRITTEN_RE, itemtime_worker_source): the lines' grammar (one regex,
the forward / batch group optional so a line carrying only the name and the two stamps parses too), the transform against the carried worker
(anchors once, the result compiles, nothing but the named statements changes, a changed file is refused by name), and the clocks around a
mock batch call — one ITEM line per item at the loop boundary, one STACK line before the first, one OUTPUTS_WRITTEN line after the last with
the item and design counts, the stamps ordered, the share arithmetic, the outputs of the timed code identical to the untimed code's. No torch, no GPU."""
import contextlib
import io
import os
import re
import sys
import types
import unittest

from proteinmpnn_opt import TAG, modes, report, stack, stage

KIT = stack.kit_home()
WORKER = os.path.join(KIT, modes.WORKER_DIR, modes.WORKER)
ITEM_STAMPS_ONLY_FMT = "[" + TAG + "] ITEM name=%s" + stage.ITEM_STAMP_FMT     # a line carrying the name and the two stamps only (the grammar's forward / batch group absent)

# A worker-shaped module: the phase timers, main()'s pipelined batch loop over the worker's records (names, n, t_start, t_end, forward_s, call_s), outputs a pure function of the inputs.
FAKE_WORKER = '''
import time
OUT = []
def sync(): pass
class Worker:
    def __init__(self):
        self.timers = {"featurize": 0.0, "encoder_fwd": 0.0, "sample": 0.0, "rescore": 0.0, "d2h_write": 0.0, "rng_setup": 0.0}
    def run_batch(self, proteins, chain_id_dict, out_dir, B=8):
        t_start = time.time()
        self.timers["featurize"] += 5.0
        self.timers["sample"] += 1.0 * len(proteins); self.timers["rescore"] += 0.5 * len(proteins); self.timers["d2h_write"] += 0.25 * len(proteins)
        OUT.append([p["name"] + ":" + str(sum(map(ord, p["seq"])) * B) for p in proteins])
        t_end = time.time()
        return {"names": [p["name"] for p in proteins], "n": len(proteins), "t_start": t_start, "t_end": t_end, "forward_s": 1.75 * len(proteins), "call_s": t_end - t_start}
    def run_pipelined(self, batches, chain_id_dict, out_dir, B=8):
        for batch in batches:
            yield self.run_batch(batch, chain_id_dict, out_dir, B)
def main(names, bb_batch):
    w = Worker(); chain_id_dict = {}; proteins = [{"name": n, "seq": "ACD" * (i + 1)} for i, n in enumerate(names)]
    class args: out_folder = "/nowhere"; batch_size = 8
    seqs_per_item = args.batch_size                                   # the worker's: batch_size x its (temperature, batch) rounds
    args.bb_batch = bb_batch
    T = {}; t0 = time.time(); offs = {}; o = 0
    for p in proteins: offs[p["name"]] = o; o += len(p["seq"])          # the stream-offset pre-pass: every input read once before the loop
    w.offsets = offs; T["t_offsets_s"] = time.time()-t0; T["stream_total_offset"] = o
    order = list(range(len(proteins))); done = 0
    batches = [order[i:i+args.bb_batch] for i in range(0, len(order), args.bb_batch)]
    for fin in w.run_pipelined([[proteins[j] for j in bi] for bi in batches], chain_id_dict, args.out_folder, args.batch_size):
        done += fin["n"]
    OUT.append("written:%d" % done)
    return done
'''

def _read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def _load(src, name, modules=None):
    mod = types.ModuleType(name)
    saved = {k: sys.modules.get(k) for k in (modules or {})}
    sys.modules.update(modules or {})
    try:
        exec(compile(src, name, "exec"), mod.__dict__)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def _item_rows(text):
    """(rows, index of each ITEM line, index of each STACK line) over the printed lines; each row carries its PEAK line's fields (the line right after it)."""
    lines = text.splitlines()
    items = [i for i, ln in enumerate(lines) if re.search(stage.ITEM_LINE_RE, ln)]
    stacks = [i for i, ln in enumerate(lines) if re.search(stage.STACK_LINE_RE, ln)]
    rows = []
    for i in items:
        r = re.search(stage.ITEM_LINE_RE, lines[i]).groupdict()
        peak = re.search(stage.PEAK_LINE_RE, lines[i + 1]) if i + 1 < len(lines) else None   # one PEAK line right after every ITEM line
        r["peak"] = peak.groupdict() if peak else None
        rows.append(r)
    assert sum(1 for ln in lines if re.search(stage.PEAK_LINE_RE, ln)) == len(items), "one PEAK line per ITEM line"
    return rows, items, stacks


def _written(text):
    """(parsed OUTPUTS_WRITTEN lines, index of each) over the printed lines."""
    lines = text.splitlines()
    idx = [i for i, ln in enumerate(lines) if re.search(stage.OUTPUTS_WRITTEN_RE, ln)]
    return [re.search(stage.OUTPUTS_WRITTEN_RE, lines[i]).groupdict() for i in idx], idx


class TestItemLine(unittest.TestCase):
    def test_grammar(self):
        line = stage.ITEM_LINE_FMT % ("1BRS_AD", 0.123456, 16, 1.9753, 2.5, 4242, 1725000000.125, 1725000002.6251239)
        self.assertTrue(line.startswith(report.PREFIX + " ITEM name=1BRS_AD forward_s=0.1235 "), line)   # the kit's one line tag, then the item, then the seconds
        self.assertEqual(line, "[" + TAG + "] ITEM name=1BRS_AD forward_s=0.1235 batch_items=16 batch_forward_s=1.9753 batch_call_s=2.5000"
                               " pid=4242 t_start=1725000000.125000 t_end=1725000002.625124")
        self.assertIn("forward_s=", line)                                             # the kit executables' spelling carries the forward share
        m = re.search(stage.ITEM_LINE_RE, "1725000000.125 " + line)                   # as a reader of the log finds it: behind a stamp, unanchored
        self.assertEqual(m.groupdict(), {"name": "1BRS_AD", "s": "0.1235", "batch_items": "16", "batch_forward_s": "1.9753", "batch_call_s": "2.5000", "proc": "4242",
                                         "t_start": "1725000000.125000", "t_end": "1725000002.625124"})
        stock_line = ITEM_STAMPS_ONLY_FMT % ("1BRS_AD", 4242, 1725000000.125, 1725000002.6251239)   # the name, the pid and the two stamps only
        self.assertEqual(stock_line, "[" + TAG + "] ITEM name=1BRS_AD pid=4242 t_start=1725000000.125000 t_end=1725000002.625124")
        m = re.search(stage.ITEM_LINE_RE, stock_line)                                  # the same regex: the forward / batch group absent
        self.assertEqual(m.groupdict(), {"name": "1BRS_AD", "s": None, "batch_items": None, "batch_forward_s": None, "batch_call_s": None, "proc": "4242",
                                         "t_start": "1725000000.125000", "t_end": "1725000002.625124"})
        peak = stage.PEAK_LINE_FMT % ("1BRS_AD", 4242, 1.23449, 2.5)                    # the PEAK line that follows every ITEM line
        self.assertEqual(peak, "[" + TAG + "] PEAK item=1BRS_AD pid=4242 alloc_gib=1.234 reserved_gib=2.500")
        self.assertEqual(re.search(stage.PEAK_LINE_RE, "7.5 " + peak).groupdict(), {"item": "1BRS_AD", "proc": "4242", "alloc_gib": "1.234", "reserved_gib": "2.500"})
        self.assertIsNone(re.search(stage.PEAK_LINE_RE, line)); self.assertIsNone(re.search(stage.ITEM_LINE_RE, peak))
        self.assertTrue(stage.ITEM_LINE_FMT.endswith(stage.ITEM_STAMP_FMT))            # one tail, spelled once
        self.assertIsNone(re.search(stage.ITEM_LINE_RE, "[proteinmpnn-opt] ACTIVE mode=exact variant=soluble"))
        self.assertIsNone(re.search(stage.ITEM_LINE_RE, "[" + TAG + "] ITEM name=1BRS_AD forward_s=0.1235 batch_items=16 batch_forward_s=1.9753 batch_call_s=2.5000"))   # no stamps: not a line of this grammar
        self.assertIsNone(re.search(stage.ITEM_LINE_RE, stock_line.replace(" pid=4242", "")))   # no pid: not a line of this grammar

    def test_stack_grammar(self):
        fields = (4242, "2.5.1+cu124", "12.4", "NVIDIA_H100_80GB_HBM3", "9.0", "False", "True", "highest", "False", "False", "unset", ":4096:8")
        line = stage.STACK_LINE_FMT % fields
        self.assertEqual(line, "[" + TAG + "] STACK pid=4242 torch=2.5.1+cu124 cuda=12.4 device=NVIDIA_H100_80GB_HBM3 cc=9.0 tf32_matmul=False cudnn_tf32=True "
                               "f32_matmul_precision=highest cudnn_benchmark=False deterministic_algorithms=False alloc_conf=unset cublas_workspace=:4096:8")
        m = re.search(stage.STACK_LINE_RE, "0.001 " + line)
        self.assertEqual(m.groupdict(), dict(zip(("proc",) + stage.STACK_FIELDS[1:], [str(f) for f in fields])))   # the pid group is `proc`, as the ITEM line's
        cpu = stage.STACK_LINE_FMT % ((4242,) + ("none",) * 9 + ("unset", "unset"))      # a process without torch: every torch field `none`
        self.assertEqual(re.search(stage.STACK_LINE_RE, cpu).group("device"), "none")

    def test_prepass_grammar(self):
        line = stage.PREPASS_FMT % (4242, 1725000000.25, 1725000001.5, 102)
        self.assertEqual(line, "[" + TAG + "] PREPASS pid=4242 t_start=1725000000.250000 t_end=1725000001.500000 n=102")
        self.assertEqual(re.search(stage.PREPASS_RE, "0.5 " + line).groupdict(), {"proc": "4242", "t_start": "1725000000.250000", "t_end": "1725000001.500000", "n": "102"})
        self.assertIsNone(re.search(stage.PREPASS_RE, stage.OUTPUTS_WRITTEN_FMT % (1, 2.0, 3, 4)))

    def test_outputs_written_grammar(self):
        line = stage.OUTPUTS_WRITTEN_FMT % (4242, 1725000002.6251239, 100, 800)
        self.assertEqual(line, "[" + TAG + "] OUTPUTS_WRITTEN pid=4242 t=1725000002.625124 n_items=100 n_designs=800")
        self.assertEqual(re.search(stage.OUTPUTS_WRITTEN_RE, "0.5 " + line).groupdict(), {"proc": "4242", "t": "1725000002.625124", "items_n": "100", "designs": "800"})
        self.assertIsNone(re.search(stage.OUTPUTS_WRITTEN_RE, ITEM_STAMPS_ONLY_FMT % ("x", 1, 2.0, 3.0)))
        self.assertIsNone(re.search(stage.STACK_LINE_RE, line.replace(" cc=9.0", "")))


class TestCarriedTransforms(unittest.TestCase):
    def test_worker_anchors_once_and_compiles(self):
        src = _read(WORKER)
        self.assertEqual(sum(1 for ln in src.splitlines() if stage.ITEMTIME_WORKER_DONE in ln), 1); self.assertEqual(sum(1 for ln in src.splitlines() if stage.ITEMTIME_WORKER_LOOP in ln), 1)
        self.assertEqual(sum(1 for ln in src.splitlines() if stage.ITEMTIME_MAIN_DEF in ln), 1)
        for phase in stage.ITEMTIME_WORKER_PHASES:
            self.assertIn('self.timers["%s"]' % phase, src)
        out = stage.itemtime_worker_source(stage.lowmem_worker_source(src))          # the exact line's chain: lowmem, then the clocks
        self.assertEqual(out.count("] ITEM name=%s forward_s="), 1)
        self.assertEqual(out.count(stage.ITEMTIME_HELPERS), 1); self.assertEqual(out.count(stage.STACK_LINE_FMT), 1)   # the helpers once, in front of main()
        self.assertLess(out.index(stage.ITEMTIME_HELPERS), out.index("\ndef main("))
        self.assertEqual(out.count(stage.ITEMTIME_WORKER_DONE), 1); self.assertEqual(out.count(stage.ITEMTIME_WORKER_LOOP), 1)   # the record statement and the loop header untouched, once
        self.assertEqual(out.count(stage.ITEMTIME_WORKER_PREPASS), 1); self.assertEqual(out.count("] PREPASS pid="), 1)   # the pre-pass statement untouched, its line once, right after it
        self.assertIn("] PREPASS pid=", out.split(stage.ITEMTIME_WORKER_PREPASS, 1)[1].splitlines()[1])
        self.assertIn("_it_stack()", out.split(stage.ITEMTIME_WORKER_LOOP, 1)[0].splitlines()[-2])              # the STACK line on the line before the loop: before the first item is stamped
        body = out.split(stage.ITEMTIME_WORKER_DONE, 1)
        self.assertTrue(body[1].splitlines()[1].strip().startswith("_it_pa, _it_pr = _it_peak()"))              # the peaks read on the line after the record statement, then its ITEM lines
        self.assertIn("for _it_n in fin['names']: print(", body[1].splitlines()[2])
        self.assertNotIn("_it_sync()", out.split(stage.ITEMTIME_WORKER_LOOP, 1)[1].split("_it_outputs_written", 1)[0])   # nothing inside the loop is synchronised by the stage (the worker pipelines its batches)


    def test_changed_files_refused_by_name(self):
        with self.assertRaises(ValueError):
            stage.itemtime_worker_source(_read(WORKER).replace(stage.ITEMTIME_WORKER_DONE, 'done += len(fin["names"])'))
        with self.assertRaises(ValueError):
            stage.itemtime_worker_source(_read(WORKER).replace('self.timers["rescore"]', 'self.timers["score2"]'))
        with self.assertRaises(ValueError):
            stage.itemtime_worker_source(_read(WORKER).replace('T["stream_total_offset"] = o', 'T["stream_total"] = o'))   # the pre-pass statement renamed: no PREPASS line by a guess
        with self.assertRaises(ValueError):
            stage.itemtime_worker_source(_read(WORKER).replace("def main():", "def main():\n    pass\ndef main():", 1))   # main() defined twice: no guess where the helpers go


class TestClocksAroundMockCalls(unittest.TestCase):
    def _check_written(self, text, rows, items, designs_per_item):
        """Exactly one OUTPUTS_WRITTEN line, after the last ITEM line, by this process, not before the last t_end, counting the items and their designs."""
        w, idx = _written(text)
        self.assertEqual(len(w), 1, text); self.assertGreater(idx[0], items[-1])
        self.assertEqual(int(w[0]["proc"]), os.getpid()); self.assertGreaterEqual(float(w[0]["t"]), float(rows[-1]["t_end"]))
        self.assertEqual((int(w[0]["items_n"]), int(w[0]["designs"])), (len(rows), len(rows) * designs_per_item))

    def _check_stamps(self, rows, items, stacks):
        """One STACK line before the first ITEM line; every line names this process; t_start <= t_end per item; the stamps never run backwards across items;
        peaks parse as numbers."""
        self.assertEqual(len(stacks), 1, stacks); self.assertLess(stacks[0], items[0])
        prev = (0.0, 0.0)
        for r in rows:
            self.assertEqual(int(r["proc"]), os.getpid())                              # the designing process's own pid (the fake module runs in this one)
            ts, te = float(r["t_start"]), float(r["t_end"])
            self.assertGreater(ts, 1.6e9); self.assertLessEqual(ts, te)                # epoch seconds, ordered within the item
            if (ts, te) != prev:                                                       # a new span (items of one batch share theirs): it starts after the previous one ended
                self.assertGreaterEqual(ts, prev[1])
            prev = (ts, te)
            self.assertIsNotNone(r["peak"]); self.assertEqual((r["peak"]["item"], r["peak"]["proc"]), (r["name"], r["proc"]))   # its PEAK line: the same item, the same process
            self.assertGreaterEqual(float(r["peak"]["alloc_gib"]), 0.0); self.assertGreaterEqual(float(r["peak"]["reserved_gib"]), float(r["peak"]["alloc_gib"]))

    def test_worker_one_line_per_item_outputs_untouched(self):
        plain = _load(FAKE_WORKER, "fake_worker")
        timed = _load(stage.itemtime_worker_source(FAKE_WORKER, "fake_worker_timed"), "fake_worker_timed")
        names = ["p1", "p2", "p3", "p4", "p5"]
        self.assertEqual(plain.main(names, 2), 5)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(timed.main(names, 2), 5)
        self.assertEqual(timed.OUT, plain.OUT)                                          # the timed code's outputs are the untimed code's
        rows, items, stacks = _item_rows(buf.getvalue())
        self.assertEqual([r["name"] for r in rows], names)                             # one line per item, in processing order
        self.assertEqual([int(r["batch_items"]) for r in rows], [2, 2, 2, 2, 1])       # batches of 2, 2, 1
        for r in rows:                                                                 # forward span = sample + rescore + d2h_write deltas (1.75 s per item here), featurise excluded; the share = span / K
            self.assertAlmostEqual(float(r["batch_forward_s"]), 1.75 * int(r["batch_items"]), places=3)
            self.assertAlmostEqual(float(r["s"]), 1.75, places=3)
            self.assertGreaterEqual(float(r["batch_call_s"]), 0.0)
        self.assertEqual((rows[0]["t_start"], rows[0]["t_end"]), (rows[1]["t_start"], rows[1]["t_end"]))   # the two backbones of one batch carry the batch's stamps
        self._check_stamps(rows, items, stacks)
        self._check_written(buf.getvalue(), rows, items, 8)                              # n_items = the worker's own count, n_designs = items x seqs_per_item
        pre = [re.search(stage.PREPASS_RE, ln).groupdict() for ln in buf.getvalue().splitlines() if re.search(stage.PREPASS_RE, ln)]
        self.assertEqual(len(pre), 1); self.assertEqual((int(pre[0]["proc"]), int(pre[0]["n"])), (os.getpid(), 5))   # one PREPASS line per pass: this process, every input counted
        self.assertLessEqual(float(pre[0]["t_start"]), float(pre[0]["t_end"])); self.assertLessEqual(float(pre[0]["t_end"]), float(rows[0]["t_start"]))   # it closes before the first item opens
        self.assertEqual(timed.OUT[-1], "written:5")                                    # printed after the loop's block: the statement that follows it still runs
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            timed.main(names, 5)
        rows2, items2, stacks2 = _item_rows(buf2.getvalue())
        self.assertEqual((len(rows2), stacks2), (5, []))                                # the STACK line once per process: a second pass in the same module prints none
        self.assertEqual(len(_written(buf2.getvalue())[0]), 1)                          # one OUTPUTS_WRITTEN line per pass


if __name__ == "__main__":
    unittest.main()
