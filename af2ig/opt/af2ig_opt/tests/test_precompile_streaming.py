"""L7 streaming precompile (0.7.7): the first length alone and first; later lengths only after it; a design waits on its own length only; a failed
prepare surfaces through wait() as that length's exception (never a hang); loop order unchanged."""
import threading
import time
import unittest

from af2ig_opt import precompile

precompile.HOLD_TIMEOUT_S = 0.3        # the suite never waits the production hold


class TestStreaming(unittest.TestCase):
    def test_first_alone_then_rest_and_wait_own_length(self):
        log, gate = [], threading.Event()
        def prepare(item):
            key, payload = item
            log.append(("begin", key, time.time()))
            if key == 400:
                gate.wait(2.0)                          # the first compile is 'slow': nothing else may begin before it ends
            time.sleep(0.05)
            log.append(("end", key, time.time()))
            return {"source": "traced", "key": key}
        s = precompile.Streaming(prepare, threads=4)
        s.start([(400, "a"), (800, "b"), (800, "b2"), (1200, "c")])       # loop order; the duplicate 800 is one program
        time.sleep(0.3)
        begun = [k for e, k, _ in log if e == "begin"]
        self.assertEqual(begun, [400])                                    # first ALONE: 800 / 1200 not submitted while 400 compiles
        gate.set()
        res, waited = s.wait(400); self.assertEqual(res["key"], 400)      # design 1 waits on its own length only
        time.sleep(0.1); self.assertEqual([k for e, k, _ in log if e == "begin"], [400])   # still held: the loop has not written its first design yet
        s.release()                                                       # first design written -> the background lengths start
        res, _ = s.wait(1200); self.assertEqual(res["key"], 1200)
        summ = s.finish()
        self.assertEqual((summ["n"], summ["order"], summ["failed"]), (3, [400, 800, 1200], []))
        t_end_400 = [t for e, k, t in log if e == "end" and k == 400][0]
        self.assertTrue(all(t >= t_end_400 for e, k, t in log if e == "begin" and k != 400))   # the rest began only after the first was ready
        self.assertLessEqual(summ["dt_first"], summ["dt_all"])
        drained = s.drain(); self.assertEqual(sorted(k for k, *_ in drained), [400, 800, 1200])

    def test_failure_surfaces_by_name_not_a_hang(self):
        def prepare(item):
            key, _ = item
            if key == 800: raise RuntimeError("compile failed for 800")
            return key
        s = precompile.Streaming(prepare, threads=2)
        s.start([(400, None), (800, None), (1200, None)])
        self.assertEqual(s.wait(400)[0], 400)
        with self.assertRaises(RuntimeError):
            s.wait(800, timeout=5)                                        # that length's designs fail by name in the loop
        self.assertEqual(s.wait(1200, timeout=5)[0], 1200)                # later lengths unaffected
        summ = s.finish()
        self.assertEqual([k for k, _ in summ["failed"]], [800])
        self.assertEqual(s.wait(777), (None, 0.0))                        # a length never scheduled: no wait (the loop compiles it lazily, as stock)

    def test_add_rest_after_start_and_host_sized_concurrency(self):
        began = {}
        def prepare(item):
            began[item[0]] = time.time(); time.sleep(0.2 if item[0] == 400 else 0.01); return item[0]
        s = precompile.Streaming(prepare, threads=3)
        s.start([(400, None)])                                            # the driver starts the first compile, THEN featurizes the rest and adds them
        time.sleep(0.05); s.add_rest([(800, None), (1200, None)]); s.close()
        self.assertEqual([s.wait(k, timeout=5)[0] for k in (400, 800, 1200)], [400, 800, 1200])
        self.assertTrue(began[800] >= began[400] + 0.19 and began[1200] >= began[400] + 0.19, began)   # submitted only after the first was ready
        self.assertEqual(s.finish()["order"], [400, 800, 1200])
        self.assertEqual([precompile.background_threads(6, cpus=c) for c in (1, 4, 8, 12, 26, 96)], [1, 1, 5, 6, 6, 6])   # 8 vCPUs: five background compiles (measured: five at once 327 s vs ~108 s each alone)
        self.assertEqual(precompile.background_threads(2, cpus=96), 2)

    def test_first_failure_does_not_block_the_rest(self):
        def prepare(item):
            if item[0] == 400: raise ValueError("first fails")
            return item[0]
        s = precompile.Streaming(prepare, threads=1)
        s.start([(400, None), (800, None)])
        with self.assertRaises(ValueError): s.wait(400, timeout=5)
        self.assertEqual(s.wait(800, timeout=5)[0], 800)
        s.finish()


if __name__ == "__main__":
    unittest.main()
