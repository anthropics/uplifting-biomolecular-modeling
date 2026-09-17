"""One process, two token counts: the engine's per-stack ChunkSizeTuner compares the argument record of its last tuning with the next chunked
call's and raises when the confidence stack switches between its batched call form (<= its per-sample token cutoff: z rank 4) and its
per-sample form (above it: z rank 5); the `tuner_guard` cell (opt_core.of3_trunk.tuner_guard on this engine's chunk_utils) answers such a
comparison 'changed' so the tuner re-tunes. Runs on CPU against the engine's own class (skipped where the engine is not importable)."""
import inspect

import pytest

torch = pytest.importorskip("torch")


def test_two_token_counts_in_one_process_re_tune_instead_of_raising():
    cu = pytest.importorskip("openfold3.core.utils.chunk_utils")
    from openfold3_opt.cells import tuner_guard

    def block(s, z, chunk_size=None, **kw):                                                     # a representative block: (s, z) -> (s, z)
        return s + 0, z + 0

    st = tuner_guard.install({tuner_guard.ENV: "1"})
    assert st["state"] == "on" and getattr(cu.ChunkSizeTuner._compare_arg_caches, "_of3opt_tuner_guard", False)
    tuner = cu.ChunkSizeTuner()
    params = inspect.signature(tuner.tune_chunk_size).parameters

    def tune(s, z):                                                                              # the engine's own signature (3.x: min_chunk_size; 0.5.x: max_chunk_size only)
        kw = {"min_chunk_size": 4} if "min_chunk_size" in params else {"max_chunk_size": 16}
        return tuner.tune_chunk_size(representative_fn=block, args=(s.clone(), z.clone()), **kw)
    s400, z400 = torch.zeros(5, 40, 8), torch.zeros(5, 40, 40, 4)                                # <= cutoff: samples folded into the leading dim
    s800, z800 = torch.zeros(1, 80, 8), torch.zeros(1, 1, 80, 80, 4)                             # > cutoff: one sample, batch dims kept
    n0 = sum(st["resets"].values())
    assert isinstance(tune(s400, z400), int)
    rec400 = tuner.cached_arg_data
    assert isinstance(tune(s800, z800), int) and tuner.cached_arg_data != rec400                 # re-tuned for the per-sample form, no exception
    assert isinstance(tune(s400, z400), int) and tuner.cached_arg_data == rec400                 # and back
    assert isinstance(tune(s400, z400), int)                                                     # the same form again: the engine's own 'consistent'
    resets = sum(st["resets"].values()) - n0
    assert resets in (0, 2), st["resets"]                                                        # 2 on an engine whose comparison raises on the rank change (3.x); 0 where it answers itself
