"""boltz2_opt.prefetch — the persistent featurizer of the pipelined worker (registry lever ``prefetch``).

CPU tests (torch required; skipped without it): the parent's ONE generator draw equals the stock DataLoader iterator's (value and the CPU
generator state after it), the helper's per-input worker preamble equals ``_worker_loop``'s for worker id 0 (torch / python / numpy seeds,
thread count, WorkerInfo), the switch words / dispositions / LEVER line grammar. With ``BOLTZ2_TEST_PROCESSED=<dir>`` (a boltz ``processed``
directory of one parsed input) and boltz importable: the featurized batch of the persistent path, in-process AND through a forked helper,
is byte-identical tensor by tensor to the stock loader's.
"""
import hashlib
import os
import random

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from .. import prefetch as PF  # noqa: E402
from .. import prep as PREP    # noqa: E402

PREP.zygote()   # the worker's launcher forks the zygote before CUDA; a test process does the same here (first use), before any test touches CUDA


class _SeedProbe(torch.utils.data.Dataset):
    """__getitem__ reports what a worker's preamble left behind."""

    def __len__(self):
        return 1

    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        return {"torch_initial_seed": torch.initial_seed(), "py_random": random.random(), "np_random": float(np.random.random()),
                "torch_rand": float(torch.rand(())), "threads": torch.get_num_threads(),
                "worker": None if wi is None else (wi.id, wi.num_workers, wi.seed)}


def _first(x):
    return x[0]


def _digest():
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:16]


def test_base_seed_draw_is_the_stock_iterators_draw():
    ds = _SeedProbe()
    torch.manual_seed(20260911)
    it = iter(torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1, pin_memory=False, shuffle=False, collate_fn=_first))
    stock_seed, stock_after = it._base_seed, _digest()
    next(it); del it
    torch.manual_seed(20260911)
    mine = PF.draw_base_seed()
    assert mine == stock_seed and _digest() == stock_after, (mine, stock_seed)
    # num_workers=0 draws the same way (the CLI's --num_workers keeps its meaning; the value only counts forks under stock)
    torch.manual_seed(7)
    it0 = iter(torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0, collate_fn=_first)); s0 = it0._base_seed
    torch.manual_seed(7)
    assert PF.draw_base_seed() == s0


def test_every_inputs_draw_sits_where_the_stock_iterators_does():
    """Input after input: the parent's generator trajectory under the persistent path (one draw per iter()) equals the stock loaders'."""
    ds = _SeedProbe()
    torch.manual_seed(5150)
    stock = []
    for _ in range(4):                                       # four inputs, a fresh stock loader + iterator each (the pipelined worker's make_iter)
        it = iter(torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1, pin_memory=False, shuffle=False, collate_fn=_first))
        stock.append((it._base_seed, _digest())); next(it); del it
        torch.rand(3)                                        # the model's own CPU draws between inputs (any fixed consumer)
    torch.manual_seed(5150)
    mine = []
    for _ in range(4):
        mine.append((PF.draw_base_seed(), _digest()))
        torch.rand(3)
    assert mine == stock, (mine, stock)


def test_worker_preamble_equals_worker_loop_for_worker_0():
    ds = _SeedProbe()
    torch.manual_seed(424242)
    it = iter(torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1, pin_memory=False, shuffle=False, collate_fn=_first))
    base_seed = it._base_seed
    stock = next(it); del it
    keep = torch.get_num_threads()
    try:
        mine = PF.featurize({}, base_seed, 1, dataset=ds, collate_fn=_first)
    finally:
        torch.set_num_threads(keep)
        torch.utils.data._utils.worker._worker_info = None
    assert mine == stock, (mine, stock)
    assert mine["threads"] == 1 and mine["worker"] == (0, 1, base_seed)


def test_switch_words_dispositions_and_line(monkeypatch):
    assert PF.words({}) == [] and not PF.requested({}) and PF.levers_of({}) == []
    assert PF.requested({"BOLTZ_PREFETCH": "persistent"}) and PF.levers_of({"BOLTZ_PREFETCH": "persistent,nopin"}) == ["prefetch"]
    assert PF.problems({"BOLTZ_PREFETCH": "persistent,bogus"}) and "unknown word" in PF.problems({"BOLTZ_PREFETCH": "bogus"})[0]
    assert PF.worker_pipeline(["w.py", "--pipeline", "1"]) == 1 and PF.worker_pipeline(["w.py", "--pipeline", "0"]) == 0 and PF.worker_pipeline(["w.py"]) is None
    monkeypatch.setenv("BOLTZ_PREFETCH", "persistent")
    monkeypatch.setattr(PF.sys, "argv", ["bz_worker.py", "--pipeline", "0"])
    PF.STATE.update(installed=False, disposition=None)
    assert PF.apply() == [] and PF.dispositions() == {"prefetch": "skipped_by_name:pipeline0"}
    ln = PF.line()
    assert ln.startswith("[boltz2-opt] LEVER name=prefetch state=skipped reason=pipeline0 impl=boltz2_opt.prefetch origin=kit") and " " not in ln.split("execution=")[1]
    PF.STATE.update(installed=True, disposition=None, served=3, fallback=0, fallback_by={}, base_seed_draws=3, forked_before_cuda=True, helper_pid=1234, pin=True, bit_compare=None, refused=None, ref_forks=1)
    assert not PF.gate()["ok"] and "bit_compare_not_run" in PF.gate()["why"]          # fail-closed until the first input was bit-compared
    PF.STATE.update(bit_compare="equal:77")
    ln = PF.line(); g = PF.gate()
    assert "state=on" in ln and "served=3" in ln and "forks_avoided=2 ref_forks=1" in ln and "forked_before_cuda=True" in ln and "bit_compare=equal:77" in ln and g["ok"], (ln, g)
    PF.STATE.update(bit_compare="differ:batch.coords:bytes", refused="refused:featurized_differ")
    assert not PF.gate()["ok"] and "refused" in PF.gate()["why"] and "execution=refused:featurized_differ" in PF.line()
    PF.STATE.update(bit_compare="equal:77", refused=None)
    PF.STATE.update(fallback=1, fallback_by={"helper_died:exitcode=-9": 1})
    assert not PF.gate()["ok"] and "fallbacks" in PF.gate()["why"] and "fallback_by=helper_died:exitcode=-9:1" in PF.line()
    PF.STATE.update(installed=False, served=0, fallback=0, fallback_by={}, base_seed_draws=0, forked_before_cuda=None, helper_pid=None, bit_compare=None, refused=None, ref_forks=0)


def test_compare_batches_names_the_difference():
    a = {"x": torch.arange(6, dtype=torch.int64).view(2, 3), "m": torch.tensor([True, False]), "f": torch.tensor([0.1, -0.0]), "l": [1, "s"]}
    b = {"x": a["x"].clone(), "m": a["m"].clone(), "f": torch.tensor([0.1, 0.0]), "l": [1, "s"]}
    assert PF.compare_batches(a, a) == [] and PF.compare_batches(a, b) == ["batch.f:bytes"]          # -0.0 vs 0.0: equal values, different bytes — caught
    b["f"] = a["f"].clone().to(torch.float64)
    assert PF.compare_batches(a, b) == ["batch.f:dtype_or_shape"] and PF.compare_batches(a, {"x": 1}) == ["batch:keys"]


def test_registry_modes_attach_wiring():
    from .. import registry, modes
    from ..worker_launch import ATTACH
    assert "prefetch" in registry.LEVERS and registry.LEVERS["prefetch"]["switch"] == "BOLTZ_PREFETCH" and registry.LEVERS["prefetch"]["tier"] == 1
    assert ATTACH["prefetch"]["module"] == "boltz2_opt.prefetch" and ATTACH["prefetch"]["report_key"] == "prefetch_report"
    for mode in ("exact", "fast"):
        row = modes.resolve(mode)
        assert row["env"].get("BOLTZ_PREFETCH", "").startswith("persistent"), (mode, row["env"].get("BOLTZ_PREFETCH"))
        assert row["attach"][-1] == "prefetch", (mode, row["attach"])   # LAST: its fork follows every other attachment's hook


# ------------------------------------------------------------------------------------------------- the byte-identity leg on a real parsed input
PROCESSED = os.environ.get("BOLTZ2_TEST_PROCESSED")


def _tensor_bytes_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool((a.contiguous().view(torch.uint8) if a.dtype != torch.bool else a.contiguous().to(torch.uint8)).equal(
        b.contiguous().view(torch.uint8) if b.dtype != torch.bool else b.contiguous().to(torch.uint8)))


def _compare_batches(x, y, path="batch"):
    diffs = []
    if isinstance(x, dict):
        if set(x) != set(y):
            return [f"{path}: keys differ {sorted(set(x) ^ set(y))}"]
        for k in sorted(x):
            diffs += _compare_batches(x[k], y[k], f"{path}.{k}")
    elif isinstance(x, torch.Tensor):
        if not _tensor_bytes_equal(x, y):
            diffs.append(f"{path}: tensor differs (dtype {x.dtype}/{y.dtype}, shape {tuple(x.shape)}/{tuple(y.shape)})")
    elif isinstance(x, (list, tuple)):
        if len(x) != len(y):
            return [f"{path}: length {len(x)} != {len(y)}"]
        for i, (a, b) in enumerate(zip(x, y)):
            diffs += _compare_batches(a, b, f"{path}[{i}]")
    else:
        if x != y:
            diffs.append(f"{path}: {x!r} != {y!r}")
    return diffs


def _datamodule():
    from pathlib import Path
    from boltz.data.types import Manifest
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    p = Path(PROCESSED)
    cache = Path(os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz")))
    manifest = Manifest.load(p / "manifest.json")
    return Boltz2InferenceDataModule(manifest=manifest, target_dir=p / "structures", msa_dir=p / "msa", mol_dir=cache / "mols", num_workers=1,
                                     constraints_dir=(p / "constraints") if (p / "constraints").exists() else None,
                                     template_dir=(p / "templates") if (p / "templates").exists() else None,
                                     extra_mols_dir=(p / "mols") if (p / "mols").exists() else None)


@pytest.mark.skipif(not PROCESSED, reason="BOLTZ2_TEST_PROCESSED=<processed dir of one parsed input> names the real-input leg")
def test_featurized_batch_is_byte_identical_to_the_stock_loader():
    pytest.importorskip("boltz")
    dm = _datamodule()
    h = PF.Helper(attaches=())                               # spawned through the zygote (forked at import above, before CUDA), as the worker's attach does
    torch.manual_seed(99)
    it = iter(dm.predict_dataloader())                       # stock: DataLoader(num_workers=1, pin_memory=True) -> forked worker
    base_seed, digest_stock = it._base_seed, _digest()
    stock = next(it); del it
    # in-process replica of worker 0
    keep = torch.get_num_threads()
    try:
        mine = PF.featurize(PF.dataset_kwargs_of(dm), base_seed, 1)
    finally:
        torch.set_num_threads(keep); torch.utils.data._utils.worker._worker_info = None
    diffs = _compare_batches({k: v for k, v in stock.items()}, mine)
    assert not diffs, "\n".join(diffs[:20])
    # through a forked helper + the patched DataModule (the worker's route), pin in parent
    PF.STATE.update(installed=False, disposition=None, served=0, fallback=0, fallback_by={}, base_seed_draws=0, per_item=[], bit_compare=None, refused=None, ref_forks=0)
    try:
        import boltz.data.module.inferencev2 as inf
        PF.STATE.update(helper=h, helper_pid=h.pid, forked_before_cuda=(h.zygote_cuda_initialized is False), pin=True,
                        orig_predict_dataloader=inf.Boltz2InferenceDataModule.predict_dataloader, installed=True)
        torch.manual_seed(99)
        it2 = iter(PF._patched_predict_dataloader(dm))
        assert it2.base_seed == base_seed and _digest() == digest_stock          # the parent's generator: same draw, same state after iter()
        via_helper = next(it2)
        with pytest.raises(StopIteration):
            next(it2)
        diffs = _compare_batches(stock, via_helper)
        assert not diffs, "\n".join(diffs[:20])
        assert all(v.is_pinned() for v in via_helper.values() if isinstance(v, torch.Tensor)) == all(v.is_pinned() for v in stock.values() if isinstance(v, torch.Tensor))
        assert PF.STATE["served"] == 1 and PF.STATE["fallback"] == 0 and str(PF.STATE["bit_compare"]).startswith("equal:") and PF.gate()["ok"], PF.report()
        assert _digest() == digest_stock                                            # the first-input bit-compare left the parent's generator where one draw leaves it
        assert PF.STATE["helper_cuda_initialized"] in (None, False) and h.helper_cuda_initialized is False
        # helper death: the next input goes through the stock loader, named, and nothing hangs
        import signal, time as _time
        os.kill(h.pid, signal.SIGKILL); _time.sleep(0.5)
        t0 = _time.perf_counter()
        torch.manual_seed(99)
        it3 = iter(PF._patched_predict_dataloader(dm))
        b3 = next(it3)
        assert _digest() == digest_stock                                            # the fallback's stock iterator drew the same base_seed at the same position
        assert _time.perf_counter() - t0 < 120 and not _compare_batches(stock, b3)
        assert PF.STATE["fallback"] == 1 and any(k.startswith(("helper_died", "helper_not_alive", "submit:")) for k in PF.STATE["fallback_by"]), PF.STATE["fallback_by"]
        assert not PF.gate()["ok"] and "fallbacks" in PF.gate()["why"]
    finally:
        h.close(); PF.STATE.update(installed=False, helper=None)
