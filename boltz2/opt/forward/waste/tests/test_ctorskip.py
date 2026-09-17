"""ctorskip (boltz_waste_levers): the RNG-only stand-in for boltz's trunc_normal_init_ leaves the post-construction RNG state (torch CPU, numpy,
python, cuda) and every parameter / buffer of `Boltz2.load_from_checkpoint(strict=True)` identical to the stock build. Needs BOLTZ_CACHE with
boltz2_conf.ckpt (skips otherwise); ~1 min per seed. Run: pytest, or `python3 test_ctorskip.py [seeds...]`."""
import hashlib, os, random, sys
import numpy as np
try:
    import pytest
except ModuleNotFoundError:
    pytest = None
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import boltz_waste_levers as W
CKPT = os.path.join(os.environ.get("BOLTZ_CACHE", ""), "boltz2_conf.ckpt")
if pytest is not None and not os.path.isfile(CKPT):
    pytest.skip("BOLTZ_CACHE/boltz2_conf.ckpt required", allow_module_level=True)


def _kw():
    from dataclasses import asdict
    from boltz.main import Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs, BoltzSteeringParams
    dp = Boltz2DiffusionParams(); dp.step_scale = 1.5
    return dict(predict_args={"recycling_steps": 3, "sampling_steps": 200, "diffusion_samples": 1, "max_parallel_samples": 1, "write_confidence_summary": True,
                              "write_full_pae": True, "write_full_pde": False}, map_location="cpu", diffusion_process_args=asdict(dp), ema=False, use_kernels=True,
                pairformer_args=asdict(PairformerArgsV2()), msa_args=asdict(MSAModuleArgs(subsample_msa=False, num_subsampled_msa=1024, use_paired_feature=True)),
                steering_args=asdict(BoltzSteeringParams()))


def _state():
    st = {"torch": torch.get_rng_state().clone(), "np": np.random.get_state(), "py": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state().clone()
    return st


def _model_digest(m):
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    for k, v in sorted(m.named_buffers()):
        h.update(b"buf:" + k.encode()); h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _build(seed, skip):
    from pytorch_lightning import seed_everything
    from boltz.model.models.boltz2 import Boltz2
    W.remove()
    if skip:
        got = W.apply({"BOLTZ_WASTE": "ctorskip", "BOLTZ_WASTE_VERBOSE": "0"}); assert got == ("ctorskip",) and W.report()["state"]["ctorskip"]["state"] == "on", W.report()["state"]
    seed_everything(seed)
    m = Boltz2.load_from_checkpoint(CKPT, strict=True, **_kw()); m.eval()
    st = _state(); dg = _model_digest(m); calls = W.STATS["ctorskip_calls"]
    W.remove(); del m
    return st, dg, calls


def _check(seed):
    st0, d0, _ = _build(seed, skip=False)
    st1, d1, calls = _build(seed, skip=True)
    assert calls > 400, calls                                                   # 472 trunc_normal_init_ calls at boltz 2.2.1
    assert torch.equal(st0["torch"], st1["torch"]); assert st0["py"] == st1["py"]
    assert (np.asarray(st0["np"][1]) == np.asarray(st1["np"][1])).all() and st0["np"][2:] == st1["np"][2:]
    if "cuda" in st0:
        assert torch.equal(st0["cuda"], st1["cuda"])
    assert d0 == d1


def test_ctorskip_seed0():
    _check(0)


def test_ctorskip_seed3():
    _check(3)


def test_ctorskip_pins_hold_here():
    assert W._ctorskip_check() == "", W._ctorskip_check()


if __name__ == "__main__":
    assert os.path.isfile(CKPT), "BOLTZ_CACHE/boltz2_conf.ckpt required"
    assert W._ctorskip_check() == "", W._ctorskip_check()
    for s in [int(x) for x in (sys.argv[1:] or ["0", "3"])]:
        _check(s); print(f"ok  seed {s}: rng state + parameters identical, stock vs ctorskip", flush=True)
    print("ALL PASSED")
