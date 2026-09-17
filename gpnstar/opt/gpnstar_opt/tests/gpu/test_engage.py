"""GPU engage test — skipped without CUDA, without the pinned stack, or without the primary checkpoint in the cache (HF_HOME).
The primary model (v100-200m) at its pinned revision; one real-shaped 8x128 batch built from the shipped fixture (accel/inputs.make_batch:
row 0 the real alignment window, the other rows shifted copies with sparse substitutions).
  exact   kit logits == stock logits (torch.equal) under the documented invocation's numerics — TF32 matmuls (what `gpn star … --tf32`
          sets, through torch's precision interface; under IEEE fp32 the same checks run and identity is reported, not certified); mean forward time over ITER forwards at least MIN_SPEEDUP x faster than stock under the same
          numerics; the K/V route reported for each shape is the one the layer-0 check settled on (de-dup where enough rows
          de-duplicate, else the static unified K/V route), with no reject.
  engage  apply() on a CUDA model prints ACTIVE + one LEVER line per lever, and a KV line after the first forward; a CPU-resident model is
          refused by name.
"""
import os
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
pytest.importorskip("gpn")
pytest.importorskip("transformers")

from gpnstar_opt import registry, report, stack                   # noqa: E402
from gpnstar_opt.accel import api as accel_api                    # noqa: E402
from gpnstar_opt.accel import inputs as accel_inputs              # noqa: E402
from gpnstar_opt.accel import load_model                          # noqa: E402

KEY = registry.PRIMARY
B, L = 8, 128
B_SPEED = 64                    # the speed floor is held at a saturating batch; at B=8 the forward is launch-bound and the gain depends on the host
N = registry.models()[KEY]["n_species"]
CENTER = L // 2
ITER = 20
MIN_SPEEDUP = 1.1               # kit vs stock under the SAME numerics at B_SPEED (fp32 and TF32 alike); the numbers themselves are README's, not this floor's
FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "fixture_v100.npz")


def _load(device="cuda"):
    """The pinned primary checkpoint exactly as upstream builds it (fp32, eval): from the staged snapshot directory under HF_HOME when it is
    there (what `gpn star … --model-path <snapshot>` loads — no Hub access needed), else by repository id at the pinned revision."""
    snap = registry.snapshot_dir(KEY)
    try:
        if os.path.isdir(snap):
            from gpn import register_auto_classes
            from transformers import AutoModelForMaskedLM
            register_auto_classes("star")
            model = AutoModelForMaskedLM.from_pretrained(snap).eval()
            return model.to(device) if device is not None else model
        return load_model(KEY, device=device)
    except Exception as e:  # noqa: BLE001 — the snapshot is not in the cache, or the hub is offline without it
        pytest.skip(f"the {KEY} checkpoint is not loadable here ({type(e).__name__}: {str(e)[:160]}); stage it: run.sh install --weights DIR, then HF_HOME=DIR")


@pytest.fixture(scope="module")
def batch():
    tok = accel_inputs.load_fixture_tokens(FIXTURE)
    assert tok.shape == (128, N), tok.shape
    return accel_inputs.make_batch(B, L, N, fixture_tokens=tok, seed=0, device="cuda")


@pytest.fixture(scope="module")
def batch_speed():
    tok = accel_inputs.load_fixture_tokens(FIXTURE)
    return accel_inputs.make_batch(B_SPEED, L, N, fixture_tokens=tok, seed=1, device="cuda")


@pytest.fixture(scope="module")
def stock():
    return _load()


def _fwd(model, batch):
    with torch.inference_mode():
        return model(**batch).logits


def _time(model, batch, n=ITER):
    _fwd(model, batch)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        _fwd(model, batch)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def _matmul_precision(value):
    """Set the fp32 CUDA-matmul precision the way transformers' `--tf32` does on this stack (torch's precision interface); returns the old value."""
    mm = torch.backends.cuda.matmul
    old = str(getattr(mm, "fp32_precision", "none"))
    mm.fp32_precision = value
    return old


def _settled_route(ks):
    """The route word the levers' own report says the layer-0 check settled on for the last shape (validation_log mode → KV route word)."""
    log = (ks.get("raw") or {}).get("validation_log") or []
    used = (ks.get("raw") or {}).get("last_mode_used")
    word = {"dedup": "dedup", "sdedup": "dedup", "p3b": "unifiedkv"}
    return word.get(used) or (word.get(log[-1]["mode"]) if log else None)


@pytest.mark.parametrize("numerics", ["ieee", "tf32"])
def test_exact_is_bitwise_and_faster_and_routes_dedup(stock, batch, batch_speed, numerics):
    old = _matmul_precision(numerics)             # the CALLER's numerics: the levers set none and follow these
    try:
        kit = _load()
        accel_api.make_exact(kit, dedup=True)
        diffs, routes = {}, {}
        for b in (batch, batch_speed):                # identity is checked for EVERY shape before anything else is asserted
            nb = b["input_ids"].shape[0]
            ref = _fwd(stock, b)
            out = _fwd(kit, b)                      # the first forward of this shape runs the layer-0 K/V check and settles the route
            assert out.shape == ref.shape and out.dtype == ref.dtype == torch.float32
            again = _fwd(kit, b)                    # and again on the settled route
            diffs[nb] = (float((out - ref).abs().max()), float((again - ref).abs().max()))
            ks = stack.kv_state(kit)
            routes[nb] = (ks["route"], ks["rejects"], ks["pairs"], _settled_route(ks))
        if numerics == "tf32":                        # the certified invocation (README Run block: `--tf32`): bit-identical at every shape, first and settled forward
            assert all(d == (0.0, 0.0) for d in diffs.values()), f"numerics={numerics}: max|dlogits| first/settled forward per batch size = {diffs}"
        for nb, (route, rejects, pairs, settled) in routes.items():
            # the route is the policy's: de-dup where the layer-0 check kept it for the shape, else the static unified K/V route — never stock here
            assert route == settled and route in ("dedup", "unifiedkv") and rejects == 0 and pairs >= 1, (nb, routes)
        assert stack.tf32_state() is (numerics == "tf32")
        for b, floor in ((batch, None), (batch_speed, MIN_SPEEDUP)):
            t_stock, t_kit = _time(stock, b), _time(kit, b)
            speedup = t_stock / t_kit
            nb = b["input_ids"].shape[0]
            print(f"exact numerics={numerics} B={nb} L={L} N={N}: stock {t_stock * 1e3:.2f} ms/forward, kit {t_kit * 1e3:.2f} ms/forward, speedup {speedup:.2f}x over {ITER} forwards")
            assert floor is None or speedup > floor, (nb, speedup)
        del kit
        torch.cuda.empty_cache()
        if numerics == "ieee":                        # identity with stock is certified for the documented `--tf32` invocation only (STOCK.md, Numerics)
            pytest.skip("bit-identity with stock is certified for the documented invocation (--tf32, README Run block); under IEEE fp32 matmuls the "
                        f"same levers run without that certification — shapes, routes and speed checked above; observed max|dlogits| first/settled per batch size: {diffs}")
    finally:
        _matmul_precision(old if old in ("ieee", "tf32") else "ieee")


def test_apply_prints_active_lever_and_kv_lines(batch, capfd):
    import gpnstar_opt
    stack._reset_for_tests()
    kit = _load()
    rec = gpnstar_opt.apply(kit)
    _fwd(kit, batch)
    torch.cuda.synchronize()
    err = capfd.readouterr().err.splitlines()
    active = [s for s in err if report.RE_ACTIVE.match(s)]
    levers = [s for s in err if report.RE_LEVER.match(s)]
    kv = [s for s in err if report.RE_KV.match(s)]
    assert len(active) == 1, err
    assert f" mode=exact levers={'+'.join(report.LEVERS)} " in active[0] and f" model={registry.label(KEY)} " in active[0]
    assert f" tf32={report.tf32_field(stack.tf32_state())}" in active[0] and " tf32=unknown" not in active[0]
    assert [report.RE_LEVER.match(x).group("name") for x in levers] == list(report.LEVERS) and all(" state=on " in x for x in levers)
    m = report.RE_KV.match(kv[0]) if kv else None
    assert m and m.group("route") in ("dedup", "unifiedkv") and m.group("shape") == f"{B}x{L}" and m.group("reason") == "kvcheck", kv
    assert m.group("route") == _settled_route(stack.kv_state(kit)), (kv, stack.kv_state(kit)["raw"])
    assert rec["model_key"] == KEY and rec["levers"] == list(report.LEVERS)
    assert gpnstar_opt.status()["kvcheck"]["rejects"] == 0
    removed = gpnstar_opt.disable()
    assert removed["hooks"] == 2 and removed["models"] == 1 and removed["levers"] == "kept"
    assert report.RE_REMOVED.match([s for s in capfd.readouterr().err.splitlines() if " REMOVED " in s][0])
    assert gpnstar_opt.status()["active"] is False
    del kit
    torch.cuda.empty_cache()


def test_a_cpu_resident_model_is_refused_by_name():
    cpu_model = _load(device="cpu")
    with pytest.raises(Exception) as ex:
        stack.apply(cpu_model, rep={})
    assert "move it to the GPU first" in str(ex.value)


def test_memory_guard_routes_to_stock_projections_by_name():
    """A forced out-of-memory inside the reduced K/V path (fault injection) lands the shape on the stock projections: the output
    stays bitwise the stock's, kv_mode_report names the shape under memory_fallbacks, and the route override is 'stock'."""
    import copy
    import torch
    from gpnstar_opt.accel import api, inputs
    from gpnstar_opt.accel.patches import core_model, kv_mode_report
    stock = _load()
    fx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "fixture_v100.npz")
    b = inputs.make_batch(8, 128, 100, fixture_tokens=inputs.load_fixture_tokens(fx), seed=3, device="cuda")
    with torch.inference_mode():
        ref = stock(**b).logits
    m = copy.deepcopy(stock)
    api.make_exact(m, dedup=False)
    core_model(m)._exact_state.debug_force_oom = 1
    with torch.inference_mode():
        out = m(**b).logits
        again = m(**b).logits
    rep = kv_mode_report(m)
    assert torch.equal(out, ref) and torch.equal(again, ref)
    assert rep["memory_fallbacks"] == {8 * 128 * len(core_model(m)._exact_state.clade_species): 1}
    assert set(rep["override_per_stock_rows"].values()) == {"stock"} and rep["last_mode_used"] == "stock"


def test_memory_plan_steps_aside_by_name_before_allocating():
    """With (pretend) no memory obtainable the reduced route is not even built: the shape runs the stock projections by name
    (kv_mode_report memory_fallbacks + memory_plans), the output is the stock's bit for bit; another shape with memory de-dups again."""
    import copy
    import torch
    from gpnstar_opt.accel import api, inputs
    from gpnstar_opt.accel.patches import core_model, kv_mode_report
    stock = _load()
    fx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "fixture_v100.npz")
    tok = inputs.load_fixture_tokens(fx)
    b8 = inputs.make_batch(8, 128, 100, fixture_tokens=tok, seed=5, device="cuda")
    b4 = inputs.make_batch(4, 128, 100, fixture_tokens=tok, seed=6, device="cuda")
    with torch.inference_mode():
        ref8, ref4 = stock(**b8).logits, stock(**b4).logits
    m = copy.deepcopy(stock)
    api.make_exact(m, dedup=True)
    st = core_model(m)._exact_state
    st.debug_memory_budget = 1
    with torch.inference_mode():
        out8 = m(**b8).logits
    rep = kv_mode_report(m)
    m8 = 8 * 128 * len(st.clade_species)
    assert torch.equal(out8, ref8)
    assert rep["memory_fallbacks"] == {m8: 1} and rep["override_per_stock_rows"] == {m8: "stock"} and rep["last_mode_used"] == "stock"
    assert rep["memory_plans"][m8]["budget_bytes"] == 1 and rep["memory_plans"][m8]["need_bytes"] > 1 and rep["validation_log"] == []
    st.debug_memory_budget = None
    with torch.inference_mode():
        out4 = m(**b4).logits
    rep = kv_mode_report(m)
    assert torch.equal(out4, ref4)
    m4 = 4 * 128 * len(st.clade_species)
    assert rep["last_mode_used"] in ("dedup", "p3b") and m4 not in (rep["memory_fallbacks"] or {}) and rep["validation_log"][-1]["ok"] is True, rep
