"""CPU tests of lever output_overlap (the multimer batch generator re-stated: forwards on the CALLER's thread in the stock order, the batch's host
statements — _make_outputs + ranking + the writer's text — in ONE spawn-context worker process, yields in the stock order), replayed against the
STOCK generator on a scripted runner (real _normalize_inputs / _bucket_inputs / _iter_batch, scripted features / forward / outputs): identical
yields (names, keys, ranking, arrays, distogram slices) over several buckets x batch sizes x 2 seeds; featurize + forward on the caller's thread,
outputs in another pid; NO thread created in the kit process; in-flight bound; a forward that raises first yields the finished batches, then
re-raises; an exception in the worker's statements re-raised in the caller (typed, worker traceback attached) and the gate refused; a worker that
dies -> the batch recomputed on the caller by name (files complete, gate refused); early close then a fresh call; AFO_OUTPUT_OVERLAP=0 = the
stock body counted `disabled`; the writer memo (to_mmcif / confidence_scores precomputed, served byte-for-byte, pickles intact); install refuses
by name when fold_iter_batch is already wrapped; the digest table matches stock/src; registry / modes / installers (exact, fast AND big rows);
the monomer runner counted `runner:monomer`. The stock host modules (torch, gemmi, numba, scipy, einops) are a skip BY REASON where absent."""
import multiprocessing
import os
import pickle
import threading
import time
import types

import numpy as np
import pytest

from atlasfold_opt.hooks import output_overlap as OO


def _rm():
    """(runner_multimer, runner) — or a skip by reason: the stock host stack is not importable in this interpreter."""
    try:
        from atlasfold import runner_multimer as RM
        from atlasfold import runner as R
    except ImportError as e:                                              # ModuleNotFoundError included: torch / gemmi / numba / scipy / einops absent
        pytest.skip(f"atlasfold's stock host modules are not importable here ({type(e).__name__}: {e}); the kit's CPU image carries them")
    return RM, R


# ---- the scripted `_make_outputs` statements: MODULE-LEVEL so they travel to the worker process by name (as the stock staticmethod does)
def scripted_make_outputs(*, complex_input, out, batch_idx, num_samples, seed):
    res = {}
    for i in range(num_samples):
        score = float(out["plddt"][batch_idx, i, : complex_input.num_residues].mean())
        res[(int(seed), i)] = types.SimpleNamespace(name=complex_input.name, num_residues=complex_input.num_residues, ranking_score=score, seed=int(seed), sample=i,
                                                    coords=out["sample_coords"][batch_idx, i, : complex_input.num_residues].copy(), pid=os.getpid())
    return res


def raising_make_outputs(*, complex_input, out, batch_idx, num_samples, seed):
    if multiprocessing.parent_process() is not None:                      # in the worker only (the run's last batch is the caller's by design and must not mask it)
        raise FloatingPointError(f"scripted outputs failure on {complex_input.name}")
    return scripted_make_outputs(complex_input=complex_input, out=out, batch_idx=batch_idx, num_samples=num_samples, seed=seed)


def crashing_make_outputs(*, complex_input, out, batch_idx, num_samples, seed):
    if multiprocessing.parent_process() is not None:                      # in the worker: die without a word; on the caller (the recompute by name): the scripted statements
        os._exit(7)
    return scripted_make_outputs(complex_input=complex_input, out=out, batch_idx=batch_idx, num_samples=num_samples, seed=seed)


def marker_writer(out_dir, output, format, save_confidence_arrays=False, save_distogram=False):   # MODULE-LEVEL: travels to the worker by name, like the CLI's writer
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"early.{format}.txt").write_text(f"{output.name} {time.time():.6f} pid={os.getpid()}\n")
    return {"record": output.name, "pid": os.getpid()}                     # stands for the CLI writer's best_record


class _Scripted:
    """Instance-level stand-ins for featurize + forward (recording (thread, call, args)) and a module-level `_make_outputs`."""
    def __init__(self, runner, fwd_sleep=0.0, fail_at=None, post=scripted_make_outputs):
        self.log, self.lock, self.n_fwd = [], threading.Lock(), 0
        self.fwd_sleep, self.fail_at = fwd_sleep, fail_at
        runner._make_batch_features = self.make_batch_features
        runner.model_run = self.model_run
        runner._make_outputs = post

    def _rec(self, *w):
        with self.lock:
            self.log.append((threading.current_thread().name,) + w)

    def make_batch_features(self, complexes, bucket_length):
        self._rec("feat", tuple(c.name for c in complexes), int(bucket_length))
        return {"names": [c.name for c in complexes], "L": int(bucket_length), "aatype": np.zeros((len(complexes), int(bucket_length)), np.int64)}

    def model_run(self, feat, *, seed, num_samples, num_recycles, mlm_prob, sampling_config, return_distogram=False):
        self._rec("fwd", tuple(feat["names"]), int(seed))
        self.n_fwd += 1
        if self.fail_at is not None and self.n_fwd == self.fail_at:
            raise FloatingPointError("scripted forward failure")
        time.sleep(self.fwd_sleep)
        B, L = len(feat["names"]), feat["L"]
        rs = np.random.RandomState(seed * 7919 + L * 31 + B)
        out = {"plddt": rs.rand(B, num_samples, L).astype(np.float32), "sample_coords": rs.rand(B, num_samples, L, 3).astype(np.float32)}
        if return_distogram:
            out["distogram.logits"] = rs.rand(B, L, L, 3).astype(np.float32); out["distogram.boundaries"] = np.linspace(2.0, 22.0, 2).astype(np.float32)
        return out


def _new_runner(**kw):
    RM, _R = _rm()
    cls = RM.MultimerFoldingRunner
    r = cls.__new__(cls)                       # no model: the generator only needs the batching helpers + the scripted statements
    r.model, r.device = None, "cpu"
    return RM, cls, r, _Scripted(r, **kw)


def _inputs(RM):
    seqs = {"a40": "A" * 40, "b45": "C" * 45, "c130": "D" * 130, "d61": "E" * 61, "e260": "F" * 120 + "G" * 140, "f33": "H" * 33, "g600": "K" * 600}
    return [RM.MultimerInput(n, [s[: len(s) // 2], s[len(s) // 2:]]) for n, s in seqs.items()]


def _summ(batches):
    """comparable summary of a list of yielded batch lists"""
    out = []
    for batch in batches:
        for fo in batch:
            keys = sorted(fo.outputs)
            out.append((fo.name, fo.length, tuple(fo.ranking), tuple((k, round(fo.outputs[k].ranking_score, 7), fo.outputs[k].coords.tobytes()) for k in keys),
                        tuple((s, v.tobytes()) for s, v in sorted(fo.distogram_logits.items())), None if fo.distogram_boundaries is None else fo.distogram_boundaries.tobytes()))
    return out


def _pids(batches):
    return {s.pid for batch in batches for fo in batch for s in fo.outputs.values()}


KW = dict(num_samples=3, seeds=[11, 12], num_recycles=1, mlm_prob=0.2, sampling_config=None, length_buckets=None, max_tokens_per_batch=256, return_distogram=True)


def _reference(n_inputs=None, **over):
    """the STOCK generator's yields on the scripted runner (the innermost stock function whether or not the lever is installed)"""
    RM, cls, r0, s0 = _new_runner()
    stock = getattr(cls.fold_iter_batch, "__wrapped_stock__", cls.fold_iter_batch)
    return RM, list(stock(r0, _inputs(RM)[:n_inputs], **dict(KW, **over))), s0


@pytest.fixture
def installed(monkeypatch):
    RM, R = _rm()
    monkeypatch.delenv(OO.ENV_SWITCH, raising=False)
    assert not hasattr(RM.MultimerFoldingRunner.fold_iter_batch, "__wrapped_stock__")
    ins = OO.install("exact", "atlasfold-opt", {})
    assert ins.applied, ins.reason
    try:
        yield ins
    finally:
        OO.uninstall(stop=False)                                           # the ONE worker process serves the whole module (stopped in teardown_module)


def teardown_module(module):
    OO.stop_worker()


def test_digest_table_matches_stock_src():
    _rm()
    assert OO.changed_sources() == {}


def test_pipeline_yields_the_stock_batches_bit_for_bit(installed):
    RM, ref, s0 = _reference()
    threads_before = {t.ident for t in threading.enumerate()}
    RM, cls, r1, s1 = _new_runner(fwd_sleep=0.02)
    got = list(r1.fold_iter_batch(_inputs(RM), **KW))
    assert len(ref) >= 3                                                   # several buckets / batch sizes
    assert _summ(got) == _summ(ref)
    calls = lambda log: [e[1:] for e in log]                                # the same featurize / forward statements, same arguments, same ORDER ...
    assert calls(s1.log) == calls(s0.log)
    main = threading.current_thread().name                                 # ... all on the caller's thread
    assert {e[0] for e in s1.log} == {main}
    assert os.getpid() not in _pids(got[:-1]) and len(_pids(got[:-1])) == 1      # the outputs statements ran in the worker process ...
    assert _pids(got[-1:]) == {os.getpid()}                                # ... except the run's LAST batch: the caller's, at once (nothing follows it to overlap with)
    assert {t.ident for t in threading.enumerate()} <= threads_before      # no thread was created in this process
    led = installed.facts["ledger"]
    assert led.served == len(ref) and led.get("queued") == len(ref) - 1 and led.get("last") == 1 and not led.get("inline")
    assert 1 <= led.get("max_inflight") <= OO.MAX_INFLIGHT + 1
    assert led.get("post_s") > 0 and led.get("worker") == "process" and led.get("threads") == 0
    assert installed.gates[0]().ok
    line = installed.lines[0]()
    for tok in ("output_overlap", "worker=process", "queued=", "last=", "overlapped_s=", "wait_s=", "boot_wait_s=", "threads=0"):
        assert tok in line, (tok, line)


def test_single_input_single_seed(installed):
    RM, ref1, _ = _reference(1, seeds=5, return_distogram=False)
    RM, cls, r1, _s = _new_runner()
    got = list(r1.fold_iter_batch(_inputs(RM)[:1], **dict(KW, seeds=5, return_distogram=False)))
    assert len(got) == 1 and _summ(got) == _summ(ref1)


def test_forward_error_yields_the_finished_batches_then_raises(installed):
    RM, ref, _ = _reference()
    RM, cls, r, s = _new_runner(fail_at=3)                                # 2 seeds per batch: forward #3 is batch 2's first -> batch 1 is complete
    gen = r.fold_iter_batch(_inputs(RM), **KW)
    got = []
    with pytest.raises(FloatingPointError, match="scripted forward failure"):
        for b in gen:
            got.append(b)
    assert len(got) == 1 and _summ(got) == _summ(ref[:1])                 # stock had yielded batch 1 before forward #3: so does the pipeline, then the error


def test_worker_exception_is_reraised_in_the_caller_and_named(installed):
    RM, cls, r, s = _new_runner(post=raising_make_outputs)
    with pytest.raises(FloatingPointError) as ei:
        list(r.fold_iter_batch(_inputs(RM), **KW))
    text = str(ei.value) + "".join(getattr(ei.value, "__notes__", []) or [])
    assert "scripted outputs failure" in text and "worker" in text
    led = installed.facts["ledger"]
    assert led.errors and not installed.gates[0]().ok                      # never a silent pass: FINAL gates=refused


def test_worker_death_recomputes_on_the_caller_by_name(installed):
    RM, ref, _ = _reference()
    RM, cls, r, s = _new_runner(post=crashing_make_outputs)
    got = list(r.fold_iter_batch(_inputs(RM), **KW))
    assert _summ(got) == _summ(ref)                                        # every batch delivered (the caller recomputed what the dead worker held) ...
    led = installed.facts["ledger"]
    assert any(k.startswith("worker:") for k in led.fallbacks) and led.get("inline") >= 1
    assert not installed.gates[0]().ok                                     # ... and the run says so at exit (unexpected fallback -> gate refused)
    w = OO.start_worker(led)                                               # a dead worker is replaced on the next start
    assert w is not None and w.usable()


def test_early_close_then_a_fresh_call(installed):
    RM, ref, _ = _reference()
    RM, cls, r, s = _new_runner(fwd_sleep=0.01)
    gen = r.fold_iter_batch(_inputs(RM), **KW)
    first = next(gen)
    assert len(first) >= 1
    gen.close()
    assert s.n_fwd <= 2 * (OO.MAX_INFLIGHT + 2)                            # forwards issued <= (yielded + in flight + the one being handed) batches x 2 seeds
    RM, cls, r2, s2 = _new_runner()
    got = list(r2.fold_iter_batch(_inputs(RM), **KW))                      # the abandoned batches' results never leak into a later call (job ids are per interpreter)
    assert _summ(got) == _summ(ref)
    assert not any(t.name == OO.WORKER_NAME for t in threading.enumerate())


def test_stock_validation_errors_verbatim(installed):
    RM, cls, r, _s = _new_runner()
    with pytest.raises(ValueError, match="No inputs provided"):
        list(r.fold_iter_batch([], **KW))
    with pytest.raises(ValueError, match="No seeds provided"):
        list(r.fold_iter_batch(_inputs(RM), **dict(KW, seeds=[])))
    with pytest.raises(ValueError, match="num_samples must be positive"):
        list(r.fold_iter_batch(_inputs(RM), **dict(KW, num_samples=0)))


def test_env_switch_runs_the_stock_body_counted(installed, monkeypatch):
    monkeypatch.setenv(OO.ENV_SWITCH, "0")
    assert OO.enabled() is False
    RM, cls, r, s = _new_runner()
    got = list(r.fold_iter_batch(_inputs(RM), **KW))
    assert len(got) >= 3 and {e[0] for e in s.log} == {threading.current_thread().name} and _pids(got) == {os.getpid()}   # everything here = the stock body
    assert installed.facts["ledger"].fallbacks.get("disabled") == 1


def test_writer_memo_serves_the_precomputed_text_byte_for_byte(installed):
    RM, _R = _rm()
    from atlasfold.common import protein
    rs = np.random.RandomState(3)
    chains = [protein.Protein.create(name=n, sequence=q, coordinates=(rs.rand(len(q), 14, 3) * 20).astype(np.float32), b_factors=(rs.rand(len(q)) * 100).astype(np.float32))
              for n, q in (("A", "MKV"), ("B", "GAS"))]
    N = 6
    sample = RM.ProteinMultimerOutput(name="t", chains=chains, seed=1, sample_index=0, plddt=rs.rand(N).astype(np.float32), pae=rs.rand(N, N).astype(np.float32),
                                      pde=rs.rand(N, N).astype(np.float32), ptm=0.5, iptm=0.4, fraction_disordered=0.1, chain_ptm=[0.5, 0.6], interface_iptm={(0, 1): 0.3})
    stock_text = protein.ProteinMultimer.to_mmcif(sample, model="multimer")
    assert OO.MEMO_ATTR not in sample.__dict__ and sample.to_mmcif(model="multimer") == stock_text          # no memo: the stock call through the wrapper
    assert OO._precompute(sample, "cif") == 1 and "confidence_scores" in sample.__dict__
    back = pickle.loads(pickle.dumps(sample))                                                               # the worker's object as it arrives on the caller
    led = installed.facts["ledger"]; n0 = int(led.get("memo_served", 0) or 0)
    assert back.to_mmcif(model="multimer") == stock_text and int(led.get("memo_served")) == n0 + 1
    assert back.to_mmcif(model="monomer") == protein.ProteinMultimer.to_mmcif(sample, model="monomer")      # other arguments: the stock call, no memo
    assert back.confidence_scores == sample.confidence_scores and back.to_pdb(model="multimer") == protein.ProteinMultimer.to_pdb(sample, model="multimer")
    assert OO._writer_format(["atlasfold", "multimer", "--format", "pdb"]) == "pdb" and OO._writer_format(["x", "--format=cif"]) == "cif" and OO._writer_format(["x"]) == "cif"


def test_monomer_runner_is_stock_by_name(installed):
    _RM, R = _rm()
    f = R.FoldingRunner.fold_iter_batch
    assert hasattr(f, "__wrapped_stock__") and "stock_by_name" in f.__qualname__


def test_install_refuses_when_not_innermost(installed):
    ins2 = OO.install("exact", "atlasfold-opt", {})
    assert not ins2.applied and ins2.reason.startswith("not_innermost:")


def test_registry_modes_installers():
    from atlasfold_opt import modes, registry
    from atlasfold_opt.hooks import installers
    assert "output_overlap" in installers() and registry.LEVERS["output_overlap"]["cls"] == "exact"
    assert tuple(registry.LEVERS["output_overlap"]["expected"]) == ("disabled", "runner:monomer")
    for mode in (modes.EXACT, modes.FAST, modes.BIG):                  # exact class, speed >= 0 at every size: every tier's row (R3), before lever_report
        row = modes.MODES[mode]
        assert "output_overlap" in row and row.index("output_overlap") < row.index("lever_report")
    assert "output_overlap" not in modes.PLANNED.get(modes.EXACT, [])


def test_phase_labels_name_the_batch_being_forwarded(installed):
    """The PHASE item label is the batch whose forward runs (3-record stream, one look-ahead in the pipeline): with the lever and with the
    stock generator the labels come in the stock batch order — never the peeked batch, never the last one repeated."""
    from atlasfold_opt import phase_timing as PT
    RM, cls, r, s = _new_runner()
    cls._iter_batch = PT.batch_recorder(cls.__dict__["_iter_batch"]) if not getattr(cls._iter_batch, "_afo_phase", None) else cls._iter_batch   # what phase_timing.install does to the runner class
    seen = []
    fwd = s.model_run
    def model_run(feat, **k):
        seen.append((PT._STATE["item"], "+".join(feat["names"])))            # (the label the PHASE line would carry, the batch really forwarded)
        return fwd(feat, **k)
    r.model_run = model_run
    ins = _inputs(RM)[:3]
    list(cls.fold_iter_batch(r, ins, **KW))                                   # lever on (installed)
    on = list(seen); seen.clear()
    stock = cls.fold_iter_batch.__wrapped_stock__
    list(stock(r, ins, **KW))                                                 # the stock generator (lever off)
    off = list(seen)
    assert on and off and [b for _, b in on] == [b for _, b in off]          # same batches, same order
    assert all(label == batch for label, batch in on), on                    # the label IS the forwarded batch (was: the peeked one)
    assert all(label == batch for label, batch in off), off


def test_each_records_files_exist_before_the_next_forward_returns(installed, monkeypatch, tmp_path):
    """First output on time: a batch handed to the outputs worker is WRITTEN there (the CLI's writer and arguments) the moment its outputs are
    ready — while the caller forwards the next batch — not when the loop reaches its yield after that forward.  Three single-record batches with
    a slow scripted forward: when forward k returns, the files of every record forwarded before it exist."""
    RM, cls, r, s = _new_runner(fwd_sleep=2.5)
    monkeypatch.setattr(OO, "write_opts", lambda argv=None: {"out_dir": str(tmp_path), "format": "cif", "save_confidence_arrays": False, "save_distogram": False,
                                                            "writer": "atlasfold_opt.tests.test_output_overlap_cpu:marker_writer"})
    seen_at_return, forwarded = [], []
    fwd = s.model_run
    def model_run(feat, **k):
        out = fwd(feat, **k)
        deadline = time.time() + 10.0                                          # the worker's write races this thread only by its own compute (ms here); bounded wait for a loaded host
        want = set(forwarded)
        while want and time.time() < deadline and not all((tmp_path / n / "early.cif.txt").exists() for n in want):
            time.sleep(0.05)
        seen_at_return.append({n for n in want if (tmp_path / n / "early.cif.txt").exists()})
        forwarded.extend(feat["names"])
        return out
    r.model_run = model_run
    by_name = {i.name: i for i in _inputs(RM)}
    ins = [by_name["a40"], by_name["c130"], by_name["e260"]]                  # three buckets -> three single-record batches
    batches = list(cls.fold_iter_batch(r, ins, **dict(KW, seeds=[11], max_tokens_per_batch=512)))
    assert len(batches) == 3 and len(seen_at_return) == 3
    assert seen_at_return[0] == set()                                          # nothing forwarded before the first
    assert seen_at_return[1] == set(forwarded[:1]), (seen_at_return, forwarded)      # record 1's files exist when forward 2 returns
    assert seen_at_return[2] == set(forwarded[:2]), (seen_at_return, forwarded)      # records 1-2 when forward 3 returns (record 3 = the run's last batch: the caller's, written by the loop)
    L = installed.facts["ledger"]
    assert int(L.get("early_written")) == 2 and L.get("early_by") == "worker" and L.get("early_failed", None) in (None, 0)
    # written ONCE: the CLI loop's write (loop_write = what its guarded nested writer dispatches to) skips the two worker-written records — their
    # files keep the worker's mtime and the loop receives the writer's recorded return value — and writes the last record (the caller's batch) itself
    import pathlib
    m_before = {n: (tmp_path / n / "early.cif.txt").stat().st_mtime_ns for n in forwarded[:2]}
    assert not (tmp_path / forwarded[2] / "early.cif.txt").exists()
    recs = {}
    for outs in batches:
        for o in outs:
            recs[o.name] = OO.loop_write(pathlib.Path(str(tmp_path)) / o.name, o, "cif", False, False)
    assert {n: (tmp_path / n / "early.cif.txt").stat().st_mtime_ns for n in forwarded[:2]} == m_before          # not re-written by the loop
    assert all(recs[n]["record"] == n and recs[n]["pid"] != os.getpid() for n in forwarded[:2])                  # the worker's return values
    assert (tmp_path / forwarded[2] / "early.cif.txt").exists() and recs[forwarded[2]]["pid"] == os.getpid()     # the last record: written by the loop, once
    assert int(L.get("written_once")) == 2 and int(L.get("loop_written")) == 1 and OO.WRITTEN == {}


def test_stock_writer_is_the_cli_function_itself():
    """The early write uses the CLI's own nested `write_outputs` code object (no free variables) with its own defaults — the same statements
    the CLI loop runs when the batch reaches it, hence the same bytes."""
    _rm()
    import inspect, atlasfold.cli.multimer as M
    fn = OO.stock_writer()
    pristine = [c for c in M.run.__code__.co_consts if hasattr(c, "co_name") and c.co_name == "write_outputs" and "_afo_output_overlap_write" not in c.co_names]
    assert fn.__code__ is (pristine[0] if pristine else OO._GUARD["stock_code"]) and fn.__globals__ is vars(M)   # the stock code object (run's constant, or the one the loop guard saved when it swapped it)
    assert str(inspect.signature(fn)) == "(out_dir, output, format, save_confidence_arrays=False, save_distogram=False)"
    assert OO.write_opts(["multimer", "--input-fasta", "x", "--out-dir", "o", "--format", "pdb"]) == {"out_dir": "o", "format": "pdb", "save_confidence_arrays": False, "save_distogram": False, "writer": "stock"}
    assert OO.write_opts(["multimer", "--input-fasta", "x"]) == {}         # no CLI out-dir: nothing is written early (the loop writes)


def test_cli_loop_writer_is_guarded_to_write_each_record_once(installed, tmp_path, monkeypatch):
    """install() swaps the code-object constant of the CLI's nested `write_outputs` (atlasfold.cli.multimer.run) for a dispatcher into
    loop_write — same parameters / defaults / name, no free variables — so a record the worker wrote is not written again; the original code
    object stays available to stock_writer(); the swap is idempotent."""
    import types, pathlib, atlasfold.cli.multimer as M
    L = installed.facts["ledger"]
    assert L.get("loop_write") == "guarded" and OO.guard_cli_loop() == "guarded"
    codes = [c for c in M.run.__code__.co_consts if isinstance(c, types.CodeType) and c.co_name == "write_outputs"]
    assert len(codes) == 1 and "_afo_output_overlap_write" in codes[0].co_names and not codes[0].co_freevars
    assert getattr(codes[0], "co_qualname", "run.<locals>.write_outputs") == "run.<locals>.write_outputs"
    assert vars(M)["_afo_output_overlap_write"] is OO.loop_write
    assert OO.stock_writer().__code__ is OO._GUARD["stock_code"] and "_afo_output_overlap_write" not in OO.stock_writer().__code__.co_names
    f = types.FunctionType(codes[0], vars(M), "write_outputs", (False, False))            # built the way `run` builds it (MAKE_FUNCTION with the defaults tuple)
    monkeypatch.setattr(OO, "write_opts", lambda argv=None: {"out_dir": str(tmp_path), "format": "cif", "writer": "atlasfold_opt.tests.test_output_overlap_cpu:marker_writer"})
    target = pathlib.Path(str(tmp_path)) / "recA"
    OO.WRITTEN[os.path.abspath(str(target))] = {"record": "recA", "pid": -1}
    out = types.SimpleNamespace(name="recA")
    assert f(target, out, "cif") == {"record": "recA", "pid": -1} and not (target / "early.cif.txt").exists()   # worker-written: skipped, its recorded return value
    assert f(target, out, "cif")["pid"] == os.getpid() and (target / "early.cif.txt").exists()                    # not (or no longer) registered: written by the loop
