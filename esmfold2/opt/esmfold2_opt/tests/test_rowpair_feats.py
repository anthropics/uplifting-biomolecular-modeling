"""The model-input features across the ranks of the row-sharded line (esmfold2_opt.rowpair_feats over opt_core.mem.rowpair.rankdata): the
census words; under a 2-rank gloo group on CPU — rank 0 featurises and rank 1 receives byte-identical (features, chain_infos) while never
calling its own featuriser (one call, and two consecutive calls served in step — the second from rank 0's cache), rank 0's featurisation
error raises the family's `refused: feats_rank0_failed` on BOTH ranks with no rank left waiting (a bounded run), rank 0's GPU out-of-memory
error is re-raised on rank 0 as itself while rank 1 names `feats_rank0_failed: OutOfMemoryError`, a leaf corrupted on rank 1 after the
hand-over raises `refused: feats_ranks_differ` on both, a featuriser returning something other than a (features, chain_infos) pair is rank 0's
failure by name; the per-rank digest census of a process without the builder
binding (equal inputs -> the family's `[feats] … feats_ranks_equal=yes` line on both ranks behind the kit's prefix; one byte flipped on rank 1
-> `=no` on BOTH ranks, nothing raised); n_gpu=1 leaves the builder untouched, the binding is outermost and restorable; no group -> no census;
and the ONE hash seed of the P rank processes of `pred --n_gpu P` through the kit's own launcher (rowpair.launch -> opt_core run_rank_processes,
the rank command replaced by a one-line seed printer): a PYTHONHASHSEED the launching process names reaches every rank as named
(`source=inherited`), and with none named every rank starts with the family's default (`source=default`) — the launcher's `[esmfold2-opt]
RANKENV hashseed=<v> source=… ranks=2` line (the kit's tag) says which."""
import contextlib
import io
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

try:
    from opt_core.mem.rowpair import RowpairRefused, launch as RL, rankdata as RK
except ImportError as _e:                                                   # an older / absent core: named skip (the entry routes refuse producer_missing by name)
    raise unittest.SkipTest(f"producer_missing:opt_core.mem.rowpair.rankdata ({_e}): the feats census tests need the pinned opt_core")
from esmfold2_opt import rowpair, rowpair_feats as RF


def _torch_or_skip():
    try:
        import torch
        import torch.distributed  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"torch (+ torch.distributed) is not importable here ({type(e).__name__}: {e}): test skipped by name")
    return torch


class Words(unittest.TestCase):
    def test_census_words(self):
        self.assertEqual(RF.WHAT, "feats"); self.assertEqual(RF.FORM_PER_RANK, "per_rank"); self.assertIn(RF.FORM_PER_RANK, RK.DATA_FORMS)
        self.assertEqual(RF.WORD_FORM, "data_form")
        self.assertEqual(RF.WORD_EQUAL, "feats_ranks_equal")                                        # the family's census key is the fold line's field
        self.assertEqual(set(RF.fold_fields()), {"data_form", "feats_digest", "feats_ranks_equal", "feats_excluded", "feats_none", "feats_tensor_leaves", "feats_nontensor_leaves", "feats_unhashed"})
        self.assertIn("num_loops", RF.CALL_SETTINGS); self.assertIn("num_sampling_steps", RF.CALL_SETTINGS); self.assertIn("early_exit", RF.CALL_SETTINGS)
        self.assertEqual(RF.FORM_BCAST, "rank0_bcast"); self.assertIn(RF.FORM_BCAST, RK.DATA_FORMS); self.assertEqual(RF.SRC, 0); self.assertEqual(RF.CHAINS_KEY, "chain_infos")

    def test_p1_leaves_the_builder_untouched(self):
        """n_gpu=1: install / install_rank read nothing and bind nothing — the builder's prepare_input is the object it was."""
        class Untouchable:
            def __getattr__(self, name):
                raise AssertionError(f"install(P=1) read model.{name}")

        class Builder:
            def prepare_input(self, input, seed=None, device=None):
                return {"x": input}, []
        b = Builder()
        before = (b.prepare_input, dict(vars(b)))
        for rep in (rowpair.install(Untouchable(), 1, builder=b), rowpair.install_rank(Untouchable(), 1, builder=b)):
            self.assertEqual(rep["sharding"], "none"); self.assertEqual(rep["data_form"], "per_rank"); self.assertEqual(rep["patches"], [])
        self.assertEqual(b.prepare_input, before[0]); self.assertEqual(dict(vars(b)), before[1]); self.assertNotIn("prepare_input", vars(b))
        self.assertEqual(rowpair.PATCHES.names(), [])

    def test_the_binding_is_outermost_and_restorable(self):
        """prepare_input_rank0 wraps what the builder INSTANCE holds (the kit's cache / seed-guard wrappers); the install record restores it."""
        torch = _torch_or_skip()
        RF.reset()

        class Builder:
            def prepare_input(self, input, seed=None, device=None):
                raise AssertionError("the class method is under the instance wrappers")
        b = Builder()
        calls = []

        def cached(input, seed=None, device=None):                                # stands for the kit's feature cache + seed guard on the instance
            calls.append((input, seed, device))
            return {"token_attention_mask": torch.ones(1, 4, dtype=torch.bool)}, ["chains"]
        b.prepare_input = cached
        rowpair.PATCHES.replace(b, "prepare_input", RF.prepare_input_rank0(b.prepare_input))
        try:
            self.assertEqual(RF.STATE["form"], "rank0_bcast"); self.assertEqual(getattr(b.prepare_input, "_esmfold2_opt_data_form", None), "rank0_bcast")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                feats, chains = b.prepare_input("spi", seed=3, device="cpu")            # no group in this process: rank 0's path, the hand-over a passthrough
            self.assertEqual(calls, [("spi", 3, "cpu")]); self.assertEqual(chains, ["chains"]); self.assertEqual(list(feats), ["token_attention_mask"])
            self.assertIn("[esmfold2-opt] rowpair feats: data_form=rank0_bcast src=0 rank=0 call=1 N=4 keys=1 digest=", err.getvalue())
            self.assertIn(" feats_ranks_equal=n/a feats_status=ok ", err.getvalue()); self.assertIn(" feat_s=", err.getvalue())   # no group: nothing to compare (the family's n/a)
            self.assertEqual(RF.fold_fields()["data_form"], "rank0_bcast")
        finally:
            rowpair.PATCHES.restore()
            RF.reset()
        self.assertIs(b.prepare_input, cached)                                    # the instance wrapper it held, not the class method

    def test_no_group_no_census(self):
        """Outside a rank process (no group): the forward-entry census does nothing and prints nothing; the fold fields name the form only."""
        torch = _torch_or_skip()
        RF.reset()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rec = RF.census_per_rank({"token_index": torch.arange(6)[None], "num_loops": 2, "msa": None})
        self.assertIsNone(rec); self.assertEqual(err.getvalue(), "")
        self.assertEqual(RF.fold_fields(), {"data_form": "per_rank", "feats_digest": None, "feats_ranks_equal": None, "feats_excluded": None, "feats_none": None,
                                           "feats_tensor_leaves": None, "feats_nontensor_leaves": None, "feats_unhashed": None})
        self.assertEqual(rowpair.report()["data_form"], "per_rank")

    def test_digest_view_names_what_it_leaves_aside(self):
        """The digest is the family's digest_features over the call minus the fold SETTINGS (excluded by name) and minus optional inputs given
        as None (listed): those do not change it and are returned by name; every other entry enters — a dtype, a shape, one byte, a key's
        presence, a non-setting scalar, a list of tensors — and an entry without a canonical form is named as unhashed, never dropped."""
        torch = _torch_or_skip()
        base = {"token_index": torch.arange(6)[None], "ref_pos": torch.linspace(0, 1, 18).reshape(1, 6, 3)}
        fd, excluded, none = RF.digest_inputs(base)
        d0 = fd.digest
        self.assertEqual((len(d0), excluded, none, fd.words()), (64, (), (), "tensor_leaves=2 nontensor_leaves=0 nontensor_unhashed=none"))
        self.assertEqual(d0, RK.feature_digest(base)); self.assertEqual(RF.feature_digest(base), d0)                # an all-tensor dict: the family's digest itself
        fd2, excluded2, none2 = RF.digest_inputs({**base, "num_loops": 2, "num_sampling_steps": 200, "early_exit": False, "msa": None, "lm_hidden_states": None})
        self.assertEqual(fd2.digest, d0)                                                                            # settings and absent optionals do not enter …
        self.assertEqual(excluded2, ("early_exit", "num_loops", "num_sampling_steps")); self.assertEqual(none2, ("lm_hidden_states", "msa"))   # … and are named
        self.assertTrue(set(excluded2) <= set(RF.CALL_SETTINGS))
        self.assertNotEqual(RF.feature_digest({**base, "ref_pos": base["ref_pos"].double()}), d0)                 # dtype
        self.assertNotEqual(RF.feature_digest({**base, "ref_pos": base["ref_pos"].reshape(1, 3, 6)}), d0)         # shape
        flipped = base["ref_pos"].clone(); flipped.view(torch.uint8)[0, 0, 0] ^= 1
        self.assertNotEqual(RF.feature_digest({**base, "ref_pos": flipped}), d0)                                  # one byte
        self.assertNotEqual(RF.feature_digest({**base, "msa": torch.zeros(1, 1, 6, dtype=torch.long)}), d0)       # a key present vs absent
        self.assertNotEqual(RF.feature_digest({**base, "recycle": 3}), d0)                                        # a non-setting scalar enters (canonical JSON)
        self.assertNotEqual(RF.feature_digest({**base, "templates": [base["ref_pos"]]}), d0)                      # a list of tensors enters
        fd3, _, _ = RF.digest_inputs({**base, "handle": object()})
        self.assertEqual(fd3.unhashed, ("/handle:object",)); self.assertIn("nontensor_unhashed=/handle:object", fd3.words())   # no canonical form: named, not dropped


# ============================================================================================ 2-rank gloo group on CPU (spawned workers)
def _mp(P, entry, *args, run_timeout_s=600):
    return RL.run_sharded(P, entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120, run_timeout_s=run_timeout_s)


def _feats(torch, N=12):
    """A small stand-in for the prepare_input dict (token-level, atom-level, pair-shaped and MSA entries + the scalar settings of a fold call)."""
    return {"token_index": torch.arange(N)[None], "ref_pos": torch.linspace(0, 1, 3 * N * 3).reshape(1, 3 * N, 3),
            "token_bonds": torch.zeros(1, N, N, 1), "token_attention_mask": torch.ones(1, N, dtype=torch.bool),
            "msa": torch.zeros(1, 1, N, dtype=torch.long), "lm_hidden_states": None, "num_loops": 2, "num_sampling_steps": 8}


def _entry_census(flip_rank1: bool):
    """Every rank builds the same call dict (rank 1 flips one byte of ref_pos when asked), takes the forward-entry census with stderr captured,
    and all-gathers (record, captured text) so rank 0 returns every rank's."""
    import torch
    from opt_core.mem.rowpair import dist as RD
    P, rank = RD.world()
    call = _feats(torch)
    if flip_rank1 and rank == 1:
        t = call["ref_pos"].clone(); t.view(torch.uint8)[0, 5, 1] ^= 4; call["ref_pos"] = t
    RF.reset()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rec = RF.census_per_rank(call)
        rec2 = RF.census_per_rank(call)                                          # a second fold: call=2
    fields = RF.fold_fields()
    return RD.comm().allgather_obj({"rank": rank, "P": P, "rec": rec, "rec2_call": rec2["call"], "text": err.getvalue(), "fields": fields})


class CensusUnderGloo(unittest.TestCase):
    def setUp(self):
        _torch_or_skip()

    def test_equal_inputs_agree_on_every_rank(self):
        allv = _mp(2, _entry_census, False)
        self.assertEqual(len(allv), 2)
        d0 = allv[0]["rec"]["digest"]
        for r, v in enumerate(allv):
            rec = v["rec"]
            self.assertEqual((rec["form"], rec["rank"], rec["P"], rec["call"], rec["keys"]), ("per_rank", r, 2, 1, 5))   # the five tensors; None / ints are not digested
            self.assertTrue(rec["equal"]); self.assertEqual(rec["digest"], d0); self.assertEqual(v["rec2_call"], 2)
            lines = [ln for ln in v["text"].splitlines() if ln.startswith("[esmfold2-opt] [feats] ")]
            self.assertEqual(len(lines), 2, v["text"])                                           # ONE family line per census, behind the kit's prefix
            self.assertEqual(lines[0], f"[esmfold2-opt] [feats] rank {r} digest {d0[:16]} feats_ranks_equal=yes ranks=2 digests={d0[:16]},{d0[:16]}")
            self.assertEqual(v["fields"], {"data_form": "per_rank", "feats_digest": d0[:16], "feats_ranks_equal": "yes", "feats_excluded": "num_loops,num_sampling_steps",
                                          "feats_none": "lm_hidden_states", "feats_tensor_leaves": 5, "feats_nontensor_leaves": "none", "feats_unhashed": "none"})

    def test_one_byte_flipped_on_rank1_is_named_on_both_ranks_and_refuses_nothing(self):
        allv = _mp(2, _entry_census, True)
        d = [v["rec"]["digest"] for v in allv]
        self.assertNotEqual(d[0], d[1])
        for r, v in enumerate(allv):
            self.assertFalse(v["rec"]["equal"])
            self.assertIn(f"[esmfold2-opt] [feats] rank {r} digest {d[r][:16]} feats_ranks_equal=no ranks=2 digests={d[0][:16]},{d[1][:16]}", v["text"])
            self.assertEqual(v["fields"]["feats_ranks_equal"], "no")


# ==================================================================== rank 0 featurises, rank 1 receives (2-rank gloo group on CPU, spawned)
class Chain(object):
    """A picklable stand-in for upstream's ChainInfo records (plain attributes, no tensors) — travels with the structure, not as a tensor."""
    def __init__(self, chain_id, n):
        self.chain_id, self.asym_id, self.tokens = chain_id, n, list(range(n))

    def __eq__(self, other):
        return isinstance(other, Chain) and vars(self) == vars(other)


def _oom_class():
    """torch's OutOfMemoryError (what opt_core.oom.is_oom recognises by class); no GPU is touched by constructing one."""
    import torch
    return getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError


def _entry_bcast(case: str):
    """Every rank calls featurise_rank0 with its own `inner` (twice under `two_calls`): rank 0's featurises (or raises / returns something else by
    `case`; its second call returns the tensors it already built — the feature cache's behaviour), rank 1's raises if it is ever called. Each rank
    reports what it got or what was raised; rank 0 returns every rank's report (all-gathered)."""
    import torch
    from opt_core.mem.rowpair import dist as RD
    P, rank = RD.world()
    feats = {k: v for k, v in _feats(torch).items() if torch.is_tensor(v)}
    chains = [Chain("A", 5), Chain("B", 7)]
    inner_calls = []
    built = {}

    def inner_rank0(input, seed=None, device=None):
        inner_calls.append((input, seed, str(device)))
        if case == "rank0_raises":
            raise ValueError("no CCD entry for ligand ZZZ")
        if case == "rank0_ooms":
            raise _oom_class()("CUDA out of memory. Tried to allocate 30.00 GiB")
        if case == "not_a_pair":
            return dict(feats)                                                    # not a (features, chain_infos) pair
        if not built:                                                             # the first call featurises; later calls are cache hits (the same tensors)
            built["pair"] = ({k: v.clone() for k, v in feats.items()}, list(chains))
        return built["pair"]

    def inner_rank1(input, seed=None, device=None):
        raise AssertionError("a rank > 0 featurised")
    RF.reset()
    if case == "corrupt_rank1" and rank == 1:                                      # one leaf altered on rank 1 AFTER the hand-over, before the digest
        split = RF._split

        def corrupting(tree):
            f, c = split(tree)
            f["msa"] = f["msa"] + 1
            return f, c
        RF._split = corrupting
    n_calls = 2 if case == "two_calls" else 1
    err = io.StringIO()
    try:
        digests, recs = [], []
        with contextlib.redirect_stderr(err):
            for _ in range(n_calls):
                got_feats, got_chains = RF.featurise_rank0(inner_rank0 if rank == 0 else inner_rank1, "spi", seed=7, device="cpu")
                digests.append(RF.feature_digest(got_feats)); recs.append(RF.STATE["last"])
        out = {"rank": rank, "raised": None, "digest": digests[-1], "digests": digests, "keys": sorted(got_feats), "chains_equal": got_chains == chains,
               "chains_type": type(got_chains[0]).__name__, "devices": sorted({str(v.device) for v in got_feats.values()}),
               "want_digest": RF.feature_digest(feats), "rec": recs[-1], "calls_seen": [r["call"] for r in recs], "state_calls": RF.STATE["calls"],
               "text": err.getvalue(), "inner_calls": inner_calls}
    except RowpairRefused as e:
        out = {"rank": rank, "raised": str(e), "raised_type": type(e).__name__, "cause": type(e.__cause__).__name__ if e.__cause__ is not None else None,
               "inner_calls": inner_calls, "text": err.getvalue()}
    except Exception as e:                                                        # anything else a rank raised (rank 0's own out-of-memory error): reported by class
        out = {"rank": rank, "raised": str(e), "raised_type": type(e).__name__, "cause": None, "inner_calls": inner_calls, "text": err.getvalue()}
    return RD.comm().allgather_obj(out)


class Rank0FeaturisesUnderGloo(unittest.TestCase):
    def setUp(self):
        _torch_or_skip()

    def test_rank1_receives_rank0s_features_byte_identical_and_never_featurises(self):
        allv = _mp(2, _entry_bcast, "ok")
        self.assertEqual(len(allv), 2)
        r0, r1 = allv
        self.assertIsNone(r0["raised"], r0); self.assertIsNone(r1["raised"], r1)
        self.assertEqual([tuple(c) for c in r0["inner_calls"]], [("spi", 7, "cpu")])
        self.assertEqual(r1["inner_calls"], [])                                                    # ranks > 0 never featurise
        self.assertEqual(r0["digest"], r0["want_digest"]); self.assertEqual(r1["digest"], r0["digest"])   # byte-identical on the receiver
        self.assertEqual(r1["keys"], r0["keys"]); self.assertNotIn("chain_infos", r1["keys"])
        self.assertTrue(r0["chains_equal"]); self.assertTrue(r1["chains_equal"]); self.assertEqual(r1["chains_type"], "Chain")
        self.assertEqual(r1["devices"], ["cpu"])
        for r, v in enumerate(allv):
            rec = v["rec"]
            self.assertEqual((rec["form"], rec["src"], rec["rank"], rec["P"], rec["call"], rec["N"], rec["keys"], rec["equal"], rec["status"]),
                             ("rank0_bcast", 0, r, 2, 1, 12, 5, True, "ok"))
            self.assertEqual(rec["tensors"], 5)                                                    # the five feature tensors travelled (the chain records ride pickled)
            self.assertIn(f"[esmfold2-opt] rowpair feats: data_form=rank0_bcast src=0 rank={r} call=1 N=12 keys=5 digest={r0['digest'][:16]} feats_ranks_equal=yes "
                          f"feats_status=ok tensors=5 gib=", v["text"])
            self.assertIn(" rng=carried ", v["text"]); self.assertEqual(v["text"].count("rowpair feats:"), 1, v["text"])
            self.assertIn(" excluded=none none=none tensor_leaves=5 nontensor_leaves=0 nontensor_unhashed=none", v["text"])
            self.assertEqual((tuple(rec["excluded"]), tuple(rec["none"])), ((), ()))
        self.assertIsNotNone(r0["rec"]["feat_s"]); self.assertIsNone(r1["rec"]["feat_s"])            # feat_s: rank 0's featurisation; receivers print none
        self.assertIn("feat_s=none", r1["text"])

    def test_two_consecutive_calls_are_two_hand_overs_in_step(self):
        """The fold loop's shape: prepare_input twice per (item, seed) — the second a cache hit on rank 0, handed over again. Both ranks print
        call=1 and call=2, the k-th call meets at the k-th store key, the digests are equal both times, rank 1 never featurises."""
        allv = _mp(2, _entry_bcast, "two_calls")
        r0, r1 = allv
        for r, v in enumerate(allv):
            self.assertIsNone(v["raised"], v)
            self.assertEqual(v["calls_seen"], [1, 2]); self.assertEqual(v["state_calls"], 2)
            self.assertEqual(v["digests"], [r0["want_digest"], r0["want_digest"]])
            self.assertEqual(v["text"].count("rowpair feats:"), 2, v["text"])
            self.assertIn(f"rowpair feats: data_form=rank0_bcast src=0 rank={r} call=1 N=12 ", v["text"]); self.assertIn(f"rank={r} call=2 N=12 ", v["text"])
        self.assertEqual(len(r0["inner_calls"]), 2); self.assertEqual(r1["inner_calls"], [])

    def test_rank0_out_of_memory_is_reraised_as_itself_on_rank0_and_named_on_the_others(self):
        """A GPU out-of-memory error in rank 0's featurisation propagates on rank 0 AS ITSELF (never re-worded into a refusal there), after the
        other ranks were told — they raise the family's refusal naming it, and no rank waits."""
        allv = _mp(2, _entry_bcast, "rank0_ooms", run_timeout_s=180)
        r0, r1 = allv
        self.assertEqual(r0["raised_type"], "OutOfMemoryError", r0); self.assertIn("CUDA out of memory", r0["raised"])
        self.assertEqual(r1["raised_type"], "RowpairRefused", r1); self.assertIn("refused: feats_rank0_failed: OutOfMemoryError: CUDA out of memory", r1["raised"])
        self.assertEqual(r1["inner_calls"], [])

    def test_rank0_featurisation_error_reaches_every_rank_and_no_rank_waits(self):
        allv = _mp(2, _entry_bcast, "rank0_raises", run_timeout_s=180)                             # a rank left waiting would end this run by its timeout, not by a result
        for v in allv:
            self.assertIsNotNone(v["raised"], v); self.assertIn("refused: feats_rank0_failed: ValueError: no CCD entry for ligand ZZZ", v["raised"])
        self.assertEqual(allv[0]["cause"], "ValueError"); self.assertIsNone(allv[1]["cause"])       # rank 0's is chained to the original
        self.assertEqual(allv[1]["inner_calls"], [])

    def test_a_leaf_corrupted_on_rank1_refuses_on_every_rank(self):
        allv = _mp(2, _entry_bcast, "corrupt_rank1", run_timeout_s=180)
        for v in allv:
            self.assertIsNotNone(v["raised"], v); self.assertIn("refused: feats_ranks_differ: rank ", v["raised"])

    def test_a_featuriser_returning_no_pair_is_rank0s_failure_by_name(self):
        allv = _mp(2, _entry_bcast, "not_a_pair", run_timeout_s=180)
        for v in allv:
            self.assertIsNotNone(v["raised"], v); self.assertIn("refused: feats_rank0_failed: ValueError: ", v["raised"])   # dict unpacking into (features, chain_infos) failed on rank 0; every rank learned it
        self.assertEqual(allv[1]["inner_calls"], [])


# ================================================================================== one PYTHONHASHSEED for the P rank processes (the launcher's)
SEED_PRINTER = [sys.executable, "-c", "import os; print('seed', os.environ.get('PYTHONHASHSEED'), flush=True)"]


def _launch_seed_printer(P: int, env: dict):
    """rowpair.launch (the kit's launcher side) with opt_core's run_rank_processes WRAPPED, not faked: the rank command becomes a one-line
    printer of the rank's PYTHONHASHSEED, everything else (the rank environment the kit and the core build, the RANKENV line) is the real one.
    Returns ``(seeds, stderr)``: the seed word every rank printed (rank r's transcript is the launcher's rank<r>.log) and the launcher's stderr."""
    real = RL.run_rank_processes
    box = {}

    def ranks_print_their_seed(n_gpu, argv, **kw):
        box["recs"] = real(n_gpu, SEED_PRINTER, **{**kw, "cpu_ok": True, "isolate_devices": False, "log_dir": tempfile.mkdtemp(prefix="feats_seed_")})
        return box["recs"]
    argv = ["--variant", "fast", "--mode", "big", "--n_gpu", str(P), "--input", "x.json", "--out_dir", "o", "--seeds", "1"]
    err = io.StringIO()
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(RL, "run_rank_processes", ranks_print_their_seed), \
            mock.patch.object(rowpair, "refuse_unless_visible", lambda P, visible=None: P), contextlib.redirect_stderr(err):
        rc = rowpair.launch(argv, "big", P, stream=io.StringIO())
    assert rc == 0, box
    seeds = []
    for rec in box["recs"]:
        with open(rec["log"], encoding="utf-8", errors="replace") as fh:
            m = re.search(r"^seed (\S+)$", fh.read(), re.M)
        seeds.append(m.group(1) if m else None)
    return seeds, err.getvalue()


def _env_without_rank_words() -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith("ROWPAIR_") and k not in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "PYTHONHASHSEED")}


class OneHashSeedForAllRanks(unittest.TestCase):
    def test_an_inherited_hash_seed_reaches_every_rank(self):
        """PYTHONHASHSEED named by the launching process (a user's export, the image's ENV) is every rank's: source=inherited."""
        seeds, err = _launch_seed_printer(2, {**_env_without_rank_words(), "PYTHONHASHSEED": "123"})
        self.assertEqual(seeds, ["123", "123"])
        self.assertIn("[esmfold2-opt] RANKENV hashseed=123 source=inherited ranks=2", err)           # the launch's line carries the kit's tag (ROWPAIR_TAG handed to the core launcher)

    def test_ranks_get_one_hash_seed_when_none_is_named(self):
        """No PYTHONHASHSEED in the launching process: every rank starts with the family's default seed (source=default) — never a per-process random one."""
        seeds, err = _launch_seed_printer(2, _env_without_rank_words())
        self.assertEqual(seeds, [RK.HASHSEED_DEFAULT, RK.HASHSEED_DEFAULT]); self.assertEqual(RK.HASHSEED_DEFAULT, "0")
        self.assertIn(f"[esmfold2-opt] RANKENV hashseed={RK.HASHSEED_DEFAULT} source=default ranks=2", err)


if __name__ == "__main__":
    unittest.main()
