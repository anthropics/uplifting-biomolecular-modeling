#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Derived from ProteinMPNN (Copyright (c) 2022 Justas Dauparas, MIT licence) and from the persistent worker of
# ../mpnn_pdb_parser (same project, MIT; see NOTICE).
"""mpnn_worker2.py — persistent SolubleMPNN/ProteinMPNN sampler with EXACT RNG replay of the stock
protein_mpnn_run.py invocation (dauparas/ProteinMPNN @ 8907e66), batched across backbones.

Stock invocation reproduced (protein_mpnn_run.py's design pass):
  protein_mpnn_run.py --jsonl_path parsed.jsonl --chain_id_jsonl assigned.jsonl --num_seq_per_target N --batch_size B --sampling_temp "T1 [T2 ...]"
     --seed S --backbone_noise 0.0 [--save_score 1 --save_probs 1] [--omit_AAs --fixed_positions_jsonl --omit_AA_jsonl --bias_AA_jsonl --bias_by_res_jsonl
     --pssm_jsonl --pssm_multi --pssm_threshold --pssm_log_odds_flag --pssm_bias_flag, read as stock reads them] [--path_to_model_weights soluble_model_weights --model_name v_48_020]
  S = the seed given, or for --seed 0 (upstream's default) a seed drawn as protein_mpnn_run.py draws it (np.random.randint(0, high=999)); the
  rounds are stock's `for temp in temperatures: for j in range(N // B)`, B sequences each, written in stock's order and numbering.
RNG model of the stock run (per protein, in dataset order, ONE process): randn(B,L) [randn_1] -> forward (no RNG) -> per round:
  randn(B,L) [randn_2] -> L x torch.multinomial(probs(B,21) float64, 1) -> re-scoring forward (no RNG). `--mode stream` replays this sequential
  stream (each backbone's generator = (seed, offset_b) with offset_b from the per-call Philox increments, or measured by a probe generator that performs the same calls).
Every random draw is made with a per-backbone torch.cuda.Generator on a tensor of the stock shape/dtype, so the values are
the stock values by construction, sequences AND per-residue log-probs per draw.

EXACT levers of the decode loop (sample_v2; every one keeps the stock arithmetic per element):
  --cache_enc_ctx  : the decode step's mask_fw_t / mask_bw_t read by direct indexing of the precomputed [N, L, K, 1] masks (data movement).
                     The encoder context of the decoded position (stock: h_EXV_encoder_fw[:, t]) is assembled per step from h_V / h_E with
                     stock's own gathers and elementwise product for that position; the [N, L, K, 3H] tensors stock materialises for every
                     position at once are never built (memory: at 128 rows x L 500 they were 12.5 GB per process).
  --graph_rng      : capture the WHOLE decode step (decoder + softmax + per-backbone multinomial on registered per-backbone
                     Philox generators + commit) in one CUDA graph per (shape, draw-pattern) and replay it Lmax times with a
                     device-resident step counter (no host work per step). RNG offsets advance exactly as eager.
"""
import contextlib
try:                                                     # --fused_draw: the per-backbone draws of a decode step in ONE kernel (Triton ships with the torch wheel)
    import triton, triton.language as tl
    from triton.language.extra import libdevice as _tl_libdevice
    HAVE_TRITON = True
except ImportError:                                      # no Triton: the lever cannot run (named at start-up)
    HAVE_TRITON = False

def _pow2(n):
    """the smallest power of two >= n (Triton tiles are power-of-two shaped; rows beyond B are masked)."""
    p = 1
    while p < n: p *= 2
    return p

if HAVE_TRITON:
    _PH_M0 = tl.constexpr(0xD2511F53); _PH_M1 = tl.constexpr(0xCD9E8D57); _PH_W0 = tl.constexpr(0x9E3779B9); _PH_W1 = tl.constexpr(0xBB67AE85)

    @triton.jit
    def _mulhilo32(a, b):
        p = a.to(tl.uint64) * tl.full(a.shape, b, tl.uint64)
        return (p >> 32).to(tl.uint32), (p & 0xFFFFFFFF).to(tl.uint32)

    @triton.jit
    def _fused_draw_kernel(probs_ptr, out_ptr, seed_ptr, off_ptr, ndraw_ptr, active_ptr, B: tl.constexpr, BP: tl.constexpr, A: tl.constexpr, AP: tl.constexpr, INC: tl.constexpr, F64: tl.constexpr):
        """Program b = backbone b of the decode step: S_t[b*B + r] = torch.multinomial(probs[b*B:(b+1)*B], 1, generator=g_b)[r] for every clone row r,
        computed as torch computes it (aten multinomial, n_sample 1: q = empty_like(p).exponential_(1, g_b); argmax(p / q, -1)). q's element (r, a)
        comes from Philox4x32-10 subsequence r*A + a at the generator's (seed, offset) — aten's distribution kernel runs one thread per element at
        these sizes —: in float32 the first curand_uniform4 value through -__logf(u), in float64 the first curand_uniform2_double value through
        -log(u), each with torch's 1 - eps/2 clamp; the quotient is IEEE division in that dtype; argmax takes the leftmost maximum. The generator's
        offset is off[b] + ndraw[b] * INC (INC = the Philox increment one such call consumes; offsets are multiples of 4) and ndraw[b] counts this
        backbone's draws in the batch (active steps only). Inactive backbones (stock skips their all-masked step) are left as given. The tile is
        [BP, AP] (BP, AP: B and A rounded up to powers of two, rows >= B and columns >= A masked) so any --batch_size is served. Bit-identity
        with torch's own draw is PROBED at start-up on the device, in the job's dtype (Worker.decide_fused_draw)."""
        b = tl.program_id(0)
        act = tl.load(active_ptr + b)
        seed = tl.load(seed_ptr + b); off = tl.load(off_ptr + b) + tl.load(ndraw_ptr + b) * INC
        rows = tl.arange(0, BP)[:, None]; cols = tl.arange(0, AP)[None, :]
        valid = (cols < A) & (rows < B)
        sub = (rows * A + cols).to(tl.uint32)                                  # curand subsequence = the element's index in the [B, A] tensor
        q4 = off // 4
        c0 = tl.full([BP, AP], 0, tl.uint32) + (q4 & 0xFFFFFFFF).to(tl.uint32); c1 = tl.full([BP, AP], 0, tl.uint32) + ((q4 >> 32) & 0xFFFFFFFF).to(tl.uint32)
        c2 = sub; c3 = tl.zeros([BP, AP], tl.uint32)
        k0 = tl.full([BP, AP], 0, tl.uint32) + (seed & 0xFFFFFFFF).to(tl.uint32); k1 = tl.full([BP, AP], 0, tl.uint32) + ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
        for _ in range(10):                                                    # Philox4x32-10 (curand's rounds and key schedule)
            hi0, lo0 = _mulhilo32(c0, _PH_M0)
            hi1, lo1 = _mulhilo32(c2, _PH_M1)
            n0 = hi1 ^ c1 ^ k0; n1 = lo1; n2 = hi0 ^ c3 ^ k1; n3 = lo0
            c0, c1, c2, c3 = n0, n1, n2, n3
            k0 = k0 + tl.full([BP, AP], _PH_W0, tl.uint32); k1 = k1 + tl.full([BP, AP], _PH_W1, tl.uint32)
        if F64:
            zz = c0.to(tl.uint64) ^ (c1.to(tl.uint64) << 21)                    # curand's _curand_uniform_double_hq(x, y): (0, 1]
            u = zz.to(tl.float64) * 1.1102230246251565e-16 + (1.1102230246251565e-16 / 2.0)
            eps64: tl.constexpr = 2.220446049250313e-16
            lg = tl.where(u >= 1.0 - eps64 / 2, -eps64 / 2, _tl_libdevice.log(u))        # aten's exponential transformation, double: log
            q = -1.0 * lg
        else:
            u32 = c0.to(tl.float32) * 2.3283064365386963e-10 + (2.3283064365386963e-10 / 2.0)   # curand's _curand_uniform: (0, 1]
            eps32: tl.constexpr = 1.1920928955078125e-07
            lg32 = tl.where(u32 >= 1.0 - eps32 / 2, -eps32 / 2, _tl_libdevice.fast_logf(u32))   # aten's exponential transformation, float: __logf
            q = -1.0 * lg32
        pr = tl.load(probs_ptr + (b * B + rows) * A + cols, mask=valid, other=0.0)
        w = tl.where(valid, pr / q, -1.0)
        idx = tl.argmax(w, axis=1)
        tl.store(out_ptr + b * B + tl.arange(0, BP), idx.to(tl.int64), mask=(tl.arange(0, BP) < B) & (act != 0))
        tl.store(ndraw_ptr + b, tl.load(ndraw_ptr + b) + (act != 0).to(tl.int64))
import sys, os, json, time, copy, argparse, hashlib, subprocess, re
import numpy as np, torch, torch.nn.functional as F
MPNN_DIR = os.environ.get("MPNN_DIR", "/opt/ProteinMPNN")
sys.path.insert(0, MPNN_DIR)
from protein_mpnn_utils import tied_featurize, ProteinMPNN, StructureDatasetPDB, _scores, _S_to_seq, gather_nodes, cat_neighbors_nodes

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
BASE_CPU_STATE = None

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def load_model(weights, device):
    ck = torch.load(weights, map_location=device)
    load_model.last_sha = hashlib.sha256(open(weights, "rb").read()).hexdigest()
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                        num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0, k_neighbors=ck["num_edges"])
    model.to(device); model.load_state_dict(ck["model_state_dict"]); model.eval()
    return model, ck

# the stock CLI's jsonl dictionaries, read in main() as protein_mpnn_run.py reads them and handed to stock's tied_featurize (which builds chain_M_pos,
# omit_AA_mask, pssm_coef / pssm_bias / pssm_log_odds_all and bias_by_res_all from them exactly as in the stock run)
FIXED_POSITIONS_DICT = None   # --fixed_positions_jsonl: chain_M_pos = 0 at fixed positions
OMIT_AA_DICT = None           # --omit_AA_jsonl: per-position amino acids struck from the draw
BIAS_BY_RES_DICT = None       # --bias_by_res_jsonl: per-position logit bias
PSSM_DICT = None              # --pssm_jsonl: pssm_coef / pssm_bias / pssm_log_odds per chain

def load_jsonl_dict(path, merge=False, option=None):
    """A jsonl dictionary as protein_mpnn_run.py loads it: None unless the path is an existing file; the LAST line's object (each line json.loads-ed in
    turn), or with ``merge`` every line's object merged into one dict (its --pssm_jsonl rule). A path given (non-empty) that is no file prints upstream's
    own note, ``<option> is NOT loaded``, with the path — the pass runs without that dictionary, exactly as protein_mpnn_run.py runs it."""
    if not (path and os.path.isfile(path)):
        if path:
            print("inputs: %s is NOT loaded (%s is not a file; protein_mpnn_run.py's rule: the pass runs without it)" % (option or "jsonl", path), flush=True)
        return None
    with open(path) as fh:
        json_list = list(fh)
    if merge:
        d = {}
        for json_str in json_list: d.update(json.loads(json_str))
        return d
    d = None
    for json_str in json_list: d = json.loads(json_str)
    return d

def native_header_due(T, j, T0):
    """protein_mpnn_run.py writes the native record ('>name, score=…') inside its round loop under `if b_ix == 0 and j==0 and temp==temperatures[0]`:
    once per round whose batch index is 0 and whose temperature equals the FIRST temperature's value — so once for '0.1 0.2', twice for '0.1 0.1'."""
    return j == 0 and T == T0

def featurize_one(protein, device, chain_id_dict):
    """stock tied_featurize on a batch of ONE clone (identical values to the B-clone call, row-repeated later)."""
    return tied_featurize([protein], device, chain_id_dict, FIXED_POSITIONS_DICT, OMIT_AA_DICT, None, PSSM_DICT, BIAS_BY_RES_DICT, ca_only=False)

def make_gen(device, seed, offset=None, B=8):
    """GPU: Philox generator positioned at `offset` (int). CPU (local tests only): `offset` is the list of (L, [n_draw per round])
    of the preceding backbones in dataset order; the stream is replayed by re-consuming those draws."""
    g = torch.Generator(device=device); g.manual_seed(seed)
    if device.type != "cuda" and BASE_CPU_STATE is not None:
        g.set_state(BASE_CPU_STATE)   # stock on CPU: model init consumed the default generator before the first randn
    if offset is None:
        return g
    if isinstance(offset, int):
        g.set_offset(offset)
    else:
        p = torch.full((B, 21), 1.0/21, device=device, dtype=torch.float32)
        for (L, n_draws) in offset:
            torch.randn((B, L), device=device, generator=g)                     # randn_1
            for n_draw in n_draws:                                              # each round: randn_2, then its multinomials
                torch.randn((B, L), device=device, generator=g)
                for _ in range(n_draw): torch.multinomial(p, 1, generator=g)
    return g

def probe_consumption(device, L, n_draws, B=8):
    """measure the generator offset consumed by one stock protein: randn(B,L) [randn_1] + per round randn(B,L) [randn_2] + n_draw x
    multinomial((B,21) float32, 1). n_draws = per round, the number of decode steps with mask != 0 (stock skips the draw on all-masked steps)."""
    g = make_gen(device, 0)
    torch.randn((B, L), device=device, generator=g)
    p = torch.full((B, 21), 1.0/21, device=device, dtype=torch.float32)
    for n_draw in n_draws:
        torch.randn((B, L), device=device, generator=g)
        for _ in range(n_draw):
            torch.multinomial(p, 1, generator=g)
    return int(g.get_offset())

def fmt4(x):
    return np.format_float_positional(np.float32(x), unique=False, precision=4)


# --hybrid_gemm GEMM groups by compute capability. The batched message GEMM of a decode batch runs over groups of backbones whose row count
# g*B stays at or below the card's ceiling, so that cuBLAS serves the group from the kernel family it uses for the stock per-backbone shape
# (M = B*Kn rows) and the two accumulate identically. Measured with torch 2.5.1+cu124 (cuBLAS 12.4), fp32, Kn = 48, B = 8: on sm_80
# (A100-SXM4-80GB) groups of up to 112 rows (M <= 5376: ampere_sgemm_32x32 / 64x32_sliced1x4) are bit-identical to the stock shape and
# 120 rows and more (M >= 5760: ampere_sgemm_32x128) differ by ~1e-6, so a 16-backbone batch runs as two groups of 8; on sm_90 (H100, H200)
# every group up to the 16-backbone batch (M = 6144) is bit-identical, so no ceiling applies. A card absent from the table has no ceiling.
# The probe (Worker.decide_hybrid) still judges every (B, g, Kn) cell a job uses before any output: the ceiling only chooses g so that the
# probe can pass on that card; an entry that is wrong for a stack makes the probe FAIL (the job is refused by name, --hybrid_gemm 0 runs without the lever), never costs exactness.
HYBRID_GROUP_ROWS = {(8, 0): 112}


def plan_batches(order, lengths, bb_batch, kmin):
    """The job's batches of backbone indices, in processing order: consecutive runs of at most ``bb_batch`` indices of ``order`` — exactly
    ``order[i:i+bb_batch]`` for every i when no backbone is shorter than ``kmin`` — except that a backbone shorter than ``kmin`` (= k_neighbors:
    the batched featuriser's pad-free neighbour sets need L >= k) is a batch of its own, where it runs the stock shapes (k = L, as
    protein_mpnn_run.py runs it). A backbone's outputs never depend on the batch it runs in; only speed does."""
    if bb_batch <= 1 or all(lengths[i] >= kmin for i in order):
        return [list(order[i:i + bb_batch]) for i in range(0, len(order), bb_batch)]
    batches, cur = [], []
    for i in order:
        if lengths[i] < kmin:
            if cur: batches.append(cur); cur = []
            batches.append([i])
        else:
            cur.append(i)
            if len(cur) == bb_batch: batches.append(cur); cur = []
    if cur: batches.append(cur)
    return batches

def hybrid_groups(K, B, ceiling):
    """The backbone-group sizes one K-backbone decode batch is served in by the batched message GEMM: [K] when the card has no row ceiling
    or K*B rows fit under it, else K split into the fewest near-equal groups of at most max(1, ceiling // B) backbones (largest first)."""
    if not ceiling or K * B <= ceiling:
        return [K]
    n = -(-K // max(1, ceiling // B))
    base, extra = divmod(K, n)
    return [base + 1] * extra + [base] * (n - extra)

class Worker:
    def __init__(self, model, device, temperature, seed, chunk_gemm=False, cache_enc_ctx=False, graph_rng=False):
        self.model, self.device, self.T, self.seed = model, device, temperature, seed
        self.rounds = [(temperature, 0)]            # stock's `for temp in temperatures: for j in range(NUM_BATCHES)`: one (temperature, batch index) per round of B sequences per backbone (main() sets the job's)
        self.pssm_multi, self.pssm_threshold, self.pssm_log_odds_flag, self.pssm_bias_flag = 0.0, 0.0, False, False   # upstream's --pssm_* as given (main() sets them)
        self.cache_enc_ctx, self.graph_rng = cache_enc_ctx, graph_rng
        self.save_score = self.save_probs = False   # upstream's --save_score / --save_probs write gates (main() sets them from the flags): the arrays are computed either way, as in stock; only the files are gated
        self.graph_stats = {"captures": 0, "capture_s": 0.0, "replays": 0}
        self.chunk_gemm = chunk_gemm   # run the per-step decoder GEMMs with the STOCK row count (B rows x K neighbours per backbone) so cuBLAS picks the stock kernel -> bitwise probs
        self.omit_AAs_np = np.array([AA in getattr(Worker, "OMIT_AAS", "X") for AA in ALPHABET]).astype(np.float32)
        self.bias_AAs_np = np.array(getattr(Worker, "BIAS_AAS_NP", np.zeros(len(ALPHABET))))   # upstream's bias_AAs_np: zeros, or the --bias_AA_jsonl values (float64, as stock builds it)
        self.constant = torch.tensor(self.omit_AAs_np, device=device)
        self.constant_bias = torch.tensor(self.bias_AAs_np, device=device)
        self.stream_offset = 0
        self.consumption_cache = {}
        try:
            self.commit = "unknown" if re.search(r"""[\s'"$`\\;&|<>()]""", MPNN_DIR) else subprocess.check_output(["git", "--git-dir", f"{MPNN_DIR}/.git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()   # upstream's `git --git-dir <clone>/.git rev-parse HEAD` as an argument list, no shell; a clone path a shell would split or expand is "unknown" with nothing run — what upstream's own shell string gives for it, so the .fa header stays stock's byte for byte
        except Exception:
            self.commit = "unknown"
        self.weights_sha = getattr(load_model, "last_sha", "unknown")
        self.timers = {"featurize": 0.0, "encoder_fwd": 0.0, "sample": 0.0, "rescore": 0.0, "d2h_write": 0.0, "rng_setup": 0.0}
        # batch pipeline (CUDA): upstream's per-backbone scoring forwards of the neighbouring batches run on their own streams underneath a batch's
        # decode-step replays (the decode stream has priority); one Philox generator object per batch slot, re-seeded per backbone (the graphs a
        # decode shape captured keep pointing at them); the decode step graphs and their static tensors are kept per shape and replayed for every
        # batch of that shape (self._slot). What runs, on which tensors of which values, is unchanged — only when.
        self._gens = {}
        self._fwd_slots, self._fwd_pools = {}, {}
        self.fused_draw = False; self.draw_inc = None                  # --fused_draw (main() sets it; decide_fused_draw probes it and measures the Philox increment of one draw)
        self.feats = {}                                                # backbone name -> stock's tied_featurize outputs, computed once (the offsets pass) and taken by prep
        # decode lanes: DECODE_LANES batches' decode loops in flight at once, each on its own stream with its own decode slot (graphs + tensors)
        # and generator bank — one latency-bound chain of small step kernels leaves most of the device idle; two interleave
        self.lanes = self.DECODE_LANES if device.type == "cuda" else 1
        self._slots = [None] * self.lanes; self._lane = 0
        if device.type == "cuda":
            lo, hi = torch.cuda.Stream.priority_range()
            self.sN, self.sR = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)          # native-score forwards | re-scoring forwards
            self.sD = [torch.cuda.Stream(device=device, priority=hi) for _ in range(self.lanes)]          # the decode loops (highest priority: their step kernels are small)
        else:
            self.sN = self.sR = None; self.sD = [None]

    DECODE_LANES = 3        # decode loops in flight per process (CUDA): each on its own stream, slot and generator bank
    DECODE_CHUNK = 16       # decode steps enqueued per turn of a lane before the host serves the other lane (the loops interleave on the device)

    def gen_for(self, lane, b, offset):
        """The Philox generator of (lane, batch slot b) positioned at this backbone's own offset of the stock stream (make_gen's state). On CUDA the
        generator OBJECT persists across batches — the lane's decode-step graphs registered it — and is re-seeded here; on CPU a fresh generator (make_gen)."""
        if self.device.type != "cuda":
            return make_gen(self.device, self.seed, offset)
        g = self._gens.get((lane, b))
        if g is None:
            g = self._gens[(lane, b)] = torch.Generator(device=self.device)
        g.manual_seed(self.seed); g.set_offset(offset)
        return g

    def _enter(self, c):
        """the batch whose stage runs next on the host: its backbone lengths and its encoder outputs (batches interleave stage by stage; one thread)."""
        self._cur_Ls = list(c["Ls"]); self._batch_serial = c["serial"]
        self._enc_cache_key, self._enc_cache_val = c.get("enc_key"), c.get("enc")

    def _stream(self, st):
        return torch.cuda.stream(st) if st is not None else contextlib.nullcontext()

    def _event(self, st):
        """A timing event recorded on stream st (None on CPU)."""
        if st is None: return None
        ev = torch.cuda.Event(enable_timing=True); ev.record(st); return ev

    def n_draw_stock(self, f, B, p=None, gen=None):
        """number of multinomial calls the stock sample() makes for this protein = number of steps t_ where NOT all B rows sit on a
        masked (missing-coordinate) position. Needs the B decoding orders, i.e. randn_2 of THIS protein drawn at its stream offset
        (gen positioned after randn_1). For backbones without missing residues this is simply L (no RNG needed)."""
        X, mask, chain_M, chain_M_pos = f[0], f[2], f[4], f[10]
        L = int(X.shape[1])
        if bool((mask[0] != 0).all()): return L
        assert gen is not None, "n_draw_stock: gapped backbone needs the positioned generator"
        randn_2 = torch.randn((B, L), device=X.device, generator=gen)
        chain_mask = (chain_M*chain_M_pos*mask)[0:1].expand(B, L)
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn_2)))
        mg = torch.gather(mask[0:1].expand(B, L), 1, decoding_order)
        return int((~(mg == 0).all(dim=0)).sum().item())

    def inc_mult(self, B=8):
        """CUDA: the Philox offset increment of ONE torch.multinomial((B, 21) float32, 1) call (CUDAGeneratorImpl::philox_cuda_state(increment)), measured once per B."""
        if not hasattr(self, "_inc_mult"): self._inc_mult, self._inc_randn = {}, {}
        if B not in self._inc_mult:
            g = make_gen(self.device, 0); p = torch.full((B, 21), 1.0/21, device=self.device, dtype=torch.float32)
            torch.multinomial(p, 1, generator=g); self._inc_mult[B] = int(g.get_offset())
        return self._inc_mult[B]

    def inc_randn(self, L, B=8):
        """CUDA: the Philox offset increment of ONE torch.randn((B, L)) call, measured once per (B, L)."""
        self.inc_mult(B)
        if (B, L) not in self._inc_randn:
            g = make_gen(self.device, 0); torch.randn((B, L), device=self.device, generator=g); self._inc_randn[(B, L)] = int(g.get_offset())
        return self._inc_randn[(B, L)]

    def n_draws_stock(self, f, B, gen, n_rounds):
        """per round, the number of multinomial calls the stock sample() makes for a GAPPED protein (n_draw_stock): every round draws its own randn_2 —
        its own B decoding orders — from the stream where the previous round's draws left it. gen: positioned right after randn_1 of this protein; it is
        advanced here past each round's randn_2 and its multinomials, exactly the calls stock makes."""
        out = []
        for _ in range(n_rounds):
            n_draw = self.n_draw_stock(f, B, gen=gen)                             # draws this round's randn_2 from gen
            out.append(n_draw)
            if self.device.type == "cuda":
                gen.set_offset(int(gen.get_offset()) + n_draw * self.inc_mult(B))
            else:
                p = torch.full((B, 21), 1.0/21, device=self.device, dtype=torch.float32)
                for _ in range(n_draw): torch.multinomial(p, 1, generator=gen)
        return out

    def consumption(self, L, n_draws, B=8):
        """the generator offset one stock protein consumes: randn_1 + per round (randn_2 + its multinomials); n_draws = the per-round draw counts."""
        key = (B, L, tuple(n_draws))
        if key not in self.consumption_cache:
            if getattr(self, "analytic_offsets", False) and self.device.type == "cuda":
                # Philox offsets advance by a per-call constant: the increments of ONE randn((B,L)) and ONE multinomial((B,21),1) call, counted, instead of
                # replaying every draw
                self.consumption_cache[key] = (1 + len(n_draws)) * self.inc_randn(L, B) + sum(n_draws) * self.inc_mult(B)
            else:
                self.consumption_cache[key] = probe_consumption(self.device, L, list(n_draws), B)
        return self.consumption_cache[key]

    # ---------- one batch of K backbones ----------
    # ---------- one batch of K backbones: the stages (upstream's statements, unchanged) and their order ----------
    def prep(self, proteins, chain_id_dict, B=8, lane=0):
        """featurize (stock's tied_featurize per backbone, padded to the batch), the per-backbone generators at their stock-stream offsets and the
        native-score forward's randn — on the native-score stream. Returns the batch context every later stage reads."""
        dev = self.device; K = len(proteins); N = K*B
        self._serial = getattr(self, "_serial", 0) + 1
        c = {"proteins": proteins, "K": K, "N": N, "B": B, "t_start": time.time(), "serial": self._serial, "lane": lane}
        with self._stream(self.sN):
            t0 = time.time()
            feats = [self.feats.pop(p["name"]) if p["name"] in self.feats else featurize_one(p, dev, chain_id_dict) for p in proteins]   # stock's tied_featurize per backbone (once: the offsets pass kept it)
            Ls = [int(f[0].shape[1]) for f in feats]
            Lmax = max(Ls)
            def pad_rep(x, fill=0.0):
                # x: [1, L, ...] -> [B, Lmax, ...] (stock: B identical rows; pad positions: mask 0)
                if x.dim() == 1: return x
                padn = Lmax - x.shape[1]
                if padn:
                    shape = list(x.shape); shape[1] = padn
                    x = torch.cat([x, torch.full(shape, fill, dtype=x.dtype, device=x.device)], 1)
                return x.expand(B, *x.shape[1:]).contiguous()
            cols = {}
            names = ["X","S","mask","lengths","chain_M","chain_encoding_all","chain_list_list","visible_list_list","masked_list_list",
                     "masked_chain_length_list_list","chain_M_pos","omit_AA_mask","residue_idx","dihedral_mask","tied_pos_list_of_lists_list",
                     "pssm_coef","pssm_bias","pssm_log_odds_all","bias_by_res_all","tied_beta"]
            tens = ["X","S","mask","chain_M","chain_encoding_all","chain_M_pos","omit_AA_mask","residue_idx","pssm_coef","pssm_bias","pssm_log_odds_all","bias_by_res_all"]
            for nm in tens:
                i = names.index(nm)
                fill = -100 if nm == "residue_idx" else (10000.0 if nm == "pssm_log_odds_all" else 0.0)
                cols[nm] = torch.cat([pad_rep(f[i], fill) for f in feats], 0)
            self.timers["featurize"] += time.time()-t0
            t0 = time.time()
            gens = [self.gen_for(c["lane"], b, self.offsets[proteins[b]["name"]]) for b in range(K)]   # the stock stream replayed: each backbone's own offset
            randn_1 = torch.zeros((N, Lmax), device=dev)
            for b, L in enumerate(Ls):
                randn_1[b*B:(b+1)*B, :L] = torch.randn((B, L), device=dev, generator=gens[b])
            self.timers["rng_setup"] += time.time()-t0
        c.update({"feats": feats, "Ls": Ls, "Lmax": Lmax, "cols": cols, "gens": gens, "randn_1": randn_1})
        return c

    def native(self, c):
        """the native scoring forward (stock: model(X, S, ..., randn_1)) on the native-score stream; the scores stay on the device until finalize."""
        cols, Ls, B, K = c["cols"], c["Ls"], c["B"], c["K"]
        X, S, mask, chain_M, chain_encoding_all, chain_M_pos, residue_idx = (cols[k] for k in ["X","S","mask","chain_M","chain_encoding_all","chain_M_pos","residue_idx"])
        with self._stream(self.sN):
            e0 = self._event(self.sN); t0 = time.time()
            self._enter(c)
            with torch.no_grad():
                self.encode(X, mask, residue_idx, chain_encoding_all, B, K)          # the batch's stock-shape encoder outputs: once, for its three readers
                c["enc_key"], c["enc"] = self._enc_cache_key, self._enc_cache_val
                ev_enc = self._event(self.sN)
                log_probs_nat = self.forward_scores(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, c["randn_1"], B, K)
            mask_for_loss = mask*chain_M*chain_M_pos
            if getattr(self, "stock_shape_enc", False):
                native_score = [(_scores(S[b*B:(b+1)*B, :L], log_probs_nat.float()[b*B:(b+1)*B, :L], mask_for_loss[b*B:(b+1)*B, :L]),
                                 _scores(S[b*B:(b+1)*B, :L], log_probs_nat.float()[b*B:(b+1)*B, :L], mask[b*B:(b+1)*B, :L])) for b, L in enumerate(Ls)]
            else:
                native_score = [(_scores(S, log_probs_nat.float(), mask_for_loss), _scores(S, log_probs_nat.float(), mask))]
            c.update({"mask_for_loss": mask_for_loss, "native_score_dev": native_score, "ev_enc": ev_enc, "ev_native": (e0, self._event(self.sN)), "t_native_host": time.time()-t0})
        return c

    def decode(self, c, keep=False):
        """sampling (stock's sample()) of the batch's next round — stock's `for temp in temperatures: for j in range(NUM_BATCHES)`, self.rounds — on the
        batch's decode lane: sets the stepper up — advance() then enqueues the decode DECODE_CHUNK steps a turn (the first turn waits, on the lane's stream,
        for the batch's inputs and encoder outputs, draws stock's randn_2 of the round and builds or refills the lane's decode slot). With keep the sampled
        S / probs are copied out of the slot's tensors at the last turn (the next round, or a later batch of the shape, reuses them)."""
        c.setdefault("rounds_out", [])
        c["T"], c["j"] = self.rounds[len(c["rounds_out"])]                     # this round's temperature and batch index (stock's sample numbering: j*B + b_ix + 1)
        c["keep"] = keep; c["gen"] = self._decode_steps(c); c["done"] = False
        return c

    def more_rounds(self, c):
        """True while the batch owes rounds (its re-scored rounds, c["rounds_out"], are fewer than the job's)."""
        return len(c.get("rounds_out", [])) < len(self.rounds)

    def _decode_steps(self, c):
        cols, Ls, B, K, N, Lmax, gens = c["cols"], c["Ls"], c["B"], c["K"], c["N"], c["Lmax"], c["gens"]
        dev = self.device; sD = self.sD[c["lane"]]
        X, S, mask, chain_M, chain_encoding_all, chain_M_pos, residue_idx = (cols[k] for k in ["X","S","mask","chain_M","chain_encoding_all","chain_M_pos","residue_idx"])
        if sD is not None: sD.wait_event(c["ev_enc"])                          # its own inputs and encoder outputs; not the native-score forwards queued behind them
        c["ev_sample0"] = self._event(sD); c["t_sample0"] = time.time()
        self.release_step_graphs((K, B, Lmax), lane=c["lane"])      # another decode shape than the lane's slot: its graphs and tensors go first
        randn_2 = torch.zeros((N, Lmax), device=dev)
        for b, L in enumerate(Ls):
            randn_2[b*B:(b+1)*B, :L] = torch.randn((B, L), device=dev, generator=gens[b])
        c["randn_2"] = randn_2
        sd = yield from self.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask, chain_M_pos, cols["bias_by_res_all"], gens, Ls, B, omit_AA_mask=cols["omit_AA_mask"],
                                    temperature=c["T"], pssm_coef=cols["pssm_coef"], pssm_bias=cols["pssm_bias"],
                                    pssm_log_odds_mask=(cols["pssm_log_odds_all"] > self.pssm_threshold).float())   # stock: pssm_log_odds_mask = (pssm_log_odds_all > args.pssm_threshold).float()
        return sd

    def advance(self, c):
        """One turn of the batch's decode on its lane (the lane's stream current, the batch's scope entered): up to DECODE_CHUNK steps enqueued.
        Returns False once the whole decode is enqueued (the sampled tensors then in c["sd"], the lane's events recorded)."""
        if c["done"]: return False
        sD = self.sD[c["lane"]]; self._lane = c["lane"]
        with self._stream(sD), torch.no_grad():
            self._enter(c)
            try:
                next(c["gen"]); return True
            except StopIteration as fin:
                sd = fin.value
            if c["keep"]:
                sd = {"S": sd["S"].clone(), "probs": sd["probs"].clone(), "decoding_order": sd["decoding_order"]}
            c.update({"sd": sd, "ev_sample": (c["ev_sample0"], self._event(sD)), "t_sample_host": time.time() - c["t_sample0"], "done": True})
        return False

    def rescore(self, c):
        """the re-scoring forward (stock: model(X, S_sample, ..., randn_2, use_input_decoding_order=True, ...)) on the re-scoring stream once the batch's
        sampling is in order before it; the scores stay on the device until finalize."""
        cols, Ls, B, K, sd = c["cols"], c["Ls"], c["B"], c["K"], c["sd"]
        X, S, mask, chain_M, chain_encoding_all, chain_M_pos, residue_idx = (cols[k] for k in ["X","S","mask","chain_M","chain_encoding_all","chain_M_pos","residue_idx"])
        mask_for_loss = c["mask_for_loss"]
        with self._stream(self.sR):
            if self.sR is not None: self.sR.wait_event(c["ev_sample"][1]); self.sR.wait_event(c["ev_native"][1])   # its own sampled S and mask_for_loss; not a later batch's decode
            e0 = self._event(self.sR); t0 = time.time()
            self._enter(c)
            with torch.no_grad():
                log_probs = self.forward_scores(X, sd["S"], mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, c["randn_2"], B, K, use_input_decoding_order=True, decoding_order=sd["decoding_order"])
            log_probs = log_probs.float()
            if getattr(self, "stock_shape_enc", False):
                # XATTEMPT: reduce over each backbone's OWN length (stock tensor shape [B, L_b]); a padded row changes the float32 sum tree
                scores = [(_scores(sd["S"][b*B:(b+1)*B, :L], log_probs[b*B:(b+1)*B, :L], mask_for_loss[b*B:(b+1)*B, :L]),
                           _scores(sd["S"][b*B:(b+1)*B, :L], log_probs[b*B:(b+1)*B, :L], mask[b*B:(b+1)*B, :L])) for b, L in enumerate(Ls)]
            else:
                scores = [(_scores(sd["S"], log_probs, mask_for_loss), _scores(sd["S"], log_probs, mask))]
            c.update({"log_probs": log_probs, "scores_dev": scores, "ev_rescore": (e0, self._event(self.sR)), "t_rescore_host": time.time()-t0})
        c["rounds_out"].append({k: c[k] for k in ("T", "j", "sd", "randn_2", "log_probs", "scores_dev", "ev_sample", "ev_rescore", "t_sample_host", "t_rescore_host")})   # the round's tensors stay referenced here until finalize: the streams still read them
        return c

    def finalize(self, c, out_dir):
        """wait for this batch's device work, move its results to the host and write upstream's files (stock format: the native record where stock's round
        loop writes it (native_header_due), every round's B sequences in stock's order — temperature-major, batch index, row — numbered j*B + b_ix + 1; the
        score / probability arrays concatenated over the rounds as stock concatenates its lists); returns the batch's record (names, K, t_start, t_end, forward_s = its sampling + re-scoring device
        spans + the write, call_s = t_end - t_start)."""
        cols, Ls, B, K, feats, proteins = c["cols"], c["Ls"], c["B"], c["K"], c["feats"], c["proteins"]
        S, chain_M, mask = cols["S"], cols["chain_M"], cols["mask"]
        mask_for_loss, rounds = c["mask_for_loss"], c["rounds_out"]
        if c["ev_rescore"][1] is not None:
            c["ev_rescore"][1].synchronize()                        # the host waits for THIS batch's last device work (its last round's re-scoring: every earlier round precedes it on its streams), not the device
            ms = lambda ev: ev[0].elapsed_time(ev[1]) / 1000.0
            t_nat, t_smp, t_res = ms(c["ev_native"]), sum(ms(r["ev_sample"]) for r in rounds), sum(ms(r["ev_rescore"]) for r in rounds)
        else:
            t_nat, t_smp, t_res = c["t_native_host"], sum(r["t_sample_host"] for r in rounds), sum(r["t_rescore_host"] for r in rounds)
        self.timers["encoder_fwd"] += t_nat; self.timers["sample"] += t_smp; self.timers["rescore"] += t_res
        t0 = time.time()
        native_score = np.concatenate([a.cpu().numpy() for a, _ in c["native_score_dev"]]); global_native_score = np.concatenate([g.cpu().numpy() for _, g in c["native_score_dev"]])
        host = [(r["T"], r["j"], np.concatenate([a.cpu().numpy() for a, _ in r["scores_dev"]]), np.concatenate([g.cpu().numpy() for _, g in r["scores_dev"]]),
                 r["sd"]["S"].cpu().numpy(), r["sd"]["probs"].cpu().numpy(), r["log_probs"].cpu().numpy()) for r in rounds]   # per round: T, j, scores, global_scores, S_sample, probs, log_probs on the host
        S_np = S.cpu().numpy(); mfl_np = mask_for_loss.cpu().numpy(); chain_M_np = chain_M.cpu().numpy()
        os.makedirs(os.path.join(out_dir, "seqs"), exist_ok=True)
        if self.save_probs: os.makedirs(os.path.join(out_dir, "probs"), exist_ok=True)     # as stock: the folder exists only when its gate is open
        if self.save_score: os.makedirs(os.path.join(out_dir, "scores"), exist_ok=True)
        for b, (p, f, L) in enumerate(zip(proteins, feats, Ls)):
            rows = slice(b*B, (b+1)*B)
            name_ = p["name"]
            masked_chain_length_list = f[9][0]; masked_list = f[8][0]; visible_list = f[7][0]; chain_list = f[6]
            chain_M_b = chain_M_np[rows, :L]
            lines = []
            native_seq = _S_to_seq(S_np[b*B, :L], chain_M_b[0])            # stock's native record (the same text each time it is due: the native score is the pre-loop forward's)
            start = end = 0; loa = []
            for ml in masked_chain_length_list:
                end += ml; loa.append(native_seq[start:end]); start = end
            native_seq = "".join(list(np.array(loa)[np.argsort(masked_list)]))
            l0 = 0
            for mc in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                l0 += mc; native_seq = native_seq[:l0] + "/" + native_seq[l0:]; l0 += 1
            pm = [masked_list[i] for i in np.argsort(masked_list)]; pv = [visible_list[i] for i in np.argsort(visible_list)]
            header = ">{}, score={}, global_score={}, fixed_chains={}, designed_chains={}, model_name={}, git_hash={}, seed={}\n{}\n".format(
                name_, fmt4(native_score[rows].mean()), fmt4(global_native_score[rows].mean()), pv, pm, self.model_name, self.commit, self.seed, native_seq)
            for (T, j, scores, global_scores, Ss_np, probs_np, lp_np) in host:   # stock: for temp: for j: for b_ix
                if native_header_due(T, j, self.rounds[0][0]):                 # stock: `if b_ix == 0 and j==0 and temp==temperatures[0]` inside the round loop
                    lines.append(header)
                for b_ix in range(B):
                    r = b*B + b_ix
                    srr = np.float32(np.sum((S_np[r, :L] == Ss_np[r, :L]).astype(np.float32) * mfl_np[r, :L], dtype=np.float32)) / np.float32(np.sum(mfl_np[r, :L], dtype=np.float32))   # == stock's float32 sums (0/1 terms) and quotient, on the host copies
                    seq = _S_to_seq(Ss_np[r, :L], chain_M_b[b_ix])
                    start = end = 0; loa = []
                    for ml in masked_chain_length_list:
                        end += ml; loa.append(seq[start:end]); start = end
                    seq = "".join(list(np.array(loa)[np.argsort(masked_list)]))
                    l0 = 0
                    for mc in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                        l0 += mc; seq = seq[:l0] + "/" + seq[l0:]; l0 += 1
                    lines.append(">T={}, sample={}, score={}, global_score={}, seq_recovery={}\n{}\n".format(T, j*B+b_ix+1, fmt4(scores[r]), fmt4(global_scores[r]), fmt4(srr), seq))
            open(os.path.join(out_dir, "seqs", name_ + ".fa"), "w").write("".join(lines))
            if self.save_score:   # == stock: `if args.save_score: np.savez(score_file, score=np.array(score_list, np.float32), ...)` — the lists run over every round
                np.savez(os.path.join(out_dir, "scores", name_ + ".npz"), score=np.array(np.concatenate([h[2][rows] for h in host]), np.float32), global_score=np.array(np.concatenate([h[3][rows] for h in host]), np.float32))
            if self.save_probs:   # == stock: `if args.save_probs: np.savez(probs_file, ...)` — np.concatenate over the rounds' [B, L, 21] / [B, L] arrays
                np.savez(os.path.join(out_dir, "probs", name_ + ".npz"), probs=np.array(np.concatenate([h[5][rows, :L] for h in host]), np.float32), log_probs=np.array(np.concatenate([h[6][rows, :L] for h in host]), np.float32),
                         S=np.array(np.concatenate([h[4][rows, :L] for h in host]), np.int32), mask=mfl_np[rows, :L], chain_order=list(chain_list) * B)   # XATTEMPT: stock writes one (identical) chain list per clone row
        self.timers["d2h_write"] += time.time()-t0
        t_end = time.time()
        if self._enc_cache_key == c.get("enc_key"): self._enc_cache_key = self._enc_cache_val = None
        return {"names": [p["name"] for p in proteins], "n": K, "n_seqs": K * B * len(host), "Ls": Ls, "t_start": c["t_start"], "t_end": t_end, "forward_s": t_smp + t_res + (t_end - t0), "call_s": t_end - c["t_start"]}

    def run_batch(self, proteins, chain_id_dict, out_dir, B=8):
        """one batch start to finish, every round in turn, nothing of another batch in flight (the CPU route; the first batch of a job, which carries the one-time captures)."""
        c = self.decode(self.native(self.prep(proteins, chain_id_dict, B)), keep=len(self.rounds) > 1)
        while self.advance(c): pass
        while self.more_rounds(self.rescore(c)):                            # the next round's decode after this round's re-scoring is issued
            self.decode(c, keep=True)
            while self.advance(c): pass
        return self.finalize(c, out_dir)

    def run_pipelined(self, batches, chain_id_dict, out_dir, B=8):
        """Every batch of the job through the stages, up to DECODE_LANES decode loops in flight (one per lane, their steps enqueued in turns so the
        device interleaves them) with upstream's scoring forwards of the neighbouring batches underneath on their own streams; yields each batch's
        record (finalize) in batch order as its files are written. The first batch runs alone (it carries the one-time captures); on CPU every batch does."""
        if not batches: return
        yield self.run_batch(batches[0], chain_id_dict, out_dir, B)
        if self.sN is None:
            for batch in batches[1:]:
                yield self.run_batch(batch, chain_id_dict, out_dir, B)
            return
        todo = list(batches[1:]); live = []; waiting = []          # live: decodes being enqueued (one per lane); waiting: enqueued, re-scoring issued, files owed (batch order)
        while todo or live or waiting:
            while todo and len(live) < self.lanes:                    # a free lane admits the next batch: its featurisation and native-score forward issued now
                lane = min(set(range(self.lanes)) - {c["lane"] for c in live})
                live.append(self.decode(self.native(self.prep(todo.pop(0), chain_id_dict, B, lane)), keep=True))
            for c in list(live):                                      # one turn per lane
                if not self.advance(c):
                    if len(c.get("rounds_out", [])) + 1 < len(self.rounds):
                        self.decode(self.rescore(c), keep=True)       # this round's re-scoring issued behind its decode; the batch's next round set up on the same lane
                    else:
                        live.remove(c); waiting.append(self.rescore(c))   # issued behind its own decode; runs under the other lanes' steps
            while waiting and (not live or waiting[0]["ev_rescore"][1].query()):
                yield self.finalize(waiting.pop(0), out_dir)          # its scores are on the device (or nothing is left to enqueue: wait for them)

    # ---------- shared encoder ----------
    def encode(self, X, mask, residue_idx, chain_encoding_all, B, K):
        """features + W_e + encoder layers -> (h_V, h_E, E_idx) with N = K*B rows.
        Stock runs them on ONE backbone = B identical clone rows of ITS OWN length L_b. With --stock_shape_enc (the x_all setting) the
        encoder runs per backbone on exactly the stock tensors ([B, L_b, ...] clone rows, unpadded), the outputs are padded to Lmax, and the
        result is cached for the identical re-invocations inside one batch (stock recomputes the same deterministic encoder 3x: native-score
        forward, sample, re-score forward). Bitwise by construction: same shapes, same values, same kernels as stock. Without it the encoder
        runs once on the padded [K*B, Lmax] batch (exact only for backbones with L_b == Lmax: position-wise GEMMs are M-dependent)."""
        model = self.model
        def run_encoder(X, mask, residue_idx, chain_encoding_all):
            E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
            h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device, dtype=E.dtype)
            h_E = model.W_e(E)
            del E                                                    # the raw edge features ([B, L, K, 416]) are embedded; not held through the encoder layers
            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            for layer in model.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
            return h_V, h_E, E_idx
        Ls = getattr(self, "_cur_Ls", None)
        if getattr(self, "stock_shape_enc", False) and Ls is not None:
            key = (X.data_ptr(), tuple(X.shape), mask.data_ptr(), residue_idx.data_ptr(), chain_encoding_all.data_ptr(), tuple(Ls), B, K, getattr(self, "_batch_serial", 0))
            if getattr(self, "_enc_cache_key", None) == key:
                self.graph_stats["enc_cache_hits"] = self.graph_stats.get("enc_cache_hits", 0) + 1
                return self._enc_cache_val
            Lmax = X.shape[1]; N = K * B; out = None
            with self.autocast():
                for bb in range(K):
                    L = Ls[bb]; rows = slice(bb*B, (bb+1)*B)
                    hV, hE, Ib = run_encoder(X[rows, :L], mask[rows, :L], residue_idx[rows, :L], chain_encoding_all[rows, :L])
                    if out is None:
                        # the batch's [N, Lmax, ...] outputs, each backbone's written into its own rows as it is computed (a list of pieces concatenated at
                        # the end held every piece twice: h_E is [N, Lmax, K, H]); pad positions (>= L_b) hold what the concatenation padded with —
                        # zeros for h_V / h_E, the position's own index for E_idx
                        h_V_all = torch.zeros((N, Lmax, hV.shape[2]), device=hV.device, dtype=hV.dtype)
                        h_E_all = torch.zeros((N, Lmax) + tuple(hE.shape[2:]), device=hE.device, dtype=hE.dtype)
                        E_idx_all = torch.arange(Lmax, device=Ib.device, dtype=Ib.dtype)[None, :, None].expand(N, Lmax, Ib.shape[2]).contiguous()
                        out = (h_V_all, h_E_all, E_idx_all)
                    h_V_all[rows, :L] = hV; h_E_all[rows, :L] = hE; E_idx_all[rows, :L] = Ib
            self._enc_cache_key, self._enc_cache_val = key, out
            return out
        with self.autocast():
            h_V, h_E, E_idx = run_encoder(X, mask, residue_idx, chain_encoding_all)
        return h_V, h_E, E_idx

    def release_step_graphs(self, key=None, lane=None):
        """Drop a lane's decode slot — the step CUDA graphs of one decode shape (K, B, Lmax) with their memory pool and the static tensors they read — when
        the coming batch of that lane has ANOTHER shape (key None: every lane's, and the scoring-forward graphs), and hand the allocator's unused cached
        segments back to the device. A batch of the slot's own shape keeps it: its graphs are replayed on the new batch's values (sample_v2)."""
        lanes = range(self.lanes) if lane is None else [lane]
        if key is None:
            self._fwd_slots, self._fwd_pools = {}, {}                # the scoring-forward graphs go with the last decode slots
        dropped = False
        for l in lanes:
            slot = self._slots[l]
            if slot is None or (key is not None and slot["key"] == key):
                continue
            slot["graphs"].clear(); self._slots[l] = None; dropped = True
        if dropped and self.device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.empty_cache()

    def scores_per_backbone(self, K):
        """True when the scoring forwards of a K-backbone batch are upstream's own ``model(...)`` call per backbone (--stock_shape_enc, K > 1): they
        run upstream's encoder themselves and never read the batch's cached encoder outputs (``encode``) — only sampling does. A single-backbone
        batch scores through the cached-encoder path below (three identical encoder invocations served once)."""
        return bool(getattr(self, "stock_shape_enc", False) and getattr(self, "_cur_Ls", None) is not None and K > 1)

    def forward_scores(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, B, K, use_input_decoding_order=False, decoding_order=None):
        """== ProteinMPNN.forward (stock code verbatim after the encoder); with --stock_shape_enc the literal stock call per backbone."""
        model = self.model; device = X.device
        Ls = getattr(self, "_cur_Ls", None)
        if self.scores_per_backbone(K):
            # XATTEMPT: the scoring forward (encoder + teacher-forced decoder) is the literal stock call per backbone on its own
            # unpadded [B, L_b] clone rows (same shapes => same kernels => bitwise); outputs padded back to Lmax with zeros.
            # The encoder half of that call is the batch's stock-shape encoder pass (encode(): the same statements on the same [B, L_b] tensors,
            # computed once per batch where stock computes it three times); the decoder half is upstream's post-encoder statements (decoder_scores)
            # on that backbone's own contiguous [B, L_b] slices.
            Lmax = X.shape[1]; out = None
            h_V_all, h_E_all, E_idx_all = self.encode(X, mask, residue_idx, chain_encoding_all, B, K)
            with self.autocast():
                for bb in range(K):
                    L = Ls[bb]; rows = slice(bb*B, (bb+1)*B)
                    dec = decoding_order[rows, :L].contiguous() if (use_input_decoding_order and decoding_order is not None) else None
                    lp = self.scoring_forward(h_V_all[rows, :L].contiguous(), h_E_all[rows, :L].contiguous(), E_idx_all[rows, :L].contiguous(),
                                              S[rows, :L].contiguous(), mask[rows, :L].contiguous(), chain_M[rows, :L].contiguous(), randn[rows, :L].contiguous(),
                                              use_input_decoding_order=use_input_decoding_order, decoding_order=dec)
                    if out is None: out = torch.zeros((K*B, Lmax, lp.shape[2]), device=lp.device, dtype=lp.dtype)   # pad positions: zeros, as before
                    out[rows, :L] = lp                                                                                  # copied out before the next backbone's call (a replayed graph writes into one tensor)
            return out
        if not getattr(self, "stock_shape_enc", False):
            with self.autocast():
                return model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, use_input_decoding_order=use_input_decoding_order, decoding_order=decoding_order)
        h_V, h_E, E_idx = self.encode(X, mask, residue_idx, chain_encoding_all, B, K)
        with self.autocast():
            return self.decoder_scores(h_V, h_E, E_idx, S, mask, chain_M, randn, use_input_decoding_order=use_input_decoding_order, decoding_order=decoding_order)

    def decoder_scores(self, h_V, h_E, E_idx, S, mask, chain_M, randn, use_input_decoding_order=False, decoding_order=None):
        """== ProteinMPNN.forward after its encoder (stock statements verbatim): the teacher-forced decoder and log_softmax, given the encoder's outputs."""
        model = self.model; device = h_V.device
        h_S = model.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        chain_M = chain_M*mask
        if not use_input_decoding_order:
            decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(randn)))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in model.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
        logits = model.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    FWD_GRAPHS_KEPT = 2     # captured scoring-forward shapes kept per kind (native | re-scoring); the oldest goes first (a graph keeps its pool's peak)

    def scoring_forward(self, hVb, hEb, Ib, Sb, maskb, chain_Mb, randnb, use_input_decoding_order=False, decoding_order=None):
        """upstream's scoring forward for ONE backbone after the encoder (decoder_scores) on its stock-shape [B, L_b] tensors. On CUDA with the graph levers on,
        a (kind, B, L) shape seen before is captured in a CUDA graph the second time it occurs (the first, eager, call warmed its kernels) and replayed from
        then on with the call's tensors copied into the graph's own: the same kernels on the same shapes and values, one launch instead of upstream's hundred-odd.
        Shapes seen once stay eager; FWD_GRAPHS_KEPT captured shapes are kept per kind (native | re-scoring)."""
        kind = "res" if use_input_decoding_order else "nat"
        if hVb.device.type != "cuda" or not (self.single_graph and self.graph_rng):
            return self.decoder_scores(hVb, hEb, Ib, Sb, maskb, chain_Mb, randnb, use_input_decoding_order=use_input_decoding_order, decoding_order=decoding_order)
        slots = self._fwd_slots.setdefault(kind, {})
        key = (int(hVb.shape[0]), int(hVb.shape[1]), int(Ib.shape[2]))
        sl = slots.get(key)
        if sl is None:                                               # first occurrence: eager (and the warm-up a capture needs)
            slots[key] = {"seen": 1}
            return self.decoder_scores(hVb, hEb, Ib, Sb, maskb, chain_Mb, randnb, use_input_decoding_order=use_input_decoding_order, decoding_order=decoding_order)
        args = (hVb, hEb, Ib, Sb, maskb, chain_Mb, randnb) + ((decoding_order,) if use_input_decoding_order else ())
        if "graph" not in sl:
            captured = [k_ for k_, v_ in slots.items() if "graph" in v_]
            while len(captured) >= self.FWD_GRAPHS_KEPT:
                del slots[captured.pop(0)]                           # the oldest captured shape of this kind makes room
            st = [a.clone() for a in args]
            pool = self._fwd_pools.setdefault(kind, torch.cuda.graph_pool_handle())   # one pool per kind: a kind's graphs replay on one stream, one at a time
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                out = self.decoder_scores(*st[:7], use_input_decoding_order=use_input_decoding_order, decoding_order=(st[7] if use_input_decoding_order else None))
            sl.update({"graph": g, "static": st, "out": out})
            self.graph_stats["captures"] += 1
        else:
            for dst, src in zip(sl["static"], args):
                dst.copy_(src)
        sl["graph"].replay(); self.graph_stats["replays"] += 1
        return sl["out"]

    def dec_layer(self, layer, h_V_t, h_ESV_t, mask_t, B, K):
        if not self.chunk_gemm or K == 1:
            return layer(h_V_t, h_ESV_t, mask_V=mask_t)
        if getattr(self, "hybrid_active", False):
            Kn = int(h_ESV_t.shape[-2]); groups = hybrid_groups(K, B, self.hybrid_group_rows)
            if all((B, g, Kn) in self.hybrid_cells_ok for g in groups):
                if len(groups) == 1:
                    return self.dec_layer_hybrid(layer, h_V_t, h_ESV_t, mask_t, B, K)
                outs, b0 = [], 0
                for g in groups:   # the batch's backbones in groups of g (g*B rows <= the card's ceiling): one batched message GEMM per group; a group of one is the stock call
                    rows = slice(b0 * B, (b0 + g) * B); b0 += g
                    outs.append(self.dec_layer_hybrid(layer, h_V_t[rows], h_ESV_t[rows], mask_t[rows], B, g) if g > 1 else layer(h_V_t[rows], h_ESV_t[rows], mask_V=mask_t[rows]))
                return torch.cat(outs, 0)
        return torch.cat([layer(h_V_t[b*B:(b+1)*B], h_ESV_t[b*B:(b+1)*B], mask_V=mask_t[b*B:(b+1)*B]) for b in range(K)], 0)

    hybrid_active = False
    hybrid_cells_ok = frozenset()
    hybrid_group_rows = None   # the card's row ceiling for one batched message GEMM (HYBRID_GROUP_ROWS[compute capability]); None = the whole decode batch

    def decide_hybrid(self, B, Ks, Kn, requested):
        """Job-level gate for the OPTIONAL --hybrid_gemm lever ('probed cells only, stock shape elsewhere').
        Batching the message GEMMs (W1/W2/W3 + neighbour sum) over K backbones changes the GEMM M from B*Kn to K*B*Kn; that is result-neutral
        only if the kernels the libraries select for both shapes accumulate identically, which depends on the full execution cell:
        device model / sm arch / torch + CUDA + cuBLAS versions / dtype / (B, K, Kn). This method is called ONCE, before any output is produced,
        with every (B, K, Kn) decode-step shape the job will use; it runs each decoder layer both ways on random tensors of exactly those shapes
        (private RNG generator - the default CUDA stream that determines stock's random numbers is untouched) and enables the lever for the whole
        process only if EVERY cell is bit-identical. The verdict is sticky (no mid-job switching); a shape that was not probed uses the stock shape.
        One line is printed and the record is stored in worker_timing.json["hybrid_gemm_probe"]."""
        dev = next(self.model.parameters()).device
        self.hybrid_group_rows = HYBRID_GROUP_ROWS.get(tuple(torch.cuda.get_device_capability(dev))) if dev.type == "cuda" else None
        cell = {"device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                "sm_arch": ("sm_%d%d" % torch.cuda.get_device_capability(dev)) if dev.type == "cuda" else "n/a",
                "torch": torch.__version__, "cuda": torch.version.cuda, "cublas_workspace_cfg": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "dtype": "fp32", "B": B, "Kn": Kn, "K_values": sorted(set(Ks)), "requested": bool(requested)}
        groups = {K: hybrid_groups(K, B, self.hybrid_group_rows) for K in sorted(set(Ks))}   # K backbones -> the GEMM groups that serve them ([K] unless the card's ceiling splits it)
        if self.hybrid_group_rows is not None:   # a card with a row ceiling: the record names it and the split of every K it changes
            cell.update({"group_rows_ceiling": self.hybrid_group_rows, "groups": {str(K): gs for K, gs in groups.items() if gs != [K]}})
        if not requested:
            cell.update({"verdict": "not requested -> stock-shape GEMMs (exact by construction)", "cells": {}})
            self.hybrid_active = False; self.hybrid_cells_ok = frozenset(); self.hybrid_probe_record = cell
            return cell
        results = {}; maxd = 0.0
        g = torch.Generator(device=dev); g.manual_seed(12345)
        with torch.no_grad(), self.autocast():
            for K in sorted({g for gs in groups.values() for g in gs}):   # every GEMM-group size the job's decode batches use (= the K values themselves when no ceiling splits them)
                if K == 1: results["%d,%d,%d" % (B, K, Kn)] = True; continue
                ok = True
                for layer in self.model.decoder_layers:
                    H = layer.W1.out_features; In = layer.W1.in_features
                    h_V = torch.randn((K * B, 1, H), device=dev, generator=g)
                    h_E = torch.randn((K * B, 1, Kn, In - H), device=dev, generator=g)
                    mask_V = (torch.rand((K * B, 1), device=dev, generator=g) > 0.1).float()
                    hyb = self.dec_layer_hybrid(layer, h_V, h_E, mask_V, B, K)
                    ref = torch.cat([layer(h_V[b*B:(b+1)*B], h_E[b*B:(b+1)*B], mask_V=mask_V[b*B:(b+1)*B]) for b in range(K)], 0)
                    if not torch.equal(hyb, ref):
                        ok = False; maxd = max(maxd, (hyb.float() - ref.float()).abs().max().item())
                results["%d,%d,%d" % (B, K, Kn)] = ok
        allok = all(results.values())
        cell.update({"cells": results, "max_abs_diff": maxd, "verdict": ("PROBE PASS -> using hybrid (batched message GEMMs) for all %d cell(s)" % len(results)) if allok else
                     ("PROBE FAIL (%d/%d cells not bit-identical, max|d| %.3g) -> the lever cannot engage bit-identically on this device: the job is refused below (--hybrid_gemm 0 runs the line without it)" % (sum(1 for v in results.values() if not v), len(results), maxd))})
        self.hybrid_active = allok; self.hybrid_cells_ok = frozenset(tuple(int(x) for x in k.split(",")) for k, v in results.items() if v) if allok else frozenset()
        self.hybrid_probe_record = cell
        split = ("; message GEMM groups of <= %d rows on this card: %s" % (self.hybrid_group_rows, ", ".join("K=%s -> %s" % (K, "+".join(str(g) for g in gs)) for K, gs in cell["groups"].items()) or "no K split")) if self.hybrid_group_rows is not None else ""
        print("hybrid_gemm: %s [cell: %s %s torch %s cuda %s dtype %s B=%d Kn=%d K in %s%s]" % (cell["verdict"], cell["device"], cell["sm_arch"], cell["torch"], cell["cuda"], cell["dtype"], B, Kn, cell["K_values"], split), flush=True)
        return cell


    def decide_fused_draw(self, B, requested, dtype=torch.float32, n_steps=16, K=4):
        """Start-up probe of --fused_draw on THIS device / stack: torch's own per-backbone torch.multinomial(probs, 1, generator) on K generators at
        distinct offsets versus the fused kernel, n_steps steps of random probabilities (with exactly-zero entries and one-hot rows, as masked
        amino acids and fixed positions produce), and the Philox increment one draw consumes. PASS only if every sampled index and every
        generator offset agrees; else (or without Triton / CUDA) the lever is off and main() refuses the job by name. One line printed."""
        cell = {"requested": bool(requested), "triton": HAVE_TRITON, "device": str(self.device), "B": B, "steps": n_steps, "dtype": str(dtype)}
        if not requested:
            cell["verdict"] = "not requested"; self.fused_draw = False; return cell
        if self.device.type != "cuda" or not HAVE_TRITON:
            cell["verdict"] = "PROBE FAIL (no %s) -> the fused draw kernel cannot run here" % ("CUDA device" if self.device.type != "cuda" else "Triton")
            self.fused_draw = False; print("fused_draw: %s" % cell["verdict"], flush=True); return cell
        dev = self.device; A = 21
        gp = make_gen(dev, 12345, 0); torch.multinomial(torch.full((B, A), 1.0 / A, device=dev, dtype=dtype), 1, generator=gp); inc = int(gp.get_offset())
        try:
            gcpu = torch.Generator().manual_seed(4321)
            offs = [40 + 1000 * b + 4 * b for b in range(K)]
            gens = [make_gen(dev, self.seed, o) for o in offs]
            seed_t = torch.full((K,), self.seed, dtype=torch.int64, device=dev); off_t = torch.tensor(offs, dtype=torch.int64, device=dev)
            ndraw = torch.zeros(K, dtype=torch.int64, device=dev); active = torch.ones(K, dtype=torch.int64, device=dev)
            logits = (torch.randn(n_steps, K * B, A, generator=gcpu) * 3)
            logits[::3, :, :5] = -1e9; logits[::7, 1] = -1e9; logits[::7, 1, 4] = 0.0        # exactly-zero probabilities and one-hot rows, as masked amino acids / fixed positions give
            probs_all = F.softmax(logits.to(dev).to(dtype), -1)
            ref = torch.zeros((n_steps, K * B), dtype=torch.int64, device=dev); out = torch.full((n_steps, K * B), -1, dtype=torch.int64, device=dev)
            for step in range(n_steps):
                probs = probs_all[step]
                for b in range(K): ref[step, b*B:(b+1)*B] = torch.multinomial(probs[b*B:(b+1)*B], 1, generator=gens[b])[:, 0]
                _fused_draw_kernel[(K,)](probs, out[step], seed_t, off_t, ndraw, active, B=B, BP=_pow2(B), A=A, AP=32, INC=inc, F64=(dtype == torch.float64))
            bad = int((ref != out).any(dim=1).sum().item())                                   # one device round trip for the whole probe
        except Exception as e:                                            # a kernel that does not compile or launch here cannot serve the job: named, refused by main()
            if isinstance(e, MemoryError) or "out of memory" in str(e).lower(): raise   # an out-of-memory is the caller's to see, never a probe verdict
            cell.update({"verdict": "PROBE FAIL (%s: %s) -> the fused draw kernel cannot run here" % (type(e).__name__, str(e).splitlines()[0][:120] if str(e) else ""), "error": type(e).__name__})
            self.fused_draw = False; print("fused_draw: %s" % cell["verdict"], flush=True); return cell
        offs_ok = all(int(g.get_offset()) == o + n_steps * inc for g, o in zip(gens, offs)) and bool((ndraw == n_steps).all().item())
        ok = bad == 0 and offs_ok
        cell.update({"mismatched_steps": bad, "offsets_ok": offs_ok, "inc": inc, "torch": torch.__version__, "triton": triton.__version__,
                     "verdict": ("PROBE PASS -> the fused draw kernel reproduces torch.multinomial on this device (%d steps x %d rows, %s, increment %d)" % (n_steps, K * B, str(dtype).replace("torch.", ""), inc)) if ok else
                                ("PROBE FAIL (%d/%d steps differ, offsets %s) -> the fused draw kernel does not reproduce torch.multinomial here" % (bad, n_steps, "ok" if offs_ok else "differ"))})
        self.fused_draw = ok; self.draw_inc = inc
        print("fused_draw: %s [%s %s torch %s triton %s B=%d]" % (cell["verdict"], torch.cuda.get_device_name(dev), "sm_%d%d" % torch.cuda.get_device_capability(dev), torch.__version__, triton.__version__, B), flush=True)
        return cell

    def dec_layer_hybrid(self, layer, h_V, h_E, mask_V, B, K):
        """XATTEMPT: DecLayer.forward (stock code) with the message GEMMs W1/W2/W3 run on all K*B rows of a GEMM group at once (cuBLAS picks a
        kernel that accumulates like the one for B rows at these shapes: M = rows*Kn >= 384, up to the card's row ceiling HYBRID_GROUP_ROWS —
        bitwise in the kit's GEMM-group probe and in the end-to-end per-step-probs gate), and only the small M=B GEMMs (dense.W_in/W_out)
        chunked per backbone (those DO change kernel with M). LayerNorms/GELU/sum are row-local elementwise/reduction ops with row-independent results."""
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = layer.W3(layer.act(layer.W2(layer.act(layer.W1(h_EV)))))
        dh = torch.sum(h_message, -2) / layer.scale
        h_V = layer.norm1(h_V + layer.dropout1(dh))
        dh = torch.cat([layer.dense(h_V[b*B:(b+1)*B]) for b in range(K)], 0)
        h_V = layer.norm2(h_V + layer.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

    def w_out(self, h_V_t, B, K):
        if not self.chunk_gemm or K == 1:
            return self.model.W_out(h_V_t)
        return torch.cat([self.model.W_out(h_V_t[b*B:(b+1)*B]) for b in range(K)], 0)

    def autocast(self):
        return torch.autocast("cuda", enabled=False)   # float32 throughout, as stock

    # ---------- stock sample(), batched over K backbones with per-backbone generators ----------
    def sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask, chain_M_pos, bias_by_res, gens, Ls, B, omit_AA_mask=None, temperature=None,
               pssm_coef=None, pssm_bias=None, pssm_log_odds_mask=None):
        model = self.model; device = X.device; K = len(Ls); N, Lmax = X.shape[0], X.shape[1]
        temperature = self.T if temperature is None else temperature              # the round's temperature (stock: sample(..., temperature=temp))
        if pssm_coef is None:                                                       # tied_featurize's own defaults (no --pssm_jsonl): zeros, and an all-pass log-odds mask
            pssm_coef = torch.zeros((N, Lmax), device=device); pssm_bias = torch.zeros((N, Lmax, 21), device=device); pssm_log_odds_mask = torch.ones((N, Lmax, 21), device=device)
        omit_AA_mask_flag = omit_AA_mask is not None
        h_V, h_E, E_idx = self.encode(X, mask, residue_idx, chain_encoding_all, B, K)
        h_V = h_V.float(); h_E = h_E.float()
        chain_mask = chain_mask*chain_M_pos*mask
        # decoding order: stock argsort on each backbone's OWN (B, L) tensor; pad positions appended last
        decoding_order = torch.zeros((N, Lmax), dtype=torch.int64, device=device)
        for b, L in enumerate(Ls):
            rows = slice(b*B, (b+1)*B)
            decoding_order[rows, :L] = torch.argsort((chain_mask[rows, :L]+0.0001)*(torch.abs(randn[rows, :L])))
            if L < Lmax:
                decoding_order[rows, L:] = torch.arange(L, Lmax, device=device)[None, :]
        mask_bw, mask_fw = self.order_masks(decoding_order, E_idx, mask)          # stock's statements; their temporaries end with the call
        all_probs = torch.zeros((N, Lmax, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N, Lmax), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(model.decoder_layers))]
        constant, constant_bias = self.constant, self.constant_bias
        # stock next builds h_EXV_encoder_fw = mask_fw * cat_neighbors_nodes(h_V, cat_neighbors_nodes(zeros_like(h_S), h_E, E_idx), E_idx) for ALL
        # positions ([N, L, K, 3H] plus two intermediates: 12.5 GB per process at N = 128, L = 500) and reads one position of it per decode step;
        # sample_v2's step assembles that position from h_V / h_E with the same gathers and the same float32 product (the same values), so the
        # whole tensors are never materialised.
        # draw schedule per backbone (stock: draw iff not (mask_gathered==0).all(); clones identical -> mask at decoded position)
        # XATTEMPT fix (affects only backbones with missing residues, mask==0): stock decides per STEP over its batch of B clones,
        # each clone having its OWN random decoding order: skip (no decoder, no draw) iff mask[row, order[row, t_]] == 0 for ALL B rows.
        draw_ok = []
        for b, L in enumerate(Ls):
            m = mask[b*B:(b+1)*B, :L]                                                  # [B, L]
            mg = torch.gather(m, 1, decoding_order[b*B:(b+1)*B, :L])                   # [B, L] mask at each row's own decoded position
            draw_ok.append((~(mg == 0).all(dim=0)).tolist())                            # stock: draw unless (mask_gathered==0).all()
        masked_steps = [[(t_ < L and not draw_ok[b][t_]) for t_ in range(Lmax)] for b, L in enumerate(Ls)]
        v = locals()
        sd = yield from self.sample_v2(v)                                                  # the decode loop, enqueued DECODE_CHUNK steps a turn
        return sd

    @staticmethod
    def order_masks(decoding_order, E_idx, mask):
        """== the stock statements of ProteinMPNN.sample that turn the decoding order into the decoder's backward / forward attention masks
        (mask_bw, mask_fw: [N, L, K, 1]); a function of their own so the mask temporaries are released on return (the carried statements build
        [N, L, L] intermediates; the staged executable carries the low-memory form of the same three statements in their place)."""
        device = E_idx.device
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum("ij, biq, bjp->bqp", (1-torch.triu(torch.ones(mask_size, mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        return mask_bw, mask_fw

    # ---------- the decode loop: per-step encoder context, whole-step CUDA graph incl. RNG (eager without --graph_rng) ----------
    def sample_v2(self, v):
        model = self.model; device = self.device
        E_idx, h_E, S_true, chain_mask, mask, bias_by_res = v["E_idx"], v["h_E"], v["S_true"], v["chain_mask"], v["mask"], v["bias_by_res"]
        h_V_stack, mask_bw, mask_fw, all_probs, decoding_order, gens, Ls, B = v["h_V_stack"], v["mask_bw"], v["mask_fw"], v["all_probs"], v["decoding_order"], v["gens"], v["Ls"], v["B"]
        draw_ok = v["draw_ok"]; masked_steps = v["masked_steps"]; K = len(Ls); omit_AA_mask = v["omit_AA_mask"]; omit_flag = v["omit_AA_mask_flag"]
        T = v["temperature"]                                                      # this round's temperature: a Python float, as stock's; a captured step graph is the graph of (pattern, T)
        pssm_coef_new, pssm_bias_new, pssm_log_odds_mask_new = v["pssm_coef"], v["pssm_bias"], v["pssm_log_odds_mask"]
        N, Lmax, Kn = E_idx.shape[0], E_idx.shape[1], E_idx.shape[2]
        H = h_E.shape[-1]
        dec_T_new = decoding_order.t().contiguous()   # [Lmax, N]
        # the decode slot: the step graphs of this decode shape and the tensors they read and write. A batch of the slot's shape copies its values INTO
        # those tensors (same shapes, same dtypes: the same kernels on the new values) and replays the graphs; another shape builds a new slot
        # (release_step_graphs dropped the old one). The state tensors start from zeros exactly as stock's fresh ones do.
        key = (K, B, Lmax)
        slot = self._slots[self._lane]
        if slot is None or slot["key"] != key:
            slot = {"key": key, "graphs": {}, "keep_cache": {}, "warm_shape": None,
                    "pool": torch.cuda.graph_pool_handle() if device.type == "cuda" else None,
                    "E_idx": E_idx.clone(), "h_E": h_E.clone(), "S_true": S_true.clone(), "chain_mask": chain_mask.clone(), "mask": mask.clone(), "bias_by_res": bias_by_res.clone(),
                    "mask_bw": mask_bw.clone(), "mask_fw": mask_fw.clone(), "dec_T": dec_T_new.clone(),
                    "pssm_coef": pssm_coef_new.clone(), "pssm_bias": pssm_bias_new.clone(), "pssm_log_odds_mask": pssm_log_odds_mask_new.clone(),
                    "omit_AA_mask": omit_AA_mask.clone() if omit_AA_mask is not None else None,
                    "h_V_stack": [h.clone() for h in h_V_stack], "all_probs": all_probs.clone(), "h_S": v["h_S"].clone(), "S": v["S"].clone(),
                    "zeros_EX": torch.zeros((N, 1, Kn, H), device=device, dtype=h_E.dtype),      # the node half of stock's h_EX_encoder = cat_neighbors_nodes(zeros_like(h_S), h_E, E_idx), one position
                    "t_static": torch.zeros((N,), dtype=torch.int64, device=device), "step_idx": torch.zeros((), dtype=torch.int64, device=device),
                    "rng_seed": torch.full((K,), int(self.seed), dtype=torch.int64, device=device), "rng_off": torch.zeros((K,), dtype=torch.int64, device=device),
                    "ndraw": torch.zeros((K,), dtype=torch.int64, device=device), "active_cache": {},
                    "ar": torch.arange(N, device=device)}                                       # the row index of the per-position selections
            self._slots[self._lane] = slot
        else:
            for nm, src in (("E_idx", E_idx), ("h_E", h_E), ("S_true", S_true), ("chain_mask", chain_mask), ("mask", mask), ("bias_by_res", bias_by_res), ("mask_bw", mask_bw), ("mask_fw", mask_fw), ("dec_T", dec_T_new),
                            ("pssm_coef", pssm_coef_new), ("pssm_bias", pssm_bias_new), ("pssm_log_odds_mask", pssm_log_odds_mask_new)):
                slot[nm].copy_(src)
            if omit_AA_mask is not None: slot["omit_AA_mask"].copy_(omit_AA_mask)
            slot["h_V_stack"][0].copy_(h_V_stack[0])
            for h in slot["h_V_stack"][1:]: h.zero_()
            slot["all_probs"].zero_(); slot["h_S"].zero_(); slot["S"].zero_()
        E_idx, h_E, S_true, chain_mask, mask, bias_by_res, mask_bw, mask_fw, omit_AA_mask = (slot[k] for k in ("E_idx", "h_E", "S_true", "chain_mask", "mask", "bias_by_res", "mask_bw", "mask_fw", "omit_AA_mask"))
        pssm_coef, pssm_bias, pssm_log_odds_mask = slot["pssm_coef"], slot["pssm_bias"], slot["pssm_log_odds_mask"]
        pssm_multi, pssm_bias_flag, pssm_log_odds_flag = self.pssm_multi, bool(self.pssm_bias_flag), bool(self.pssm_log_odds_flag)
        h_V_stack, all_probs, h_S, S, zeros_EX = slot["h_V_stack"], slot["all_probs"], slot["h_S"], slot["S"], slot["zeros_EX"]
        h_V_enc = h_V_stack[0]                                                    # the encoder's node embeddings (stock's h_V inside sample())
        constant, constant_bias = self.constant, self.constant_bias
        nL = len(model.decoder_layers)
        ar = slot["ar"]              # every tensor the step graphs read or write is the slot's: a graph captured for one batch is replayed for the next
        ctx = self.cache_enc_ctx
        probs_dtype = torch.promote_types(torch.float32, self.constant_bias.dtype)
        # per-backbone step kinds (stock decides per backbone = per its own batch of B clones): 2 = decoder + draw; 0 = stock SKIPS the step
        # (all B rows at a missing residue: no decoder, no draw, h_V_stack/all_probs untouched); 1 = pad step of a shorter backbone (t_ >= L_b:
        # decoder runs on pad positions, which real positions never read; no draw). A step where every backbone is 0/1 needs no decoder at all.
        def kind(b, t_):
            if t_ >= Ls[b]: return 1
            return 0 if masked_steps[b][t_] else 2
        all_masked_t = [all(kind(b, t_) != 2 for b in range(K)) for t_ in range(Lmax)]
        row_of_bb = torch.arange(N, device=device) // B
        dec_T = slot["dec_T"]                     # [Lmax, N]

        def step_core(t):
            """decoder for position t (LongTensor [N]); returns probs [N,21] (stock op sequence; gathers optionally replaced)."""
            # position t of every row, selected by index (the values stock's [:, t] slices / gathers read; one kernel each, no repeated index tensors)
            chain_mask_gathered = chain_mask[ar, t][:, None]
            bias_by_res_gathered = bias_by_res[ar, t]
            E_idx_t = E_idx[ar, t][:, None]
            h_E_t = h_E[ar, t][:, None]
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            if ctx:
                mask_fw_t = mask_fw[ar, t][:, None]
                mask_bw_t = mask_bw[ar, t][:, None]
            else:
                mask_fw_t = torch.gather(mask_fw, 1, t[:, None, None, None].repeat(1, 1, Kn, mask_fw.shape[-1]))
                mask_bw_t = torch.gather(mask_bw, 1, t[:, None, None, None].repeat(1, 1, Kn, mask_bw.shape[-1]))
            # stock reads h_EXV_encoder_fw[:, t] of mask_fw * cat_neighbors_nodes(h_V, cat_neighbors_nodes(zeros_like(h_S), h_E, E_idx), E_idx): the same two
            # gathers on this position's E_idx_t / h_E_t and the same float32 product give that slice's values without the [N, L, K, 3H] tensors
            h_EXV_encoder_t = mask_fw_t * cat_neighbors_nodes(h_V_enc, torch.cat([h_E_t, zeros_EX], -1), E_idx_t)
            mask_t = mask[ar, t][:, None]
            with self.autocast():
                for l, layer in enumerate(model.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = h_V_stack[l][ar, t][:, None]
                    h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l+1][ar, t] = self.dec_layer(layer, h_V_t, h_ESV_t, mask_t, B, K).float()[:, 0]
                h_V_t = h_V_stack[-1][ar, t]
                logits = self.w_out(h_V_t, B, K).float() / T
            probs = F.softmax(logits-constant[None, :]*1e8+constant_bias[None, :]/T+bias_by_res_gathered/T, dim=-1)
            if pssm_bias_flag:                                                 # == stock sample(): the --pssm_bias_flag statements (the gathers read position t by index: the same values)
                pssm_coef_gathered = pssm_coef[ar, t]
                pssm_bias_gathered = pssm_bias[ar, t]
                probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
            if pssm_log_odds_flag:                                             # == stock sample(): the --pssm_log_odds_flag statements
                pssm_log_odds_mask_gathered = pssm_log_odds_mask[ar, t]
                probs_masked = probs*pssm_log_odds_mask_gathered
                probs_masked += probs * 0.001
                probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True)
            if omit_flag:
                omit_AA_mask_gathered = omit_AA_mask[ar, t]
                probs_masked = probs*(1.0-omit_AA_mask_gathered)
                probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True)
            all_probs[ar, t] = (chain_mask_gathered[:, :, None]*probs[:, None, :]).float()[:, 0]
            return probs, chain_mask_gathered

        fused = bool(self.fused_draw) and device.type == "cuda"
        rng_seed, rng_off, ndraw, active_cache = slot["rng_seed"], slot["rng_off"], slot["ndraw"], slot["active_cache"]
        def active_vec(pattern):
            """[K] int64: 1 for backbones that draw at this step kind, 0 for the skipped (all-masked) ones. Built OUTSIDE graph capture."""
            if pattern not in active_cache:
                active_cache[pattern] = torch.tensor([1 if (pattern[b] == 2 or pattern[b] is True) else 0 for b in range(K)], dtype=torch.int64, device=device)
            return active_cache[pattern]
        def draw(probs, S_t, pattern):
            if fused:                                               # every drawing backbone's torch.multinomial(probs_b, 1, generator=g_b) in one kernel (decide_fused_draw probed it)
                _fused_draw_kernel[(K,)](probs, S_t, rng_seed, rng_off, ndraw, active_vec(pattern), B=B, BP=_pow2(B), A=21, AP=32, INC=self.draw_inc, F64=(probs.dtype == torch.float64))
                return S_t
            for b in range(K):
                if pattern[b] == 2 or pattern[b] is True:
                    S_t[b*B:(b+1)*B] = torch.multinomial(probs[b*B:(b+1)*B], 1, generator=gens[b])
            return S_t

        keep_cache = slot["keep_cache"]
        def keep_vec(pattern):
            """[N,1,1] float: 0 for rows of backbones with kind 0 (stock skips the step), else 1. Built OUTSIDE graph capture."""
            if pattern not in keep_cache:
                sel = torch.tensor([pattern[b] == 0 for b in range(K)], dtype=torch.bool, device=device)
                keep_cache[pattern] = (~sel[row_of_bb]).to(all_probs.dtype)[:, None, None]
            return keep_cache[pattern]

        def unskip(t, pattern):
            """rows of backbones with kind 0 at this step: restore the stock 'untouched' state (zeros) at their position t."""
            if not any(pattern[b] == 0 for b in range(K)): return
            keep = keep_vec(pattern)[:, 0]                                             # [N, 1]: 0 for rows to reset
            for l in range(nL):
                cur = h_V_stack[l+1][ar, t]
                h_V_stack[l+1][ar, t] = cur * keep.to(cur.dtype) * keep.to(cur.dtype).abs()   # x*1*1 = x ; x*0*0 = +0.0 (also for x=-0.0)
            cur = all_probs[ar, t]
            all_probs[ar, t] = cur * keep * keep.abs()

        def commit(t, S_t, chain_mask_gathered):
            S_true_gathered = S_true[ar, t][:, None]
            S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()
            temp1 = model.W_s(S_t)
            h_S[ar, t] = temp1[:, 0]
            S[ar, t] = S_t[:, 0]
            return S_t

        if not self.graph_rng:
            # eager: the same step functions run from the host loop (no CUDA device, or the CUDA-graph levers not requested)
            for t_ in range(Lmax):
                t = dec_T[t_]
                if all_masked_t[t_]:
                    chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])
                    S_t = torch.gather(S_true, 1, t[:, None])
                else:
                    pattern = tuple(kind(b, t_) for b in range(K))
                    probs, chain_mask_gathered = step_core(t)
                    unskip(t, pattern)
                    S_t = torch.gather(S_true, 1, t[:, None]).clone()
                    S_t = draw(probs, S_t, pattern)
                commit(t, S_t, chain_mask_gathered)
            return {"S": S, "probs": all_probs, "decoding_order": decoding_order}

        # ---- whole-step CUDA graph including the multinomial draws on registered per-backbone generators ----
        # one graph per draw pattern (tuple of which backbones draw at this step); patterns come in <= K+1 contiguous runs
        patterns = []
        no_masked = not any(kind(b, t_) == 0 for b in range(K) for t_ in range(Ls[b]))
        for t_ in range(Lmax):
            if getattr(self, "single_graph", False) and no_masked:
                patterns.append(tuple(2 for _ in Ls))        # one all-draw graph per batch: pad steps of shorter backbones also draw (their own
                                                             # generator only; reset analytically below) — exact
            else:
                patterns.append(None if all_masked_t[t_] else tuple(kind(b, t_) for b in range(K)))
        t_static, step_idx, graphs = slot["t_static"], slot["step_idx"], slot["graphs"]
        # the graphs point at the slot's tensors: built once per (shape, draw pattern), replayed for every batch of that shape
        self._v2_graphs = graphs
        self._v2_pool = slot["pool"]                         # one pool shared by the slot's graphs (they replay sequentially)
        def build(pattern):
            t0 = time.time()
            # warm-up on a side stream: consumes RNG and mutates state -> snapshot + restore everything it touches
            snap_gen = [g.get_state() for g in gens]; snap_nd = ndraw.clone()
            snap = [h_S.clone(), S.clone(), all_probs.clone()] + [h.clone() for h in h_V_stack[1:]]
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            keep_vec(pattern)   # allocate outside capture
            n_warm = 0 if slot["warm_shape"] == (N, Lmax, Kn) else 2
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(n_warm):
                    torch.index_select(dec_T, 0, step_idx.reshape(1)).reshape(N)  # exercise
                    t = dec_T[0].clone(); t_static.copy_(t)
                    probs, cmg = step_core(t_static); unskip(t_static, pattern); S_t = torch.gather(S_true, 1, t_static[:, None]).clone(); S_t = draw(probs, S_t, pattern); commit(t_static, S_t, cmg)
            slot["warm_shape"] = (N, Lmax, Kn)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
            for g_, st_ in zip(gens, snap_gen): g_.set_state(st_)
            ndraw.copy_(snap_nd)                                             # the fused kernel's per-backbone draw counter, likewise
            h_S.copy_(snap[0]); S.copy_(snap[1]); all_probs.copy_(snap[2])
            for h, hs in zip(h_V_stack[1:], snap[3:3+nL]): h.copy_(hs)
            g = torch.cuda.CUDAGraph()
            if not fused:                                                    # torch's draws in the graph read the registered generators; the fused kernel reads the slot's (seed, offset, count) tensors
                for b in range(K):
                    if pattern[b] == 2: g.register_generator_state(gens[b])
            with torch.no_grad(), torch.cuda.graph(g, pool=slot["pool"]):          # this lane's pool (another lane may have run since the call began)
                t_static.copy_(torch.index_select(dec_T, 0, step_idx.reshape(1)).reshape(N))
                probs, cmg = step_core(t_static)
                unskip(t_static, pattern)
                S_t = torch.gather(S_true, 1, t_static[:, None]).clone()
                S_t = draw(probs, S_t, pattern)
                commit(t_static, S_t, cmg)
                step_idx.add_(1)
            torch.cuda.synchronize()
            self.graph_stats["captures"] += 1; self.graph_stats["capture_s"] += time.time()-t0
            return g
        off0 = [g.get_offset() for g in gens]                                   # each backbone's stock-stream offset where its draws begin
        if fused:
            for p_ in set(patterns) - {None}: active_vec(p_)                            # the step kinds' draw masks, before any capture
            assert all(o % 4 == 0 for o in off0), off0                                  # torch keeps Philox offsets at multiples of 4 (the kernel reads whole blocks)
            rng_off.copy_(torch.tensor(off0, dtype=torch.int64)); ndraw.zero_()       # the kernel reads (seed, off0[b] + ndraw[b] * inc) and counts its draws
        t_ = 0; enq = 0
        while t_ < Lmax:
            p = patterns[t_]
            if p is None:   # all-masked step: stock does no decoder work and no draw; commit S_true (eager, cheap, rare)
                t = dec_T[t_]
                commit(t, torch.gather(S_true, 1, t[:, None]), torch.gather(chain_mask, 1, t[:, None]))
                t_ += 1; continue
            if (p, T) not in graphs:                                        # one graph per (draw pattern, temperature): the temperature is a constant of the captured step
                step_idx.fill_(t_)
                graphs[(p, T)] = build(p)
            step_idx.fill_(t_)
            # replay the run of identical patterns without any host-side per-step work except the launch
            run_end = t_
            while run_end < Lmax and patterns[run_end] == p: run_end += 1
            g = graphs[(p, T)]
            for r_ in range(run_end - t_):
                g.replay()
                enq += 1
                if enq % self.DECODE_CHUNK == 0: yield                # the lane's turn ends: the host serves the other lanes, then resumes here
            self.graph_stats["replays"] += run_end - t_
            t_ = run_end
        if fused or (getattr(self, "single_graph", False) and no_masked):
            # exact bookkeeping for the persistent generators: the stock stream of backbone b consumes one draw increment per step it draws at
            # (L_b minus its all-masked steps); all-draw graph replays consume Lmax of them and the fused kernel none. Each generator is set to
            # where stock's stream stands after this backbone either way.
            step_inc = self.draw_inc
            if step_inc is None:
                gp = make_gen(device, 0); torch.multinomial(torch.full((B, 21), 1.0/21, device=device, dtype=probs_dtype), 1, generator=gp); step_inc = int(gp.get_offset())
            for b, (g, L) in enumerate(zip(gens, Ls)):
                n_draws = L - sum(1 for t_ in range(L) if masked_steps[b][t_])
                g.set_offset(off0[b] + n_draws * step_inc)
        return {"S": S, "probs": all_probs, "decoding_order": decoding_order}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_path", required=True); ap.add_argument("--chain_id_jsonl", required=True); ap.add_argument("--out_folder", required=True)
    ap.add_argument("--path_to_model_weights", default=os.path.join(MPNN_DIR, "soluble_model_weights")); ap.add_argument("--model_name", default="v_48_020")
    ap.add_argument("--seed", type=int, default=37, help="as stock CLI: 0 draws a random seed per process, as protein_mpnn_run.py draws it")
    ap.add_argument("--sampling_temp", type=str, default="0.1", help="as stock CLI: one temperature or several separated by blanks ('0.1 0.2'): a round of batches per temperature")
    ap.add_argument("--num_seq_per_target", type=int, default=8); ap.add_argument("--batch_size", type=int, default=8)   # as stock CLI: num_seq_per_target // batch_size batches of batch_size sequences per temperature
    ap.add_argument("--omit_AA_jsonl", default="", help="as stock CLI"); ap.add_argument("--bias_AA_jsonl", default="", help="as stock CLI"); ap.add_argument("--bias_by_res_jsonl", default="", help="as stock CLI")
    ap.add_argument("--pssm_jsonl", default="", help="as stock CLI"); ap.add_argument("--pssm_multi", type=float, default=0.0, help="as stock CLI"); ap.add_argument("--pssm_threshold", type=float, default=0.0, help="as stock CLI")
    ap.add_argument("--pssm_log_odds_flag", type=int, default=0, help="as stock CLI"); ap.add_argument("--pssm_bias_flag", type=int, default=0, help="as stock CLI")
    ap.add_argument("--mode", default="stream", choices=["stream"], help="stream: the one-process stock RNG stream replayed over the whole jsonl (each backbone at its own Philox offset)")
    ap.add_argument("--bb_batch", type=int, default=1); ap.add_argument("--chunk_gemm", action="store_true")
    ap.add_argument("--sort_by_length", action="store_true", help="group backbones of similar length per batch (stream mode keeps each backbone's own offset, so order is free)")
    ap.add_argument("--max_length", type=int, default=200000)
    ap.add_argument("--cache_enc_ctx", action="store_true"); ap.add_argument("--graph_rng", action="store_true")
    ap.add_argument("--single_graph", action="store_true", help="XATTEMPT: one all-draw CUDA graph per batch (exact; see sample_v2)")
    ap.add_argument("--fused_draw", action="store_true", help="the per-backbone draws of a decode step in one kernel that computes torch's own multinomial (Philox exponential + argmax) bit for bit; probed at start-up, CUDA + Triton")
    ap.add_argument("--analytic_offsets", action="store_true", help="XATTEMPT: compute stream offsets from per-call Philox increments")
    ap.add_argument("--hybrid_gemm", action="store_true", help="XATTEMPT opt-in: with --chunk_gemm, run the message GEMMs (W1/W2/W3) batched over backbones and chunk only the M=B GEMMs. "
                    "Bit-identical to stock only if cuBLAS picks a batch-invariant kernel for those shapes on THIS device (true on H100/H200/B200, false on L40S in our sweep): "
                    "guarded by an on-device probe at start-up: unless batched == chunked bitwise for every group size used the job is refused by name (exit 3) before any output")
    ap.add_argument("--stock_shape_enc", action="store_true", help="XATTEMPT: encoder per backbone on stock-shaped (unpadded, B-clone) tensors + exact in-batch cache (fixes M-dependent GEMM drift for short backbones in mixed-length batches)")
    ap.add_argument("--x_all", action="store_true", help="XATTEMPT: recommended exact set = --chunk_gemm --cache_enc_ctx --graph_rng --single_graph --fused_draw --analytic_offsets --stock_shape_enc (every GEMM at the stock shape -> stock kernels on any GPU)")
    ap.add_argument("--omit_AAs", default="X", help="as stock CLI (e.g. XC)"); ap.add_argument("--fixed_positions_jsonl", default="", help="as stock CLI")
    ap.add_argument("--save_score", type=int, default=0, help="as stock CLI: 1 writes scores/<name>.npz (0, upstream's default: not written)")
    ap.add_argument("--save_probs", type=int, default=0, help="as stock CLI: 1 writes probs/<name>.npz (0, upstream's default: not written)")
    ap.add_argument("--inputs_ready", default="", help="a file the caller creates once --jsonl_path is written: the start-up that needs no inputs (CUDA, weights, the draw probe) runs first and the inputs are read when it exists")
    args = ap.parse_args()
    if args.x_all: args.chunk_gemm = args.cache_enc_ctx = args.graph_rng = args.single_graph = args.fused_draw = args.analytic_offsets = args.stock_shape_enc = True   # --hybrid_gemm is not part of --x_all: it is device-dependent and probe-gated (decide_hybrid below)
    if not torch.cuda.is_available(): args.graph_rng = args.single_graph = args.fused_draw = False   # CUDA-graph levers and the fused draw kernel are GPU-only
    seed = args.seed if args.seed else int(np.random.randint(0, high=999, size=1, dtype=int)[0])   # == protein_mpnn_run.py: --seed 0 (its default) draws the process's seed; the .fa headers carry it
    if seed != args.seed:
        print("seed: --seed 0 (upstream's default) -> drew %d for this process, as protein_mpnn_run.py draws it (recorded in every .fa header and the end-of-run record)" % seed, flush=True)
    NUM_BATCHES = args.num_seq_per_target // args.batch_size                      # == protein_mpnn_run.py: NUM_BATCHES batches of BATCH_COPIES = batch_size per temperature
    temperatures = [float(item) for item in args.sampling_temp.split()]            # == protein_mpnn_run.py
    if NUM_BATCHES < 1:
        print("inputs: --num_seq_per_target %d < --batch_size %d: num_seq_per_target // batch_size = 0 batches, nothing to design" % (args.num_seq_per_target, args.batch_size), flush=True); sys.exit(2)
    rounds = [(temp, j) for temp in temperatures for j in range(NUM_BATCHES)]     # stock's `for temp in temperatures: for j in range(NUM_BATCHES)`
    seqs_per_item = len(rounds) * args.batch_size                                  # sequences written per backbone
    Worker.OMIT_AAS = args.omit_AAs
    global FIXED_POSITIONS_DICT, OMIT_AA_DICT, BIAS_BY_RES_DICT, PSSM_DICT
    FIXED_POSITIONS_DICT = load_jsonl_dict(args.fixed_positions_jsonl, option="fixed_positions_jsonl")   # every dictionary by protein_mpnn_run.py's one rule (last line; a missing file = not loaded)
    OMIT_AA_DICT = load_jsonl_dict(args.omit_AA_jsonl, option="omit_AA_jsonl"); BIAS_BY_RES_DICT = load_jsonl_dict(args.bias_by_res_jsonl, option="bias_by_res_jsonl")
    PSSM_DICT = load_jsonl_dict(args.pssm_jsonl, merge=True, option="pssm_jsonl")
    bias_AA_dict = load_jsonl_dict(args.bias_AA_jsonl, option="bias_AA_jsonl")
    bias_AAs_np = np.zeros(len(ALPHABET))                                          # == protein_mpnn_run.py: bias_AAs_np from --bias_AA_jsonl
    if bias_AA_dict:
        for n, AA in enumerate(ALPHABET):
            if AA in list(bias_AA_dict.keys()):
                bias_AAs_np[n] = bias_AA_dict[AA]
    Worker.BIAS_AAS_NP = bias_AAs_np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T = {}
    torch.manual_seed(seed)   # as stock: seeds CPU + CUDA default generators before model construction
    t0 = time.time(); model, ck = load_model(os.path.join(args.path_to_model_weights, args.model_name + ".pt"), device); sync(); T["t_model_load_s"] = time.time()-t0
    global BASE_CPU_STATE
    BASE_CPU_STATE = torch.get_rng_state() if device.type != "cuda" else None
    w = Worker(model, device, temperatures[0], seed, chunk_gemm=args.chunk_gemm, cache_enc_ctx=args.cache_enc_ctx, graph_rng=args.graph_rng)
    w.rounds = rounds                                                              # every (temperature, batch) round of the job, B = batch_size sequences per backbone each
    w.pssm_multi, w.pssm_threshold, w.pssm_log_odds_flag, w.pssm_bias_flag = args.pssm_multi, args.pssm_threshold, bool(args.pssm_log_odds_flag), bool(args.pssm_bias_flag)   # as stock hands them to sample()
    w.fused_draw = bool(args.fused_draw)
    w.save_score, w.save_probs = bool(args.save_score), bool(args.save_probs)   # upstream's write gates, as given (0 = not written, its default)
    w.single_graph, w.analytic_offsets, w.hybrid_gemm, w.stock_shape_enc = args.single_graph, args.analytic_offsets, args.hybrid_gemm, args.stock_shape_enc
    w.model_name = args.model_name
    T["fused_draw_probe"] = w.decide_fused_draw(args.batch_size, requested=bool(args.fused_draw), dtype=torch.promote_types(torch.float32, w.constant_bias.dtype))   # the dtype stock's probs have (its bias tensor is float64)
    if args.fused_draw and not w.fused_draw:
        T["refused"] = "fused_draw: " + ("PROBE FAIL" if HAVE_TRITON else "triton unavailable")
        print("fused_draw: REFUSED -> the lever cannot engage bit-identically on this stack and the line is all of its levers; nothing designed (rerun without --fused_draw, i.e. the levers spelled out: the line without the lever)", flush=True)
        print(json.dumps(T)); sys.exit(3)
    if args.inputs_ready:                                            # started ahead of the inputs: everything above needed none; read them once the caller says they are written
        t0 = time.time()
        while not os.path.exists(args.inputs_ready):
            if time.time() - t0 > 1800:
                print("inputs: --inputs_ready %s never appeared (30 min): nothing to design" % args.inputs_ready, flush=True); sys.exit(2)
            time.sleep(0.005)
        T["t_inputs_wait_s"] = time.time() - t0
    t0 = time.time(); ds = StructureDatasetPDB([json.loads(l) for l in open(args.jsonl_path)], truncate=None, max_length=args.max_length, verbose=False); T["t_dataset_s"] = time.time()-t0
    chain_id_dict = load_jsonl_dict(args.chain_id_jsonl, option="chain_id_jsonl")   # protein_mpnn_run.py's rule: the last line's object; no file = None (every chain designed)
    proteins = list(ds)
    if args.mode == "stream":
        # stream offsets are in DATASET order regardless of processing order: precompute per backbone from (L, n_draw)
        t0 = time.time(); offs = {}; o = 0; prev = []
        for p in proteins:
            f = featurize_one(p, device, chain_id_dict); w.feats[p["name"]] = f      # stock's tied_featurize of this backbone, kept for its batch (prep)
            L = int(f[0].shape[1])
            if not bool((f[2][0] != 0).all()):
                # gapped backbone (missing residues): position a generator after randn_1 of this protein and derive each round's B decoding
                # orders exactly as stock does, to count the steps at which stock draws in that round (all-rows-masked steps are skipped)
                g1 = make_gen(device, seed, o if device.type == "cuda" else list(prev)); torch.randn((args.batch_size, L), device=device, generator=g1)
                n_draws = w.n_draws_stock(f, args.batch_size, g1, len(rounds))
            else:
                n_draws = [L] * len(rounds)
            if device.type == "cuda":
                offs[p["name"]] = o; o += w.consumption(L, n_draws, args.batch_size)
            else:
                offs[p["name"]] = list(prev)
            prev.append((L, n_draws))
        w.offsets = offs; T["t_offsets_s"] = time.time()-t0; T["stream_total_offset"] = o
    order = list(range(len(proteins)))
    if args.sort_by_length: order.sort(key=lambda i: len(proteins[i]["seq"]))
    os.makedirs(args.out_folder, exist_ok=True)
    # the batches: --bb_batch backbones each in processing order; a backbone shorter than k_neighbors residues is a batch of its own, by name (the
    # batched featuriser's pad-free neighbour sets need L >= k; alone it runs the stock shapes) — the lever steps aside for that backbone, never the job
    batches = plan_batches(order, [len(p["seq"]) for p in proteins], args.bb_batch, int(ck["num_edges"]))
    n_short = sum(1 for p in proteins if len(p["seq"]) < ck["num_edges"])
    if n_short and args.bb_batch > 1:
        print("bb_batch: %d backbone(s) shorter than k_neighbors=%d residues run one per batch (stock shapes, k = L); the other %d in batches of up to %d" % (n_short, ck["num_edges"], len(proteins) - n_short, args.bb_batch), flush=True)
    T["short_backbones"] = n_short
    # job-level, sticky decision for the optional --hybrid_gemm lever, BEFORE any output (all decode-step cells of this job are known here)
    Ks = [len(b) for b in batches]
    T["hybrid_gemm_probe"] = w.decide_hybrid(args.batch_size, Ks, int(ck["num_edges"]), requested=bool(args.hybrid_gemm and args.chunk_gemm))
    if bool(args.hybrid_gemm and args.chunk_gemm) and not w.hybrid_active:
        # the line is all of its levers: the probe says --hybrid_gemm cannot engage bit-identically on this device, so the job is refused by name
        # before any output (never a run under the line's name with a subset); --hybrid_gemm 0 is the line without the lever, by name
        T["refused"] = "hybrid_gemm: PROBE FAIL"
        print("hybrid_gemm: REFUSED -> the probe-gated lever cannot engage bit-identically on this device and the line is all of its levers; nothing designed (rerun with --hybrid_gemm 0: the line without the lever)", flush=True)
        print(json.dumps(T)); sys.exit(3)
    t_run0 = time.time(); done = 0
    for fin in w.run_pipelined([[proteins[j] for j in bi] for bi in batches], chain_id_dict, args.out_folder, args.batch_size):
        done += fin["n"]
    w.release_step_graphs()
    sync(); T["t_run_s"] = time.time()-t_run0; T["n_backbones"] = done; T["n_seqs"] = done * seqs_per_item
    T.update({"timers": w.timers, "mode": args.mode, "bb_batch": args.bb_batch, "seed": seed, "seed_requested": args.seed, "temperatures": temperatures, "num_batches": NUM_BATCHES,
              "num_seq_per_target": args.num_seq_per_target, "batch_size": args.batch_size,
              "save_score": args.save_score, "save_probs": args.save_probs, "chunk_gemm": args.chunk_gemm, "cache_enc_ctx": args.cache_enc_ctx, "graph_rng": args.graph_rng, "single_graph": args.single_graph, "fused_draw": args.fused_draw, "analytic_offsets": args.analytic_offsets, "hybrid_gemm": args.hybrid_gemm, "hybrid_gemm_active": bool(w.hybrid_active), "stock_shape_enc": args.stock_shape_enc, "graph_stats": w.graph_stats})
    n_fa = len([f for f in os.listdir(os.path.join(args.out_folder, "seqs")) if f.endswith(".fa")])
    T["n_fa"] = n_fa; T["expected_fa"] = len(proteins)
    json.dump(T, open(os.path.join(args.out_folder, "worker_timing.json"), "w"), indent=1)
    print(json.dumps(T))

if __name__ == "__main__":
    main()
