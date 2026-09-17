"""
rfd_prep — lever P (exact): per-step input preparation of RFdiffusion v1 (commit 86507b65), `Sampler._preprocess`.

Upstream `_preprocess(seq, xyz_t, t)` is called once per denoising step and
  (P1) computes `t2d = xyz_to_t2d(xyz_t)` on the CPU (an L x L x 44 template feature map: Cb distance one-hot + 6 orientation
       planes) although `SelfConditioning.sample_step` overwrites ALL 44 planes immediately afterwards
       (`t2d[..., :44] = xyz_to_t2d(prev_pred)` for t < T, zeros at t == T).  The CPU result is provably dead for the
       SelfConditioning model runner (the one every released checkpoint uses); only the 3 block-adjacency planes appended by
       the ScaffoldedSampler subclass survive, and those do not come from xyz_to_t2d.  -> we build t2d[..., :44] as zeros.
  (P2) rebuilds per-design constants on every step: msa_masked (1,1,L,48), msa_full (1,1,L,25), the 21 sequence planes of t1d
       (with a python loop over L), idx, the hotspot plane — all functions of (seq, contig, hotspots) which are constant during a
       trajectory (seq_t == seq_init at every step for this runner) — and copies each of them host->device every step.
       -> memoised per design (cache is dropped at every `sample_init`, and additionally keyed on the seq tensor's bytes).
  Everything that depends on x_t (xyz_t NaN-masking / 27-atom padding, alpha_t = get_torsions on the CPU, the time feature)
  is computed by the UNCHANGED upstream code on the same device (CPU) in the same order, so every tensor handed to the model is
  byte-equal to upstream's.  No random number is consumed by `_preprocess` (upstream or patched).

Exactness argument per output (vs upstream, same step):  msa_masked, msa_full, seq[None], idx, t1d[..., :21], hotspot & legacy
planes: equal values (same constructing code, run once);  t1d time plane: same python expression `1 - t / T` assigned into a
float32 tensor;  xyz_t / xt_in / alpha_t: upstream code verbatim;  t2d[..., :44]: dead (overwritten by the caller), t2d[..., 44:]
(scaffold-guided runner only): produced by the untouched subclass code from the returned tensor.
Guard: the patch is only active when the runner is `SelfConditioning` (or a subclass) — the plain `Sampler.sample_step` DOES feed
the CPU t2d to the model, so for that runner P1 would not be dead; we fall back to upstream there.

Use:  import rfd_prep; rfd_prep.apply(sampler)      # after sampler construction; RFD_PREP=0 -> no-op
      rfd_prep.stats()
"""
import os
import torch

_state = dict(applied=False, n_calls=0, n_full=0, n_cached=0, epoch=0)
_cache = dict()


def stats():
    return dict(_state)


def reset():
    _cache.clear()
    _state['epoch'] += 1


def apply(sampler):
    """Patch this sampler instance (SelfConditioning runner). Returns True if active."""
    if os.environ.get('RFD_PREP', '1') == '0':
        print('rfd_prep: disabled by RFD_PREP=0', flush=True)
        return False
    from rfdiffusion.inference import model_runners as MR
    if not isinstance(sampler, MR.SelfConditioning):
        print(f'rfd_prep: runner {type(sampler).__name__} is not SelfConditioning -> upstream _preprocess kept (P1 not provably dead)', flush=True)
        return False
    if _state['applied']:
        return True
    from rfdiffusion import util
    from rfdiffusion.chemical import INIT_CRDS  # noqa: F401  (import equality with upstream module)
    from rfdiffusion.util_module import ComputeAllAtomCoords  # noqa: F401
    TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES = util.torsion_indices, util.torsion_can_flip, util.reference_angles

    base_cls = MR.Sampler
    orig_preprocess = base_cls._preprocess
    orig_sample_init = type(sampler).sample_init

    def sample_init_wrapped(self, *a, **k):
        reset()                      # new design -> drop memoised constants
        return orig_sample_init(self, *a, **k)
    type(sampler).sample_init = sample_init_wrapped

    def _constants(self, seq):
        """the t- and x_t-independent part of upstream _preprocess, computed with the upstream statements verbatim."""
        L = seq.shape[0]
        msa_masked = torch.zeros((1, 1, L, 48))
        msa_masked[:, :, :, :22] = seq[None, None]
        msa_masked[:, :, :, 22:44] = seq[None, None]
        msa_masked[:, :, 0, 46] = 1.0
        msa_masked[:, :, -1, 47] = 1.0
        msa_full = torch.zeros((1, 1, L, 25))
        msa_full[:, :, :, :22] = seq[None, None]
        msa_full[:, :, 0, 23] = 1.0
        msa_full[:, :, -1, 24] = 1.0
        t1d = torch.zeros((1, 1, L, 21))
        seqt1d = torch.clone(seq)
        for idx in range(L):
            if seqt1d[idx, 21] == 1:
                seqt1d[idx, 20] = 1
                seqt1d[idx, 21] = 0
        t1d[:, :, :, :21] = seqt1d[None, None, :, :21]
        # time feature placeholder (filled per step below, same expression as upstream)
        timefeature = torch.zeros((L)).float()
        timefeature = timefeature[None, None, ..., None]
        t1d = torch.cat((t1d, timefeature), dim=-1).float()          # (1,1,L,22) on CPU, col 21 = time
        idx_t = torch.tensor(self.contig_map.rf)[None]
        # seq_tmp used by get_torsions depends only on t1d[..., :-1] (the 21 seq planes) -> constant
        seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
        dev = dict(msa_masked=msa_masked.to(self.device), msa_full=msa_full.to(self.device), seq=seq.to(self.device),
                   idx=idx_t.to(self.device))
        hot = None
        if self.preprocess_conf.d_t1d >= 24:
            hotspot_tens = torch.zeros(L).float()
            if self.ppi_conf.hotspot_res is None:
                print("WARNING: you're using a model trained on complexes and hotspot residues, without specifying hotspots.\
                         If you're doing monomer diffusion this is fine")
                hotspot_idx = []
            else:
                hotspots = [(i[0], int(i[1:])) for i in self.ppi_conf.hotspot_res]
                hotspot_idx = []
                for i, res in enumerate(self.contig_map.con_ref_pdb_idx):
                    if res in hotspots:
                        hotspot_idx.append(self.contig_map.hal_idx0[i])
                hotspot_tens[hotspot_idx] = 1.0
            hot = hotspot_tens[None, None, ..., None].to(self.device)
        return dict(L=L, t1d_cpu=t1d, seq_tmp=seq_tmp, dev=dev, hot=hot, seq_ref=seq.detach().clone())

    def preprocess_fast(self, seq, xyz_t, t, repack=False):
        _state['n_calls'] += 1
        ent = _cache.get('const')
        if ent is None or ent['L'] != seq.shape[0] or ent['seq_ref'].device != seq.device or not torch.equal(ent['seq_ref'], seq):
            ent = _constants(self, seq)
            _cache['const'] = ent
            _state['n_full'] += 1
        else:
            _state['n_cached'] += 1
        L = ent['L']
        # ---- t1d time feature: same statements as upstream, written into (a copy of) the cached CPU t1d ----
        t1d = ent['t1d_cpu'].clone()
        timefeature = torch.zeros((L)).float()
        timefeature[self.mask_str.squeeze()] = 1            # NOTE: read live — upstream get_next_pose sets mask_str[:] = False in place after step 1
        timefeature[~self.mask_str.squeeze()] = 1 - t / self.T
        t1d[0, 0, :, 21] = timefeature
        # ---- xyz_t: upstream statements verbatim (operate on the caller's tensor exactly like upstream does) ----
        if self.preprocess_conf.sidechain_input:
            xyz_t[torch.where(seq == 21, True, False), 3:, :] = float("nan")
        else:
            xyz_t[~self.mask_str.squeeze(), 3:, :] = float("nan")
        xyz_t = xyz_t[None, None]
        xyz_t = torch.cat((xyz_t, torch.full((1, 1, L, 13, 3), float("nan"))), dim=3)
        # ---- t2d: P1 — the 44 template planes are dead for the SelfConditioning runner; zeros on device ----
        t2d = torch.zeros((1, 1, L, L, 44), device=self.device)
        # ---- alpha_t: upstream statements verbatim (CPU) ----
        alpha, _, alpha_mask, _ = util.get_torsions(xyz_t.reshape(-1, L, 27, 3), ent['seq_tmp'], TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[..., 0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1, -1, L, 10, 2)
        alpha_mask = alpha_mask.reshape(1, -1, L, 10, 1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)
        # ---- device copies ----
        d = ent['dev']
        xyz_t = xyz_t.to(self.device)
        t1d = t1d.to(self.device)
        alpha_t = alpha_t.to(self.device)
        if ent['hot'] is not None:
            t1d = torch.cat((t1d, torch.zeros_like(t1d[..., :1]), ent['hot']), dim=-1)
        # clones: callers mutate some of these in place (ScaffoldedSampler: idx_pdb += 200) — a device clone is one tiny D2D kernel, no host sync
        return (d['msa_masked'].clone(), d['msa_full'].clone(), d['seq'].clone()[None], torch.squeeze(xyz_t, dim=0), d['idx'].clone(), t1d, t2d, xyz_t, alpha_t)

    base_cls._preprocess = preprocess_fast
    base_cls._preprocess_upstream = orig_preprocess
    _state['applied'] = True
    print('rfd_prep: lever P active (P1 dead CPU xyz_to_t2d skipped, P2 per-design constants memoised)', flush=True)
    return True
