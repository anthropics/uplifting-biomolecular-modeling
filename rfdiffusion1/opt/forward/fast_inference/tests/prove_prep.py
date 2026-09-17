#!/usr/bin/env python
"""tests/prove_prep.py — assert-equal proof for lever P (drivers/rfd_prep.py) on real trajectories (default: 2 designs x 50 steps, insulin_target.pdb A1-115 + 80-mer = 195 tokens, from the upstream examples).

Static part : locates the overwrite statement `t2d[..., :44] = t2d_44` in rfdiffusion/inference/model_runners.py (pinned 86507b65) and
              asserts it sits inside `class SelfConditioning` (sample_step), i.e. the 44 template planes returned by `_preprocess` are
              overwritten before the model call for the runner every released checkpoint uses.
Dynamic part: for every denoising step, calls upstream `_preprocess` and P's `_preprocess` on cloned (seq_t, x_t, t) and asserts
  (A1) msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, xyz_t, alpha_t exactly equal (NaN-aware);
  (A2) t2d: same shape/dtype/device; planes 44: (if any) exactly equal; planes :44 are allowed to differ (dead);
  then runs the whole design twice (once with upstream `_preprocess`, once with P) with a forward pre-hook on the network recording the
  t2d ACTUALLY fed to RoseTTAFold at each step and asserts (A3) those are exactly equal step by step and (A4) final coordinates equal.
Exit 0 + 'PROVE_PREP_OK {...}' only if everything holds."""
import os, sys, json, argparse, hashlib
p = argparse.ArgumentParser()
p.add_argument('--rfd-root', default='/opt/rfd'); p.add_argument('--weights', default='/weights')
p.add_argument('--pdb', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inputs_public', 'insulin_target.pdb')); p.add_argument('--contig', default='[A1-115/0 80-80]')
p.add_argument('--hotspots', default='[A59,A83,A91]'); p.add_argument('--out', default='out/prove_prep'); p.add_argument('--n', type=int, default=2)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, args.rfd_root); sys.path.insert(0, os.path.join(here, '..', 'drivers')); sys.path.insert(0, here)
os.environ.setdefault('DGLBACKEND', 'pytorch')
import torch, numpy as np, random
from hydra import compose, initialize_config_dir
from rfdiffusion.inference import utils as iu
from rfdiffusion.inference import model_runners as MR
import rfd_prep

src = open(os.path.join(args.rfd_root, 'rfdiffusion', 'inference', 'model_runners.py')).read().splitlines()
site = [i + 1 for i, l in enumerate(src) if l.strip().startswith('t2d[..., :44] = t2d_44')]
encl = []
for ln in site:
    cls = fn = None
    for j in range(ln - 1, -1, -1):
        s = src[j]
        if fn is None and s.lstrip().startswith('def ') and (len(s) - len(s.lstrip())) == 4:
            fn = s.strip().split('(')[0][4:]
        if s.startswith('class '):
            cls = s.split('(')[0][6:].strip(': '); break
    encl.append((ln, cls, fn))
print('OVERWRITE_SITE', encl, flush=True)
assert any(c == 'SelfConditioning' and f == 'sample_step' for _, c, f in encl), 'overwrite statement not found inside SelfConditioning.sample_step'

ov = [f"inference.input_pdb={args.pdb}", f"inference.output_prefix={os.path.abspath(args.out)}/des", f"inference.model_directory_path={args.weights}",
      f"inference.num_designs={args.n}", f"contigmap.contigs={args.contig}", f"ppi.hotspot_res={args.hotspots}", "inference.deterministic=True",
      "inference.write_trajectory=False", "inference.cautious=False"]
with initialize_config_dir(config_dir=f"{args.rfd_root}/config/inference", version_base=None):
    conf = compose(config_name="base", overrides=ov)
sampler = iu.sampler_selector(conf)
assert type(sampler) is MR.SelfConditioning, type(sampler)
assert rfd_prep.apply(sampler)
PRE_P = MR.Sampler._preprocess            # patched
PRE_UP = MR.Sampler._preprocess_upstream   # original


def nan_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        return False
    if a.is_floating_point():
        na, nb = torch.isnan(a), torch.isnan(b)
        return bool((na == nb).all()) and bool(torch.equal(torch.where(na, torch.zeros_like(a), a), torch.where(nb, torch.zeros_like(b), b)))
    return bool(torch.equal(a, b))


names = ['msa_masked', 'msa_full', 'seq_in', 'xt_in', 'idx_pdb', 't1d', 't2d', 'xyz_t', 'alpha_t']
counts = dict(steps_compared=0, outputs_equal=0, t2d_dead_planes_differed_steps=0, t2d_fed_to_model_equal_steps=0, designs_final_equal=0)
fed = {}
sampler.model.register_forward_pre_hook(lambda mod, a, kw: fed.__setitem__('t2d', kw['t2d'].detach().clone()), with_kwargs=True)


def run_design(i_des, which, compare):
    MR.Sampler._preprocess = PRE_UP if which == 'upstream' else PRE_P
    torch.manual_seed(i_des); np.random.seed(i_des); random.seed(i_des)
    x_init, seq_init = sampler.sample_init()
    x_t = torch.clone(x_init); seq_t = torch.clone(seq_init)
    feds = []
    for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1):
        if compare:
            a_up = PRE_UP(sampler, torch.clone(seq_t), torch.clone(x_t), t)
            a_p = PRE_P(sampler, torch.clone(seq_t), torch.clone(x_t), t)
            for k, (u, v) in enumerate(zip(a_up, a_p)):
                if names[k] == 't2d':
                    assert u.shape == v.shape and u.dtype == v.dtype and u.device == v.device, ('t2d meta', u.shape, v.shape, u.dtype, v.dtype, u.device, v.device)
                    if u.shape[-1] > 44:
                        assert nan_equal(u[..., 44:], v[..., 44:]), 't2d planes 44: differ'
                    counts['t2d_dead_planes_differed_steps'] += int(not nan_equal(u[..., :44], v[..., :44]))
                else:
                    assert nan_equal(u, v), f'step t={t}: {names[k]} differs between upstream _preprocess and P'
                    counts['outputs_equal'] += 1
            counts['steps_compared'] += 1
        px0, x_t, seq_t, plddt = sampler.sample_step(t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step)
        feds.append(fed['t2d'])
    return hashlib.sha256(x_t.cpu().numpy().tobytes()).hexdigest()[:16], feds


# warm-up trajectory first: RFdiffusion's first design after process start follows a different last-bit arithmetic path (TorchScript
# profiling executor specialising on first calls; a stock property) - all later designs are position-independent.
_ = run_design(args.n, 'upstream', compare=False)
print('warm-up design done (F2)', flush=True)
for i_des in range(args.n):
    h_up, feds_up = run_design(i_des, 'upstream', compare=True)
    h_p, feds_p = run_design(i_des, 'P', compare=False)
    assert len(feds_up) == len(feds_p)
    for a, b in zip(feds_up, feds_p):
        assert nan_equal(a, b), 't2d fed to the model differs between the upstream-preprocess run and the P run'
        counts['t2d_fed_to_model_equal_steps'] += 1
    assert h_up == h_p, f'design {i_des}: final coordinates differ ({h_up} vs {h_p})'
    counts['designs_final_equal'] += 1
    print('design', i_des, 'final x_t sha', h_up, '(upstream) ==', h_p, '(P)', flush=True)
res = dict(counts, overwrite_site=encl, prep_stats=rfd_prep.stats(), L=int(sampler.binderlen) if False else None)
print('PROVE_PREP_OK', json.dumps(res), flush=True)
json.dump(res, open(os.path.join(args.out, 'prove_prep.json'), 'w'), indent=1)
