"""tests/_tiny_sampler.py -- a miniature of the STOCK OpenDDE 1.1.1 diffusion sampler for the CPU tests of the sampler levers (stepgraph).

Real upstream classes (opendde.model.modules.diffusion.DiffusionModule, opendde.model.generator.sample_diffusion, the model-side
`opendde.model.opendde.sample_diffusion` name the model resolves at call time) at tiny widths with random weights; the only stand-in is
`TinyModel`, an object with `.diffusion_module` (what the kit's DITFAST unit and the attention levers need from the model).
Every Linear that upstream zero-initialises is re-drawn (a zero weight would make hoisted terms trivially equal)."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

import torch


def kit_paths() -> Dict[str, str]:
    here = os.path.dirname(os.path.abspath(__file__))                       # opt/opendde_opt/tests
    levers = os.path.join(os.path.dirname(os.path.dirname(here)), "forward", "fast_inference", "levers")
    return {"levers": levers, "ditfast": os.path.join(levers, "DITFAST", "tools"), "accel": os.path.join(levers, "ACCEL")}


class TinyModel:
    """What the kit's units need from the model: `.diffusion_module`."""

    def __init__(self, dm):
        self.diffusion_module = dm


def build(seed: int = 0, n_token: int = 12, atoms_per_token: Tuple[int, int] = (2, 5), c_s: int = 16, c_z: int = 8, c_s_inputs: int = 12,
          c_token: int = 32, c_atom: int = 16, c_atompair: int = 8, dit_blocks: int = 2, dit_heads: int = 2, atom_blocks: int = 1,
          atom_heads: int = 2, dtype=torch.float32, extra_bias: bool = True, device: str = "cpu", init: str = "tiny") -> Dict[str, Any]:
    """A DiffusionModule (eval, random weights) + a consistent feature dict + the sampler's cached conditioning (pair_z, p_lm, c_l)."""
    from opendde.model.modules.diffusion import DiffusionModule
    from opendde.model.opendde import update_input_feature_dict

    g = torch.Generator().manual_seed(seed)
    dm = DiffusionModule(sigma_data=16.0, c_atom=c_atom, c_atompair=c_atompair, c_token=c_token, c_s=c_s, c_z=c_z, c_z_pair_diffusion=c_z,
                         c_s_inputs=c_s_inputs, atom_encoder={"n_blocks": atom_blocks, "n_heads": atom_heads},
                         transformer={"n_blocks": dit_blocks, "n_heads": dit_heads}, atom_decoder={"n_blocks": atom_blocks, "n_heads": atom_heads})
    with torch.no_grad():
        for p in dm.parameters():                                   # upstream zero-inits many projections: re-draw everything (small, finite)
            if init == "fan_in":                                    # real widths: keep activations O(1) (matrices ~ 1/sqrt(fan_in); LN scales ~ 1)
                p.copy_(torch.randn(p.shape, generator=g) * (p.shape[-1] ** -0.5) if p.dim() > 1 else torch.randn(p.shape, generator=g) * 0.1 + 1.0)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * (0.3 if p.dim() > 1 else 0.5) + (1.0 if p.dim() == 1 else 0.0))
    dm = dm.to(dtype).eval()
    counts = torch.randint(atoms_per_token[0], atoms_per_token[1] + 1, (n_token,), generator=g)
    a2t = torch.repeat_interleave(torch.arange(n_token), counts)
    n_atom = int(a2t.numel())
    feats: Dict[str, Any] = {
        "atom_to_token_idx": a2t,
        "ref_pos": torch.randn(n_atom, 3, generator=g).to(dtype),
        "ref_charge": torch.randint(-1, 2, (n_atom,), generator=g).to(dtype),
        "ref_mask": torch.ones(n_atom, dtype=dtype),
        "ref_element": torch.nn.functional.one_hot(torch.randint(0, 128, (n_atom,), generator=g), 128).to(dtype),
        "ref_atom_name_chars": torch.nn.functional.one_hot(torch.randint(0, 64, (n_atom, 4), generator=g), 64).to(dtype),
        "ref_space_uid": a2t.clone(),
    }
    rel_c = dm.diffusion_conditioning.relpe.linear_no_bias.in_features
    feats["relp"] = torch.nn.functional.one_hot(torch.randint(0, rel_c, (n_token, n_token), generator=g), rel_c).to(dtype)
    if extra_bias:
        feats["structural_pair_attn_bias"] = (torch.randn(n_token, n_token, generator=g) * 0.1).to(dtype)
    s_inputs = torch.randn(n_token, c_s_inputs, generator=g).to(dtype)
    s_trunk = torch.randn(n_token, c_s, generator=g).to(dtype)
    z_trunk = torch.randn(n_token, n_token, c_z, generator=g).to(dtype)
    if device != "cpu":                                             # everything drawn on CPU (one reproducible stream), then moved
        dm = dm.to(device)
        feats = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in feats.items()}
        s_inputs, s_trunk, z_trunk = s_inputs.to(device), s_trunk.to(device), z_trunk.to(device)
    feats = update_input_feature_dict(feats)
    with torch.no_grad():
        pair_z = dm.diffusion_conditioning.prepare_cache(feats["relp"], z_trunk, False)
        p_lm, c_l = dm.atom_attention_encoder.prepare_cache(
            ref_pos=feats["ref_pos"], ref_charge=feats["ref_charge"], ref_mask=feats["ref_mask"], ref_element=feats["ref_element"],
            ref_atom_name_chars=feats["ref_atom_name_chars"], atom_to_token_idx=feats["atom_to_token_idx"], d_lm=feats["d_lm"], v_lm=feats["v_lm"],
            pad_info=feats["pad_info"], r_l=True, z=pair_z, inplace_safe=False)
    return {"dm": dm, "model": TinyModel(dm), "feats": feats, "s_inputs": s_inputs, "s_trunk": s_trunk, "z_trunk": z_trunk, "pair_z": pair_z,
            "p_lm": p_lm, "c_l": c_l, "n_token": n_token, "n_atom": n_atom}


def sampler_kwargs(b: Dict[str, Any], n_step: int = 3, n_sample: int = 2, seed: int = 7, chunk=None, inplace_safe: bool = True,
                   efficient_fusion: bool = True) -> Dict[str, Any]:
    """The keyword call the model makes (opendde.model.opendde.OpenDDE.run_sample_diffusion_stage -> .sample_diffusion)."""
    from opendde.model.generator import InferenceNoiseScheduler
    sched = InferenceNoiseScheduler()(N_step=n_step, device=b["s_inputs"].device, dtype=b["s_inputs"].dtype)
    return dict(denoise_net=b["dm"], input_feature_dict=b["feats"], s_inputs=b["s_inputs"], s_trunk=b["s_trunk"], z_trunk=None, pair_z=b["pair_z"],
                pair_z_spec=None, p_lm=b["p_lm"], c_l=b["c_l"], atom_window_spec=None, foldcp_attention_bias=None, N_sample=n_sample,
                noise_schedule=sched, attn_chunk_size=None, diffusion_chunk_size=chunk, inplace_safe=inplace_safe,
                enable_efficient_fusion=efficient_fusion, rollout_seed=seed, foldcp_group=None,
                gamma0=0.8, gamma_min=1.0, noise_scale_lambda=1.003, step_scale_eta=1.5, guidance_configs=None)


def run_sampler(b: Dict[str, Any], **kw) -> torch.Tensor:
    """Call the sampler by the module-global name the model resolves (so every wrapper installed on it takes part)."""
    import opendde.model.opendde as OM
    with torch.no_grad():
        return OM.sample_diffusion(**sampler_kwargs(b, **kw))
