"""Shared fixture for the k/test_ef2_{lazy_structure,bf16_confidence}.py unit tests: a SMALL random-initialised
ESMFold2-Experimental model on CUDA (2 trunk blocks, 2 diffusion token blocks, 1 confidence block; no ESMC, no checkpoint download) and
design-style inputs for it (a two-chain protein complex featurised by the fork's own `prepare_protein_features`, plus a soft sequence
`res_type_soft` that requires grad). Not a lever; imported by the tests only."""
import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def small_model(seed=0, confidence=True):
    from transformers.models.esmfold2.configuration_esmfold2 import (ESMFold2Config, FoldingTrunkConfig, DiffusionStructureHeadConfig,
                                                                      DiffusionModuleConfig, ConfidenceHeadConfig)
    from transformers.models.esmfold2.modeling_esmfold2_experimental import ESMFold2ExperimentalModel
    torch.manual_seed(seed)
    cfg = ESMFold2Config(type="experimental", num_loops=1, num_diffusion_samples=1,
                         folding_trunk=FoldingTrunkConfig(n_layers=2),
                         structure_head=DiffusionStructureHeadConfig(diffusion_module=DiffusionModuleConfig(token_num_blocks=2, atom_num_blocks=1)),
                         confidence_head=ConfidenceHeadConfig(enabled=confidence, folding_trunk=FoldingTrunkConfig(n_layers=1)))
    model = ESMFold2ExperimentalModel(cfg)
    for m in model.modules():                                  # non-trivial LayerNorm affine parameters so hoists/replays are exercised
        if isinstance(m, torch.nn.LayerNorm) and m.elementwise_affine:
            torch.nn.init.normal_(m.weight, 1.0, 0.1); torch.nn.init.normal_(m.bias, 0.0, 0.1)
    torch.nn.init.normal_(model.pair_loop_proj[1].weight, 0.0, 0.02)   # stock zero-inits it; make the recycle projection live
    return model.to(dev).eval().requires_grad_(False)


def design_inputs(target="MKTAYIAKQRQISFVKSHFSRQ", binder_len=12, seed=0):
    """Two chains (target | binder) featurised like a design step: the binder residues are alanine placeholders in the atom features and
    a soft distribution in `res_type_soft` (requires grad)."""
    from transformers.models.esmfold2.protein_utils import prepare_protein_features
    from transformers.models.esmfold2.modeling_esmfold2_common import NUM_RES_TYPES
    seq = target + "A" * binder_len
    f = prepare_protein_features(seq)
    if f["res_type"].dim() == 1:                               # tolerate an unbatched featuriser
        f = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in f.items()}
    L = len(seq)
    asym = torch.zeros(1, L, dtype=torch.int64); asym[:, len(target):] = 1
    ent = torch.ones(1, L, dtype=torch.int64); ent[:, len(target):] = 2
    f["asym_id"], f["entity_id"], f["sym_id"] = asym, ent, torch.zeros(1, L, dtype=torch.int64)
    f = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in f.items()}
    f.pop("input_ids", None)                                   # no ESMC in the fixture
    g = torch.Generator(device="cpu").manual_seed(seed)
    hard = F.one_hot(f["res_type"].long(), num_classes=NUM_RES_TYPES).float()
    soft = torch.softmax(torch.randn(1, binder_len, NUM_RES_TYPES, generator=g) * 2.0, dim=-1).to(dev)
    res_type_soft = torch.cat([hard[:, : len(target)], soft], dim=1).requires_grad_(True)
    f["res_type_soft"] = res_type_soft
    return f


def run_design_call(model, feats, seed=0, num_sampling_steps=1, num_loops=1, **kw):
    """One design-style call, the way the cookbook makes it (the ambient RNG pinned around the model call by a seed context, the
    sampler seeded by `seed=`): distogram under grad, backward to res_type_soft; returns (output, distogram fp32 clone, grad clone)."""
    from transformers.models.esmfold2.modeling_esmfold2_common import _seed_context
    x = feats["res_type_soft"]
    if x.grad is not None:
        x.grad = None
    with _seed_context(seed):
        out = model(**feats, num_diffusion_samples=1, num_sampling_steps=num_sampling_steps, num_loops=num_loops, seed=seed, **kw)
    d = out["distogram_logits"]
    (d.float().square().mean()).backward()
    return out, d.detach().float().clone(), x.grad.detach().clone()


def assert_floor(model, feats, **kw):
    """The fixture must reproduce itself bit for bit (two identical stock calls) before a lever is judged against it."""
    o1, d1, g1 = run_design_call(model, feats, **kw); o2, d2, g2 = run_design_call(model, feats, **kw)
    if not (torch.equal(d1, d2) and torch.equal(g1, g2) and torch.equal(o1["sample_atom_coords"], o2["sample_atom_coords"])):
        try:                                                   # the kit's det recipe (segment-mean scatter, deterministic attention) when the package is importable
            from ef2inv_opt import det as D
            D.apply(D.level(1))
        except Exception:                                      # noqa: BLE001
            torch.use_deterministic_algorithms(True, warn_only=True)
        o1, d1, g1 = run_design_call(model, feats, **kw); o2, d2, g2 = run_design_call(model, feats, **kw)
        assert torch.equal(d1, d2) and torch.equal(g1, g2) and torch.equal(o1["sample_atom_coords"], o2["sample_atom_coords"]), \
            "the stock fixture is not run-to-run bitwise on this machine even under the det recipe; exactness cannot be judged here"
    return o1, d1, g1
