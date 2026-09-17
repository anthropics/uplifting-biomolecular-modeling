"""protenix_fpf_ditfast.vectors — byte test vectors of the fused stacks.

The levers of this unit are TOLERANCE-class against stock (GEMM shapes differ), but the unit's OWN arithmetic is deterministic launch to launch:
each case builds a seeded DiffusionTransformer (24 token blocks, c_a=768/c_s=384/c_z=256, 16 heads) or AtomTransformer (3 blocks, c_atom=128,
c_atompair=16, 4 heads) with every parameter drawn from a CPU generator, runs the fused stack on seeded inputs, and digests the output BYTES.
`vectors.json` holds the digests recorded on the pinned image + GPU class; `run_all()` recomputes them (used by the release tests and by
downstream kernel selection; the kit hook does not call it).  Needs CUDA + protenix_fpf_apb (the attention levers) like the levers themselves.
"""
import os, json, hashlib
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _seeded_(module, g, scale=0.05):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_((torch.randn(p.shape, generator=g) * scale).to(p.device, p.dtype))
    return module


def token_case(seed, S, N, word):
    """FastTokenStack over a seeded 24-block DiffusionTransformer; a [S,N,768], s [1 or S,N,384], z [1,N,N,256] fp32 seeded; word = off|bf16|fp16."""
    from protenix.model.modules.transformer import DiffusionTransformer
    from .dit_fast import FastTokenStack
    g = torch.Generator(device="cpu").manual_seed(seed)
    dt = _seeded_(DiffusionTransformer(n_blocks=24, n_heads=16, c_a=768, c_s=384, c_z=256).cuda().eval(), g)

    class _DM:                       # the attributes FastTokenStack reads from the diffusion module
        diffusion_transformer = dt
    act = {"off": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[word]
    st = FastTokenStack(_DM, act)
    a = (torch.randn(S, N, 768, generator=g) * 1.0).cuda(); s = (torch.randn(1, N, 384, generator=g) * 1.0).cuda(); z = (torch.randn(1, N, N, 256, generator=g) * 1.0).cuda()
    with torch.no_grad():
        out = st.forward(a, s, z)
    torch.cuda.synchronize()
    return out.float().contiguous()


def atom_case(seed, S, NA):
    """FastAtomStack over a seeded 3-block AtomTransformer; q [S,NA,128], c [1,NA,128], p [1,nb,32,128,16] fp32 seeded."""
    from protenix.model.modules.transformer import AtomTransformer
    from .atom_fast import FastAtomStack
    g = torch.Generator(device="cpu").manual_seed(seed)
    at = _seeded_(AtomTransformer(n_blocks=3, n_heads=4, c_atom=128, c_atompair=16).cuda().eval(), g)
    st = FastAtomStack("vec", at, torch.float32)
    nb = (NA + 31) // 32
    q = torch.randn(S, NA, 128, generator=g).cuda(); c = torch.randn(1, NA, 128, generator=g).cuda(); p = torch.randn(1, nb, 32, 128, 16, generator=g).cuda()
    with torch.no_grad():
        out = st.forward(q, c, p)
    torch.cuda.synchronize()
    return out.float().contiguous()


def digest(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def run_case(case):
    if case["kind"] == "token":
        out = token_case(case["seed"], case["S"], case["N"], case["word"])
    else:
        out = atom_case(case["seed"], case["S"], case["NA"])
    return digest(out), list(out.shape), bool(torch.isfinite(out).all())


CASES = [
    {"name": "tok_s2_n64_off", "kind": "token", "seed": 21, "S": 2, "N": 64, "word": "off"},
    {"name": "tok_s2_n64_fp16", "kind": "token", "seed": 21, "S": 2, "N": 64, "word": "fp16"},
    {"name": "tok_s2_n64_bf16", "kind": "token", "seed": 21, "S": 2, "N": 64, "word": "bf16"},
    {"name": "tok_s5_n200_fp16", "kind": "token", "seed": 22, "S": 5, "N": 200, "word": "fp16"},
    {"name": "atom_s2_n200", "kind": "atom", "seed": 23, "S": 2, "NA": 200},
    {"name": "atom_s5_n333", "kind": "atom", "seed": 24, "S": 5, "NA": 333},
]


def run_all(path=None):
    spec = json.load(open(path or os.path.join(_HERE, "vectors.json")))
    results = []
    for case in spec["cases"]:
        got, shape, finite = run_case(case)
        results.append({"name": case["name"], "ok": got == case["sha256"], "got": got, "want": case["sha256"], "shape": shape, "finite": finite})
    return results


def record(path=None):
    """Write vectors.json from CASES on this machine (release tooling; run on the pinned image + GPU class)."""
    out = {"about": "byte test vectors of protenix_fpf_ditfast's fused stacks on seeded modules/inputs (see vectors.py); sha256 of the fp32 output bytes",
           "recorded_with": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "cc": list(torch.cuda.get_device_capability())}, "cases": []}
    for case in CASES:
        got, shape, finite = run_case(case)
        twice, _, _ = run_case(case)
        out["cases"].append(dict(case, sha256=got, shape=shape, finite=finite, repeatable=(got == twice)))
    json.dump(out, open(path or os.path.join(_HERE, "vectors.json"), "w"), indent=1)
    return out
