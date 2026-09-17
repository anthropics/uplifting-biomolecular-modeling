#!/usr/bin/env python
"""Build the IO1 fixture: deterministic inputs, UPSTREAM rfdiffusion.util's own writers' bytes. Runs where torch + /opt/rfd import (the tested image; no GPU needed).
Writes <out>/pdbio_fixture.npz (arrays) + <out>/pdbio_fixture.json (cases: kwargs + upstream output text + torch-evaluated histidine flags)."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.environ.get("RFD_ROOT", "/opt/rfd"))
from rfdiffusion import util as U
from rfdiffusion.chemical import aa2long, num2aa
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
g = torch.Generator().manual_seed(20260904)
def coords(*shape):
    x = torch.randn(*shape, generator=g) * 30.0
    x.view(-1)[::17] = 0.0; x.view(-1)[5::29] = -0.0004; x.view(-1)[7::31] *= 33.0
    return x
L, T = 11, 2
seq = torch.randint(0, 21, (L,), generator=g); seq[3] = 8; seq[9] = 8
seqT = torch.randint(0, 21, (T, L), generator=g)
bf = torch.rand(L, generator=g) * 1.6 - 0.3
idx = (torch.arange(L) + 7).tolist()
chains = ["A"] * 6 + ["B"] * 5
arrays, cases = {}, []
def single(name, atoms, **kw):
    p = os.path.join(out, name + ".pdb"); U.writepdb(p, atoms, seq, **kw)
    arrays[name] = atoms.numpy()
    a = atoms.squeeze()
    his = [bool(int(s) == 8 and a.ndim == 3 and a.shape[1] in (14, 27) and torch.linalg.norm(a[i, 9, :] - a[i, 5, :]) < 1.7) for i, s in enumerate(seq.tolist())]
    kwj = {k: (v if not hasattr(v, "tolist") else v.tolist()) for k, v in kw.items()}
    arrays["text_" + name] = np.frombuffer(open(p, "rb").read(), dtype=np.uint8); cases.append(dict(kind="single", name=name, kwargs=kwj, his_d=his))
for tag, atoms in (("ca", coords(L, 3)), ("n3", coords(L, 3, 3)), ("n4", coords(L, 4, 3)), ("n14", coords(L, 14, 3)), ("n27", coords(L, 27, 3))):
    if tag == "n14":
        atoms[3, 9] = atoms[3, 5] + 0.5
    single(tag + "_plain", atoms)
    single(tag + "_binder", atoms, binderlen=6, idx_pdb=idx, bfacts=bf)
    single(tag + "_chains", atoms, binderlen=None, idx_pdb=idx, bfacts=bf, chain_idx=chains)
def multi(name, atoms, seqx, **kw):
    p = os.path.join(out, name + ".pdb"); U.writepdb_multi(p, atoms, bf, seqx, **kw)
    arrays[name] = atoms.numpy()
    arrays["text_" + name] = np.frombuffer(open(p, "rb").read(), dtype=np.uint8); cases.append(dict(kind="multi", name=name, seq="seqT" if seqx.ndim == 2 else "seq", kwargs=kw))
for natoms in (14, 27):
    atoms = coords(T, L, natoms, 3)
    atoms[:, :, 4:, :][torch.rand(T, L, natoms - 4, generator=g) < 0.5] = float("nan")
    atoms[1, 6, 2, 0] = float("nan")
    for sname, seqx in (("seq", seq), ("seqT", seqT)):
        multi(f"m{natoms}_{sname}_noh", atoms, seqx, use_hydrogens=False)
        multi(f"m{natoms}_{sname}_bb", atoms, seqx, backbone_only=True)
        multi(f"m{natoms}_{sname}_drv", atoms, seqx, use_hydrogens=False, backbone_only=False, chain_ids=chains)
        if natoms == 27:
            multi(f"m{natoms}_{sname}_h", atoms, seqx)
arrays["seq"] = seq.numpy(); arrays["seqT"] = seqT.numpy(); arrays["bf"] = bf.numpy(); arrays["idx"] = np.array(idx)
np.savez_compressed(os.path.join(out, "pdbio_fixture.npz"), **arrays)
json.dump(dict(cases=cases, chains=chains, aa2long=[list(x) for x in aa2long], num2aa=list(num2aa), N_BACKBONE_ATOMS=U.N_BACKBONE_ATOMS, N_HEAVY=U.N_HEAVY,
               torch=torch.__version__, source="rfdiffusion.util.writepdb / writepdb_multi of the pinned checkout (stock/PINS.json), run by opt/rfdiffusion1_opt/tests/fixtures/make_pdbio_fixture.py"),
          open(os.path.join(out, "pdbio_fixture.json"), "w"))
print("fixture:", len(cases), "cases,", sum(arrays["text_" + c["name"]].size for c in cases), "bytes of upstream text")
