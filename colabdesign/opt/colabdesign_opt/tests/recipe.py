"""The recipe accessor — what BindCraft's design step asks of ColabDesign, obtained by RUNNING BindCraft's own `binder_hallucination` against a
recording stand-in model (nothing is computed; every kwarg, weight, stage and iteration count comes out of BindCraft's code reading the
pinned settings file — none is typed here). Test equipment: the package's CPU tests drive the same stand-in (tests/stubsite installs
`DesignSurface` on their `_af_design`); nothing at run time imports this module.

    from colabdesign_opt.tests import recipe
    R = recipe.recipe(binder_len=80, hotspot="56", seed=0)          # tree = the package's; target = BindCraft's example PDB unless given
    R["stages"]   # [{"phase": "soft", "call": "design_logits:1", "iters": 50, "kwargs": {...}}, ..., {"phase": "greedy", "call": "design_pssm_semigreedy:1", ...}]
    R["model"], R["prep"], R["weights"], R["opt"], R["losses_added"], R["settings"], R["derived"], R["conditional"]

Phase words are units.PHASE_OF's (soft = design_logits, temp = design_soft, hard = design_hard, greedy = design_pssm_semigreedy). Needs numpy
and BindCraft's own imports (pandas, scipy, matplotlib, Bio) — the pinned stack has them; no jax computation and no GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from types import SimpleNamespace
from typing import Optional

from colabdesign_opt import units

PLDDT_DEFAULT = 0.82                       # the mean pLDDT the stand-in reports: above every BindCraft gate, so the whole recipe is walked


class DesignSurface:
    """The ColabDesign design-model surface BindCraft's design step drives (colabdesign/af/design.py `_af_design` + utils), as a recording
    stand-in: deterministic logs, a call transcript in `self.calls`, `save_pdb` writing REAL geometry (BindCraft's example PDB, chain A: the
    first target_len residues as chain A, the next binder_len as chain B, repeats shifted apart) so BindCraft's biopython checks and DSSP run."""

    PLDDT = PLDDT_DEFAULT
    EXAMPLE_PDB = None                     # set by the user of the surface (recipe(): the vendored example; the tests: the same file)

    def restart(self, seed=None, opt=None, weights=None, **kwargs):
        import numpy as np
        if seed is not None:
            self.seed = seed
        if opt:
            for k, v in opt.items():
                self.opt[k].update(v) if isinstance(v, dict) else self.opt.__setitem__(k, v)
        if weights:
            self.opt["weights"].update(weights)
        if "rm_aa" in kwargs:
            self.rm_aa = kwargs["rm_aa"]
        self._k = 0
        self._n_predict = 0
        self._tmp = {"log": [], "best": {}, "traj": {"seq": [], "xyz": [], "plddt": [], "pae": []}}
        self.aux = {"seq": {"logits": np.zeros((1, self._binder_len, 20)), "pseudo": np.zeros((1, self._binder_len, 20))}, "log": {}}
        self.calls.append(("restart", {"seed": seed, "opt": opt, "weights": weights, **kwargs}))

    def set_opt(self, *args, **kwargs):
        if "num_recycles" in kwargs:                                   # BindCraft's own set_opt (optimise_beta); the schedule's per-step set_opt is not a recipe fact
            self.calls.append(("set_opt", dict(kwargs)))
        for k, v in kwargs.items():
            if isinstance(v, dict) and isinstance(self.opt.get(k), dict):
                self.opt[k].update(v)
            else:
                self.opt[k] = v

    def set_weights(self, *args, **kwargs):
        self.opt["weights"].update(kwargs)

    def set_seq(self, seq=None, bias=None, **kwargs):
        pass

    def clear_best(self):
        self._tmp["best"] = {}

    def _h(self, *xs):
        key = json.dumps([self.seed, self._lengths, self.rm_aa, sorted((k, round(float(v), 6)) for k, v in self.opt["weights"].items()),
                          {k: self.opt[k] for k in ("con", "i_con")}, self.opt.get("hotspot_spec"), *xs], default=str, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()

    def _aux(self, k, tag="grad"):
        import numpy as np
        h = self._h(k, tag, self.opt.get("soft"), self.opt.get("temp"), self.opt.get("hard"), self.opt.get("dropout"), self.opt.get("num_recycles"))
        models = [int(h[0], 16) % (len(self._model_names) if self.opt.get("sample_models") else 1)]
        log = {"models": models, "recycles": int(self.opt.get("num_recycles") or 0), "hard": float(self.opt["hard"]), "soft": float(self.opt["soft"]), "temp": float(self.opt["temp"]),
               "loss": round(5.0 - 0.01 * (k % 1000) + int(h[1:3], 16) / 1e4, 6), "plddt": round(self.PLDDT + int(h[3:5], 16) / 1e4, 6), "ptm": 0.4,
               "i_ptm": round(0.3 + int(h[5:7], 16) / 1e3, 6), "pae": 1.0, "i_pae": 1.2, "con": 0.9, "i_con": 0.8}
        for cb in self._callbacks["model"]["loss"]:                     # the losses BindCraft added (rg / i_ptm / helix / termini): one term each, named as ColabDesign logs them
            fn = getattr(cb, "__name__", "loss")
            log[{"loss_fn": "rg", "loss_iptm": "i_ptm", "binder_helicity": "helix"}.get(fn, fn)] = round(int(h[7:9], 16) / 1e3, 6)
        n = sum(self._lengths)
        plddt = np.full(n, self.PLDDT) + (np.arange(n) % 7) * 1e-3
        return {"log": log, "plddt": plddt, "pae": np.ones((n, n)), "seq": {"logits": np.zeros((1, self._binder_len, 20)) + (k % 1000), "pseudo": np.zeros((1, self._binder_len, 20))},
                "atom_positions": np.zeros((n, 37, 3)), "num_recycles": int(self.opt.get("num_recycles") or 0), "loss": log["loss"]}

    def run(self, num_recycles=None, num_models=None, sample_models=None, models=None, backprop=True, callback=None):
        prog = (getattr(self, "_model", None) or {}).get("grad_fn" if backprop else "fn")   # where ColabDesign's _single calls the step's program (a wrapping lever counts the call)
        if callable(prog):
            prog()
        self.aux = self._aux(self._k)

    def step(self, lr_scale=1.0, num_recycles=None, num_models=None, sample_models=None, models=None, backprop=True, callback=None, save_best=False, verbose=1):
        self.run(num_recycles=num_recycles, num_models=num_models, sample_models=sample_models, models=models, backprop=backprop, callback=callback)
        self._save_results(save_best=save_best, verbose=verbose)
        self._k += 1

    def _save_results(self, aux=None, save_best=False, best_metric=None, metric_higher_better=False, verbose=True):
        if aux is None:
            aux = self.aux
        self._tmp["log"].append(aux["log"])
        if save_best:
            metric = float(aux["log"][best_metric or self._args["best_metric"]])
            if "metric" not in self._tmp["best"] or metric < self._tmp["best"]["metric"]:
                self._tmp["best"] = {"aux": aux, "metric": metric}

    def predict(self, seq=None, bias=None, num_models=None, num_recycles=None, models=None, sample_models=False, dropout=False, hard=True, soft=False, temp=1,
                return_aux=False, verbose=True, seed=None, **kwargs):
        self._n_predict = getattr(self, "_n_predict", 0) + 1
        aux = self._aux(self._k * 1000 + self._n_predict, "forward")
        self.aux = aux
        return aux if return_aux else None

    def design(self, iters=100, soft=0.0, e_soft=None, temp=1.0, e_temp=None, hard=0.0, e_hard=None, step=1.0, e_step=None, dropout=True, opt=None, weights=None,
               num_recycles=None, ramp_recycles=True, num_models=None, sample_models=None, models=None, backprop=True, callback=None, save_best=False, verbose=1):
        self.set_opt(soft=soft, hard=hard, temp=temp, dropout=dropout)
        if e_soft is None:
            e_soft = soft
        if e_temp is None:
            e_temp = temp
        for i in range(iters):
            self.set_opt(soft=(soft + (e_soft - soft) * ((i + 1) / iters)), hard=hard, temp=(e_temp + (temp - e_temp) * (1 - (i + 1) / iters) ** 2))
            self.step(num_recycles=num_recycles, num_models=num_models, sample_models=sample_models, models=models, backprop=backprop, callback=callback, save_best=save_best, verbose=verbose)

    def design_logits(self, iters=100, **kwargs):
        self.calls.append(("design_logits", {"iters": iters, **kwargs}))
        self.design(iters, **kwargs)

    def design_soft(self, iters=100, temp=1, **kwargs):
        self.calls.append(("design_soft", {"iters": iters, "temp": temp, **kwargs}))
        self.design(iters, soft=1, temp=temp, **kwargs)

    def design_hard(self, iters=100, **kwargs):
        self.calls.append(("design_hard", {"iters": iters, **kwargs}))
        self.design(iters, soft=1, hard=1, **kwargs)

    def design_semigreedy(self, iters=100, tries=10, dropout=False, save_best=True, seq_logits=None, e_tries=None, **kwargs):
        import numpy as np
        if e_tries is None:
            e_tries = tries
        for k in ("num_models", "sample_models", "models"):
            kwargs.pop(k, None)
        verbose = kwargs.pop("verbose", 1)
        self.set_opt(hard=1.0, soft=0.0, temp=1.0, dropout=dropout)
        self.predict(None, return_aux=True, verbose=False, **kwargs)
        for i in range(iters):
            buff = []
            num_tries = (tries + (e_tries - tries) * ((i + 1) / iters))
            for t in range(int(num_tries)):
                buff.append(self.predict(seq=t, return_aux=True, verbose=False, **kwargs))
            losses = [x["loss"] for x in buff]
            self.aux = buff[int(np.argmin(losses))]
            self._k += 1
            self._save_results(save_best=save_best, verbose=verbose)

    def design_pssm_semigreedy(self, soft_iters=300, hard_iters=32, tries=10, e_tries=None, ramp_recycles=True, ramp_models=True, **kwargs):
        self.calls.append(("design_pssm_semigreedy", {"soft_iters": soft_iters, "hard_iters": hard_iters, "tries": tries, "e_tries": e_tries, "ramp_recycles": ramp_recycles,
                                                      "ramp_models": ramp_models, **kwargs}))
        if soft_iters > 0:
            self.design_3stage(soft_iters, 0, 0, ramp_recycles=ramp_recycles, **kwargs)
        self.design_semigreedy(hard_iters, tries=tries, e_tries=e_tries, **kwargs)

    def design_3stage(self, soft_iters=300, temp_iters=100, hard_iters=10, ramp_recycles=True, **kwargs):
        self.calls.append(("design_3stage", {"soft_iters": soft_iters, "temp_iters": temp_iters, "hard_iters": hard_iters}))
        self.design_logits(soft_iters, e_soft=1, ramp_recycles=ramp_recycles, **kwargs)
        self.design_soft(temp_iters, e_temp=1e-2, **kwargs)
        kw = {k: v for k, v in kwargs.items() if k not in ("dropout", "save_best")}
        self.design_hard(hard_iters, temp=1e-2, dropout=False, save_best=True, **kw)

    def get_seqs(self, get_best=True):
        h = hashlib.sha256(f"{self.seed}:{self._lengths}:{get_best}:{len(self._tmp['log'])}".encode()).hexdigest()
        aa = "ACDEFGHIKLMNPQRSTVWY"
        return ["".join(aa[int(c, 16) % 20] for c in (h * 8)[: self._binder_len])]

    def get_loss(self, x="loss"):
        import numpy as np
        return np.array([log.get(x, np.nan) for log in self._tmp["log"]])

    def animate(self, dpi=100, **kwargs):
        return "<html>stand-in animation</html>"

    def save_pdb(self, filename=None, get_best=True):
        src = self.EXAMPLE_PDB
        assert src and os.path.isfile(src), "DesignSurface.EXAMPLE_PDB must point at a real PDB"
        res_order, atoms = [], {}
        with open(src) as fh:
            for line in fh:
                if line.startswith("ATOM") and line[21] == "A":
                    key = line[22:27]
                    if key not in atoms:
                        atoms[key] = []
                        res_order.append(key)
                    atoms[key].append(line)
        need, n_src = self._target_len + self._binder_len, len(res_order)     # longer cases reuse the residues, each repeat shifted 200 A along x (real local geometry, far apart)
        out, serial = [], 1
        for i in range(need):
            key, shift = res_order[i % n_src], 200.0 * (i // n_src)
            ch, rnum = ("A", i + 1) if i < self._target_len else ("B", i - self._target_len + 1)
            for line in atoms[key]:
                line = line.rstrip("\n").ljust(80)
                x = float(line[30:38]) + shift
                out.append(f"{line[:6]}{serial:5d}{line[11:21]}{ch}{rnum:4d} {line[27:30]}{x:8.3f}{line[38:60]}{self.PLDDT * 100:6.2f}{line[66:]}\n")
                serial += 1
            if i == self._target_len - 1:
                out.append("TER\n")
        body = f"REMARK stand-in structure seed {self.seed} lengths {self._lengths} steps {len(self._tmp['log'])}\n" + "".join(out) + "TER\nEND\n"
        if filename is None:
            return body
        with open(filename, "w") as fh:
            fh.write(body)


class ModelSurface:
    """Construction and inputs of the stand-in (`mk_afdesign_model(...)`, `prep_inputs(...)`), recording their kwargs. `prep_inputs` ends in
    `self._prep_model(...)` — ColabDesign's contract (`_af_prep._prep_model`: build the executables, then restart); the class that mixes this in
    provides it (Recorder: below; the tests' stand-in: their `_af_prep`, which the nosub lever replaces)."""

    def __init__(self, protocol="fixbb", use_multimer=False, data_dir=".", num_recycles=0, best_metric="loss", debug=False, **kw):
        self.protocol = protocol
        self._args = {"use_multimer": use_multimer, "best_metric": best_metric, "debug": debug, "use_templates": protocol == "binder", "traj_iter": 1, "traj_max": 10000}
        self._args.update(kw)
        self.opt = {"num_models": 1, "sample_models": True, "num_recycles": num_recycles, "soft": 0.0, "temp": 1.0, "hard": 0.0, "dropout": True, "learning_rate": 0.1,
                    "weights": {"plddt": 0.1, "con": 0.0, "i_con": 1.0, "i_pae": 0.0, "pae": 0.0, "seq_ent": 0.0, "helix": 0.0},
                    "con": {"num": 2, "cutoff": 14.0, "binary": False, "seqsep": 9}, "i_con": {"num": 1, "cutoff": 21.6875, "binary": False}}
        self._cfg = SimpleNamespace(model=SimpleNamespace(global_config=SimpleNamespace(subbatch_size=None)))
        self._model_names = [f"model_{i}_multimer_v3" for i in (1, 2, 3, 4, 5)] if use_multimer else [f"model_{i}_ptm" for i in (1, 2, 3, 4, 5)]
        self._callbacks = {"model": {"pre": [], "post": [], "loss": []}, "design": {"pre": [], "post": []}}
        self._tmp = {"log": [], "best": {}, "traj": {"seq": [], "xyz": [], "plddt": [], "pae": []}}
        self._k = 0
        self.calls = [("mk_afdesign_model", {"protocol": protocol, "use_multimer": use_multimer, "num_recycles": num_recycles, "best_metric": best_metric, "debug": debug, "data_dir": data_dir, **kw})]
        self.data_dir = data_dir

    def _get_model(self, cfg):
        return {"runner": SimpleNamespace(config=cfg), "fn": (lambda *a, **k: None), "grad_fn": (lambda *a, **k: None)}   # callables: a lever that wraps the two programs (lowercache) sees them called by run()

    def prep_inputs(self, pdb_filename, chain, binder_len, **kwargs):
        chains = [c.strip() for c in chain.split(",")]
        seen = []
        with open(pdb_filename) as fh:
            for line in fh:
                if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] in chains and (line[21], line[22:27]) not in seen:
                    seen.append((line[21], line[22:27]))
        self._target_len, self._binder_len = len(seen), int(binder_len)
        self._lengths = [self._target_len, self._binder_len]
        self._len = self._binder_len
        self.pdb_sha = hashlib.sha256(open(pdb_filename, "rb").read()).hexdigest()
        self.seed = kwargs.get("seed")
        self.rm_aa = kwargs.get("rm_aa")
        self.opt["hotspot_spec"] = kwargs.get("hotspot")
        self.calls.append(("prep_inputs", {"pdb_filename": pdb_filename, "chain": chain, "binder_len": int(binder_len), **kwargs}))
        self._prep_model(**{k: v for k, v in kwargs.items() if k in ("seed", "rm_aa")})


class Recorder(ModelSurface, DesignSurface):
    """The recording stand-in model recipe() hands to BindCraft's binder_hallucination."""

    def _prep_model(self, **kwargs):
        self._cfg.model.global_config.subbatch_size = 4 if sum(self._lengths) > 384 else None    # stock's rule (colabdesign/af/prep.py), for the record only
        self._model = self._get_model(self._cfg)
        self.restart(**kwargs)


def install_design_methods(cls) -> None:
    """Define DesignSurface's methods on `cls` (the tests' stand-in `_af_design`, where ColabDesign defines them, so the observer wraps the same class)."""
    for name, fn in DesignSurface.__dict__.items():
        if not name.startswith("__"):                                  # the methods and the two knobs (PLDDT, EXAMPLE_PDB)
            setattr(cls, name, fn)


def recipe(tree: Optional[str] = None, *, target: Optional[str] = None, chain: str = "A", binder_len: int, hotspot: Optional[str] = None, seed: int = 0) -> dict:
    """Run BindCraft's binder_hallucination on the Recorder and return what it asked of ColabDesign (see the module docstring for the shape)."""
    from colabdesign_opt import bindcraft, settings, stack
    tree = tree or stack.tree_home()
    F = bindcraft.functions(tree)
    S = settings.load(tree)
    target = target or os.path.join(bindcraft.bindcraft_dir(tree), "example", "PDL1.pdb")
    tmp = tempfile.mkdtemp(prefix="cd_opt_recipe_")
    params = os.path.join(tmp, "params")
    os.makedirs(os.path.join(params, "params"))
    D = bindcraft.derive(F, S, tree=tree, out_dir=tmp, params_dir=params, binder_len=binder_len, seed=seed, tag="recipe")
    for n in D["design_models"]:
        open(bindcraft.params_file(params, n, bool(D["advanced"].get("use_multimer_design"))), "a").close()
    cu = F.colabdesign_utils

    class _Recorder(Recorder):
        EXAMPLE_PDB = os.path.join(bindcraft.bindcraft_dir(tree), "example", "PDL1.pdb")

    saved = {k: getattr(cu, k) for k in ("mk_afdesign_model", "clear_mem")}
    cu.mk_afdesign_model, cu.clear_mem = _Recorder, (lambda: None)
    try:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            af = cu.binder_hallucination(D["design_name"], target, chain, hotspot or "", int(binder_len), int(seed), D["helicity_value"], D["design_models"], D["advanced"], D["design_paths"], D["failure_csv"])
    finally:
        for k, v in saved.items():
            setattr(cu, k, v)
    calls = af.calls
    by = {}
    stages, ordinal = [], {}
    for name, kw in calls:
        if name in ("mk_afdesign_model", "prep_inputs", "restart"):
            by.setdefault(name, kw)
        elif name in units.PHASE_OF and name != "design_semigreedy":
            ordinal[name] = ordinal.get(name, 0) + 1
            kw = dict(kw)
            iters = kw.pop("hard_iters") if name == "design_pssm_semigreedy" else kw.pop("iters")
            stages.append({"phase": units.PHASE_OF[name], "call": f"{name}:{ordinal[name]}", "iters": int(iters), "kwargs": kw})
    return {"settings": {**settings.record(S), "advanced": S["advanced"]},
            "derived": {"design_models": D["design_models"], "helicity_value": D["helicity_value"], "design_name": D["design_name"], "terminate": (af.aux.get("log") or {}).get("terminate", "")},
            "model": by.get("mk_afdesign_model", {}), "prep": by.get("prep_inputs", {}), "weights": dict(af.opt["weights"]), "opt": {"con": dict(af.opt["con"]), "i_con": dict(af.opt["i_con"])},
            "losses_added": [getattr(cb, "__name__", "?") for cb in af._callbacks["model"]["loss"]], "stages": stages,
            "conditional": {"set_opt": [kw for name, kw in calls if name == "set_opt"]}, "calls": calls}
