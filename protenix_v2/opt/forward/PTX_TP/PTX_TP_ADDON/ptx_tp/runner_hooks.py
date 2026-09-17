# Portions derived from Protenix v2.0.0 (https://github.com/bytedance/Protenix), Copyright 2024 ByteDance and/or its affiliates,
# used under the Apache License, Version 2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ptx_tp.runner_hooks -- Protenix-specific glue installed on every rank by ptx_tp.apply_from_env() (torchrun, PTX_TP=P>1).

 1. dataloader   every rank iterates EVERY sample (DistributedSampler(num_replicas=1, rank=0)); num_workers forced to 0 (featurization in-process).
 2. features     PTX_TP_FEAT=bcast (default): rank 0 runs the STOCK InferenceDataset.__getitem__ and broadcasts (data, error_message) with
                 ptx_tp.bcast.broadcast_tensordict (receivers get CUDA tensors; atom_array stays on rank 0 -- only rank 0 dumps).
                 PTX_TP_FEAT=all: every rank featurizes and ptx_tp.bcast.assert_replicated checks equality (debug).
 3. RNG          at the start of every InferenceRunner.predict rank 0 broadcasts its python / numpy / torch-CPU / torch-CUDA RNG states; every rank then
                 executes the same RNG-consuming statements as single-card stock (MSA sampling, mc-dropout decision + mask, diffusion noise) =>
                 replicated tensors are identical on all ranks and rank 0's stream is identical to the stock CLI's.
 4. model        Protenix._main_inference_loop -> ptx_tp.trunk.main_inference_loop_tp unless Layout.replicated (N < P*128 => stock, TP no-op).
 5. outputs      DataDumper.dump is a no-op on ranks != 0; runner.dump_dir/error_dir writes happen on rank 0 only (others get a rank-suffixed error dir).
 6. guard lift   XL_LIFT_GUARD=1 (or PTX_TP_LIFT_GUARD=1): runner.inference.update_inference_configs no longer raises above 2,560 tokens for protenix-v2; the
                 item keeps the precision settings protenix-v2 runs at every size stock accepts (confidence head under AMP, diffusion sampler fp32 at EVERY N --
                 upstream's generic N > 3,840 bf16-sampler policy, written for the other checkpoints' memory, is never applied).
                 An out-of-memory above the guard is the runner's named item failure, never a lower-precision retry.
 7. lazy relp    PTX_TP_RELP=lazy and no single-card lazy relp (PTX_XL_LAZY_RELP=1) applied: RelativePositionEncoding.generate_relp stores
                 ptx_tp.trunk.TPLazyRelp (same .rows(i0,i1) API) -- the [N,N,139] fp32 one-hot is never materialised; row blocks are generated inside z_init /
                 prepare_cache by the seams.  With the single-card lazy relp active we do nothing here (its LazyRelp is consumed through the same .rows API).
All hooks print one line when installed; nothing here changes numerics."""
from __future__ import annotations

import os
import random
import sys
import time
from typing import Any, Dict

import torch
import torch.distributed as dist

import ptx_tp
from ptx_tp import bcast as BC
from ptx_tp import dist as D

_INSTALLED = False


def rng_state() -> Dict[str, Any]:
    """The python / numpy / torch (CPU + CUDA) random states of this process (rank 0's are broadcast at predict)."""
    st = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    try:
        import numpy as np
        st["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state()
    return st


def set_rng_state(st: Dict[str, Any]) -> None:
    if "python" in st:
        random.setstate(st["python"])
    if "numpy" in st:
        try:
            import numpy as np
            np.random.set_state(st["numpy"])
        except Exception:
            pass
    if "torch_cpu" in st:
        torch.set_rng_state(st["torch_cpu"].cpu() if isinstance(st["torch_cpu"], torch.Tensor) else st["torch_cpu"])
    if "torch_cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["torch_cuda"].cpu() if isinstance(st["torch_cuda"], torch.Tensor) else st["torch_cuda"])


def _rank() -> int:
    return dist.get_rank() if D.is_dist() else int(os.environ.get("RANK", "0"))


HOST_KEYS = ("token_bonds",)          # [N,N]-class features kept on HOST at large N; the trunk moves 128-row slices on demand


_LN_SEEN = set()


def _ln_autoblock(fn, x, lim, kind, _depth=0):
    """Inputs with >= lim (2^31) elements are evaluated in LEADING-DIM blocks into a preallocated output (LayerNorm is per-row =>
    identical arithmetic per row; torch's CUDA layer_norm is WRONG above 2^32 elements). Leading dims of size 1
    (e.g. stock prepare_cache's [1, N, N, c] pair LayerNorm) are descended recursively; a block that is still
    too large is split along its own next dim. PTX_TP_LN_AUTOBLOCK=0 -> raise instead (diagnostic mode)."""
    if os.environ.get("PTX_TP_LN_AUTOBLOCK", "1") != "1" or x.dim() < 2:
        raise RuntimeError(f"[ptx_tp LN guard] {kind} input {tuple(x.shape)} = {x.numel():,} elements >= {lim:,}: row-block this call site")
    if x.shape[0] == 1:                                    # descend through a size-1 leading dim
        y = _ln_autoblock(fn, x[0], lim, kind, _depth + 1) if x[0].numel() >= lim and x.dim() > 2 else fn(x[0][None])[0] if x[0].numel() < lim else None
        if y is None:
            raise RuntimeError(f"[ptx_tp LN guard] {kind} input {tuple(x.shape)}: cannot block (normalized row itself >= {lim:,} elements?)")
        return y[None]
    per_row = x.numel() // x.shape[0]
    step = max(1, (lim - 1) // per_row)
    key = (kind, tuple(x.shape))
    if key not in _LN_SEEN and _depth == 0:
        _LN_SEEN.add(key)
        print(f"[ptx_tp LN guard r{os.environ.get('RANK', '0')}] {kind} input {tuple(x.shape)} = {x.numel():,} >= {lim:,}: AUTO-BLOCKED over the leading dims (dim-0 step {step})", flush=True)
    out = None
    for s0 in range(0, x.shape[0], step):
        xb = x[s0:s0 + step]
        if xb.numel() >= lim:                               # one leading-dim slice still too large (per_row >= lim): split it along its next dim
            ys = [(_ln_autoblock(fn, xb[r], lim, kind, _depth + 1) if xb[r].numel() >= lim else fn(xb[r][None])[0]) for r in range(xb.shape[0])]
            y = torch.stack(ys, 0)
        else:
            y = fn(xb)
        if out is None:
            out = torch.empty((x.shape[0],) + tuple(y.shape[1:]), dtype=y.dtype, device=y.device)
        out[s0:s0 + step] = y
        del y
    return out
def install_layernorm_numel_guard() -> None:
    """Any CUDA layer_norm input with >= PTX_TP_LN_NUMEL_LIMIT (2^31) elements is evaluated in leading-dim blocks by _ln_autoblock (torch's
    CUDA layer_norm returns wrong tail rows above 2^32; 2^31 is the threshold used). Wraps torch.nn.functional.layer_norm (which the model's
    LayerNorm modules and the seams reach) and protenix's FusedLayerNorm.forward."""
    import torch.nn.functional as F
    lim = int(os.environ.get("PTX_TP_LN_NUMEL_LIMIT", str(2 ** 31)))
    if not getattr(F.layer_norm, "_ptx_tp_guard", False):
        _orig_ln = F.layer_norm

        def layer_norm_guarded(input, *a, **k):
            if input.is_cuda and input.numel() >= lim:
                return _ln_autoblock(lambda t: _orig_ln(t, *a, **k), input, lim, "layer_norm")
            return _orig_ln(input, *a, **k)

        layer_norm_guarded._ptx_tp_guard = True
        F.layer_norm = layer_norm_guarded
        torch.nn.functional.layer_norm = layer_norm_guarded
    try:
        from protenix.model.layer_norm.layer_norm import FusedLayerNorm
        if not getattr(FusedLayerNorm, "_ptx_tp_numel_guard", False):
            _orig_f = FusedLayerNorm.forward

            def fwd(self, x, *a, **k):
                if torch.is_tensor(x) and x.is_cuda and x.numel() >= lim:
                    return _ln_autoblock(lambda t: _orig_f(self, t, *a, **k), x, lim, "fast_layernorm")
                return _orig_f(self, x, *a, **k)

            FusedLayerNorm.forward = fwd
            FusedLayerNorm._ptx_tp_numel_guard = True
    except Exception:
        pass


def install_empty_layernorm_guard() -> None:
    """Ranks that own ZERO rows (e.g. N=2,104 / P=8: 17 row blocks -> ranks 6-7 empty) feed 0-row slices through the seams; the fused
    fast_layernorm kernel rejects numel()==0 with 'CUDA error: invalid argument'. Guard: empty input -> empty output of the same shape/dtype
    (torch's own layer_norm path is used for the dtype decision so it matches what the module returns for non-empty input of that dtype)."""
    try:
        from protenix.model.layer_norm.layer_norm import FusedLayerNorm
    except Exception:
        FusedLayerNorm = None
    import torch.nn as nn
    for cls in [c for c in (FusedLayerNorm,) if c is not None]:
        if getattr(cls, "_ptx_tp_empty_guard", False):
            continue
        orig = cls.forward

        def fwd(self, x, *a, _orig=orig, **k):
            if torch.is_tensor(x) and x.numel() == 0:
                return torch.empty_like(x)
            return _orig(self, x, *a, **k)

        cls.forward = fwd
        cls._ptx_tp_empty_guard = True


def install_host_resident_features(RI) -> None:
    """runner.inference.to_device skips HOST_KEYS when N_token >= PTX_TP_HOST_FEATS_ABOVE (default 8000): they stay on the host."""
    if getattr(RI, "_ptx_tp_to_device_patched", False):
        return
    orig = RI.to_device
    thr = int(os.environ.get("PTX_TP_HOST_FEATS_ABOVE", "8000"))

    def to_device_hostkeys(obj, device):
        fd = obj.get("input_feature_dict") if isinstance(obj, dict) else None
        stash = {}
        if isinstance(fd, dict):
            n = int(fd["token_bonds"].shape[0]) if torch.is_tensor(fd.get("token_bonds")) else 0
            if n >= thr:
                for k in HOST_KEYS:
                    if k in fd and torch.is_tensor(fd[k]):
                        stash[k] = fd.pop(k).to("cpu")
        out = orig(obj, device)
        if stash:
            (out.get("input_feature_dict") if isinstance(out, dict) else fd).update(stash)
            if int(os.environ.get("RANK", "0")) == 0:
                ptx_tp.log(f"host-resident features (N>={thr}): {[(k, tuple(v.shape), str(v.dtype)) for k, v in stash.items()]}")
        return out

    RI.to_device = to_device_hostkeys
    RI._ptx_tp_to_device_patched = True


def drop_dead_constraint_features(data) -> None:
    """PTX_TP_DROP_CONSTRAINT=1 (default in TP mode). protenix's inference featurizer ALWAYS attaches input_feature_dict['constraint_feature']
    = {contact [N,N,2] fp32, contact_atom [N,N,2] fp32, substructure [N,N,*], pocket [...]} even when the input JSON has no constraints; for
    protenix-v2 every constraint sub-embedder is disabled (ConstraintEmbedder.forward returns None) so these dense token-pair tensors (~35 GiB at
    31,140 tokens, moved to EVERY rank's GPU by the runner) are never read. They are nested, so top-level feature listings miss them. The trunk raises
    loudly if a model with an ENABLED constraint embedder runs without them (poison rule)."""
    if os.environ.get("PTX_TP_DROP_CONSTRAINT", "1") != "1":
        return
    fd = data.get("input_feature_dict", data) if isinstance(data, dict) else None
    if not isinstance(fd, dict) or "constraint_feature" not in fd:
        return
    cf = fd.pop("constraint_feature")
    fd["constraint_feature_dropped"] = torch.ones(1, dtype=torch.int64)
    try:
        tot, desc = 0, []
        def walk(x, pre=""):
            nonlocal tot
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(v, pre + k + ".")
            elif torch.is_tensor(x):
                b = x.numel() * x.element_size(); tot += b
                desc.append(f"{pre[:-1]}{tuple(x.shape)} {str(x.dtype).replace('torch.', '')} {b/2**30:.2f} GiB")
        walk(cf)
        if int(os.environ.get("RANK", "0")) == 0:
            ptx_tp.log(f"dropped dead constraint_feature (protenix-v2: all constraint embedders disabled): {tot/2**30:.2f} GiB :: " + "; ".join(desc[:8]))
    except Exception:
        pass
    del cf


def _to_host_inplace(obj, pin: bool = False):
    """Move every tensor leaf of a nested dict/list to CPU in place (receivers after the feature broadcast)."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if torch.is_tensor(v) and v.is_cuda:
                c = v.to("cpu")
                obj[k] = c.pin_memory() if pin else c
            elif isinstance(v, (dict, list)):
                _to_host_inplace(v, pin)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if torch.is_tensor(v) and v.is_cuda:
                c = v.to("cpu")
                obj[i] = c.pin_memory() if pin else c
            elif isinstance(v, (dict, list)):
                _to_host_inplace(v, pin)


def _print_top_tensors(fd, k: int = 8):
    try:
        items = [(v.numel() * v.element_size(), n, tuple(v.shape), str(v.dtype).replace("torch.", ""), str(v.device)) for n, v in fd.items() if torch.is_tensor(v)]
        items.sort(reverse=True)
        tot = sum(b for b, *_ in items) / 2**30
        desc = "; ".join(f"{n} {sh} {dt} {b/2**30:.2f} GiB" for b, n, sh, dt, dev in items[:k])
        print(f"[ptx_tp r{int(os.environ.get('RANK', '0'))}] feature tensors: total {tot:.2f} GiB (largest on {items[0][4] if items else '?'}); top-{k}: {desc}", flush=True)
    except Exception as e:
        ptx_tp.log(f"feature tensor listing failed: {e!r}")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    import runner.inference as RI
    import protenix.data.inference.infer_dataloader as IDL
    from runner.dumper import DataDumper
    from protenix.model.protenix import Protenix
    from ptx_tp import trunk as T

    install_host_resident_features(RI)
    install_empty_layernorm_guard()
    install_layernorm_numel_guard()
    P = ptx_tp.tp_size_from_env()
    feat_mode = os.environ.get("PTX_TP_FEAT", "bcast").lower()

    # -- 0. the stock runner calls dist.init_process_group("nccl") whenever WORLD_SIZE>1 (runner/inference.py init_env); the launcher has
    #       already created the group -> make a second init a no-op instead of an error.
    _orig_ipg = dist.init_process_group

    def _init_process_group(*a, **k):
        if dist.is_initialized():
            return None
        return _orig_ipg(*a, **k)

    dist.init_process_group = _init_process_group
    torch.distributed.init_process_group = _init_process_group

    # -- 0b. preprocess_input (writes <out_dir>/<name>-update-msa.json) runs on rank 0 only; the path is broadcast (no concurrent writers)
    try:
        import runner.batch_inference as BI
        _orig_pre = BI.preprocess_input

        def preprocess_input(*a, **k):
            if not D.is_dist():
                return _orig_pre(*a, **k)
            path = _orig_pre(*a, **k) if _rank() == 0 else None
            return D.broadcast_obj(path, src=0)

        BI.preprocess_input = preprocess_input
    except Exception as e:  # pragma: no cover
        ptx_tp.log(f"preprocess_input hook not installed: {e!r}")

    # -- 1. dataloader: all ranks see all samples, in-process featurization
    _orig_gidl = IDL.get_inference_dataloader

    def get_inference_dataloader(configs):
        configs.num_workers = 0
        ds = IDL.InferenceDataset(configs=configs)
        sampler = IDL.DistributedSampler(dataset=ds, num_replicas=1, rank=0, shuffle=False)
        return IDL.DataLoader(dataset=ds, batch_size=1, sampler=sampler, collate_fn=IDL.collate_fn_identity, num_workers=0)

    IDL.get_inference_dataloader = get_inference_dataloader
    RI.get_inference_dataloader = get_inference_dataloader

    # -- 2. featurization: rank-0 + broadcast (or all + assert)
    _orig_getitem = IDL.InferenceDataset.__getitem__

    def __getitem__(self, index):
        rank = _rank()
        t0 = time.time()
        if feat_mode == "all" or not D.is_dist():
            data, atom_array, err = _orig_getitem(self, index)
            compact_templates_rank0(data)
            drop_dead_constraint_features(data)
            if D.is_dist():
                BC.assert_replicated({k: v for k, v in data.items() if k != "input_feature_dict"} | {"f": data.get("input_feature_dict", {})}, "features")
            return data, atom_array, err
        if rank == 0:
            data, atom_array, err = _orig_getitem(self, index)
            compact_templates_rank0(data)
            drop_dead_constraint_features(data)
            payload = {"data": data, "err": err}
        else:
            atom_array = None
            payload = None
        payload = BC.broadcast_tensordict(payload, src=0)
        if rank != 0:
            data, err = payload["data"], payload["err"]
            if rank == 1:
                _print_top_tensors(data.get("input_feature_dict", data), k=10)
            if os.environ.get("PTX_TP_FEAT_RECV_DEVICE", "cpu") == "cpu":       # receivers hold the features on HOST exactly like rank 0
                _to_host_inplace(data, pin=os.environ.get("PTX_TP_FEAT_PIN", "0") == "1")
                del payload
                torch.cuda.empty_cache()
        if ptx_tp.det_recipe():
            BC.assert_replicated(data.get("input_feature_dict", data), "input_feature_dict")
        if rank == 0:
            n_tok = int(data["N_token"].item()) if "N_token" in data else -1
            ptx_tp.log(f"features for sample {index} ({data.get('sample_name')}, N_token={n_tok}) featurized on rank 0 and broadcast to {P} ranks in {time.time() - t0:.1f}s")
            _print_top_tensors(data.get("input_feature_dict", data), k=10)
        try:
            from ptx_tp.trunk import mem_line as _ml; _ml("after feature broadcast (features on host)")     # every rank; collective-free
        except Exception:
            pass
        return data, atom_array, err

    IDL.InferenceDataset.__getitem__ = __getitem__

    # -- 3. RNG sync at predict + phase log
    _orig_predict = RI.InferenceRunner.predict

    def predict(self, data):
        if D.is_dist():
            st = D.broadcast_obj(rng_state() if _rank() == 0 else None, src=0)
            if _rank() != 0:
                set_rng_state(st)
        return _orig_predict(self, data)

    RI.InferenceRunner.predict = predict

    # -- 4. model main loop
    _orig_loop = Protenix._main_inference_loop

    def _main_inference_loop(self, input_feature_dict, label_dict, N_cycle, mode, inplace_safe=True, chunk_size=4, symmetric_permutation=None, mc_dropout=False):
        N = input_feature_dict["residue_index"].shape[-1]
        layout = T.make_layout(N)
        if _rank() == 0:
            _mc1_report(self, N, N_cycle, mc_dropout, tp=not (layout.replicated or layout.P <= 1 or label_dict is not None), layout=layout)
        if layout.replicated or layout.P <= 1 or label_dict is not None:
            if _rank() == 0:
                ptx_tp.log(f"N_token={N} < P*B={layout.P * layout.B} (or labels given): replicated regime -> STOCK main loop on every rank")
            return _orig_loop(self, input_feature_dict=input_feature_dict, label_dict=label_dict, N_cycle=N_cycle, mode=mode, inplace_safe=inplace_safe,
                              chunk_size=chunk_size, symmetric_permutation=symmetric_permutation, mc_dropout=mc_dropout)
        return T.main_inference_loop_tp(self, input_feature_dict, N_cycle=N_cycle, mode=mode, inplace_safe=inplace_safe, chunk_size=chunk_size, mc_dropout=mc_dropout)

    Protenix._main_inference_loop = _main_inference_loop

    # -- 5. rank-0-only outputs
    _orig_dump = DataDumper.dump

    def dump(self, *a, **k):
        if _rank() != 0:
            return None
        return _orig_dump(self, *a, **k)

    DataDumper.dump = dump
    _orig_init = RI.InferenceRunner.__init__

    def __init__(self, configs):
        _orig_init(self, configs)
        if _rank() != 0:   # keep stray error files of other ranks out of the rank-0 tree
            self.error_dir = os.path.join(self.dump_dir, f"ERR_rank{_rank()}")
            os.makedirs(self.error_dir, exist_ok=True)

    RI.InferenceRunner.__init__ = __init__

    # -- 6. guard lift (optional)
    if os.environ.get("XL_LIFT_GUARD", "0") == "1" or os.environ.get("PTX_TP_LIFT_GUARD", "0") == "1":
        install_guard_lift()

    # -- 7. lazy relp fallback (optional)
    if os.environ.get("PTX_TP_RELP", "full").lower() == "lazy":
        install_lazy_relp_fallback()

    # -- 8. template features: the rank-0 featurizer never builds the dense [T,N,N,44] fp32 template pair tensors
    #       (176 B/pair/template on the HOST: 683 GB at 31k tokens x 4 slots); only per-token template arrays travel.
    install_template_rows_mode()


    # -- 9. never build the [N_atom, N_atom] int64 bond_mask (training-only feature; 473 GB at 249k atoms)
    if os.environ.get("PTX_TP_DROP_BOND_MASK", "1") == "1":
        install_bond_mask_drop()
    ptx_tp.log(f"runner hooks installed: dataloader(all-samples,num_workers=0) feat={feat_mode} rng-sync dump(rank0) main_loop(TP unless replicated)")


def install_guard_lift() -> None:
    """XL_LIFT_GUARD semantics: above 2,560 tokens protenix-v2 no longer raises; the item keeps
    protenix-v2's own precision settings at every N -- confidence head under AMP (skip_amp.confidence_head False, the stock value for protenix-v2 at every
    size) and the diffusion sampler fp32 (skip_amp.sample_diffusion True). Upstream's generic N > 3,840 policy (sampler under bf16 autocast) is for the
    other checkpoints and is never applied: the atom encoder would read the noisy positions through a bf16 Linear and the denoised coordinates would be
    rounded to bf16 every step. There is no bf16-sampler switch (a set XL_DIFF_FP32 other than 1 is refused by name); an out-of-memory above the guard
    is the runner's named item failure. UNSUPPORTED BY UPSTREAM."""
    v = os.environ.get("XL_DIFF_FP32")
    if v not in (None, "", "1"):
        raise ValueError(f"XL_DIFF_FP32={v!r}: the diffusion sampler runs fp32 at every size under the guard lift (there is no bf16-sampler setting); unset it")
    import runner.inference as RI
    if getattr(RI.update_inference_configs, "_ptx_tp_lifted", False):
        return
    _orig = RI.update_inference_configs

    def _uic(configs, n_token):
        if n_token > 2560 and configs.model_name in ["protenix-v2"]:
            configs.skip_amp.confidence_head = False
            configs.skip_amp.sample_diffusion = True          # fp32 sampler at every N (upstream's > 3,840 bf16 policy is not protenix-v2's)
            print(f"[xl_guard] upstream protenix-v2 n_token>2560 guard LIFTED at N_token={n_token} (unsupported by upstream) "
                  f"skip_amp.sample_diffusion={configs.skip_amp.sample_diffusion} (fp32 sampler)", flush=True)
            return configs
        return _orig(configs, n_token)

    _uic._ptx_tp_lifted = True
    RI.update_inference_configs = _uic
    print("[xl_guard] guard-lift installed (XL_LIFT_GUARD=1 semantics)", flush=True)


def install_lazy_relp_fallback() -> None:
    import protenix.model.modules.embedders as EM
    from ptx_tp.trunk import TPLazyRelp
    RPE = EM.RelativePositionEncoding
    if getattr(RPE.generate_relp, "__qualname__", "").endswith("[lazy]") or "LazyRelp" in getattr(RPE.generate_relp, "__doc__", "") or "":
        return
    _gen, _fwd = RPE.generate_relp, RPE.forward

    def generate_relp(self, input_feature_dict):
        if "asym_id" not in input_feature_dict:
            return _gen(self, input_feature_dict)
        cur = input_feature_dict.get("relp")
        if cur is not None and hasattr(cur, "rows"):     # a single-card LazyRelp already there
            return input_feature_dict
        input_feature_dict["relp"] = TPLazyRelp(input_feature_dict, self.r_max, self.s_max)
        return input_feature_dict

    def forward(self, relp_feature, row_block: int = 128):
        if not hasattr(relp_feature, "rows"):
            return _fwd(self, relp_feature)
        N = relp_feature.shape[-2]
        out = None
        for i0 in range(0, N, row_block):
            i1 = min(N, i0 + row_block)
            blk = self.linear_no_bias(relp_feature.rows(i0, i1))
            if out is None:
                out = torch.empty(tuple(blk.shape[:-3]) + (N, N, blk.shape[-1]), dtype=blk.dtype, device=blk.device)
            out[..., i0:i1, :, :] = blk
        return out

    generate_relp.__qualname__ = "RelativePositionEncoding.generate_relp[lazy]"
    RPE.generate_relp = generate_relp
    RPE.forward = forward
    ptx_tp.log("lazy relp fallback installed (PTX_TP_RELP=lazy; TPLazyRelp.rows per row block; dense [N,N,139] never materialised)")




def install_template_rows_mode() -> None:
    """Featurizer side (rank 0, host): for DUMMY templates (no template coordinates at all = use_template=false) Templates.as_protenix_dict
    returns only the per-token arrays (template_aatype / atom_positions / atom_mask) plus the compact keys template_pseudo_beta_mask_1d /
    template_backbone_frame_mask_1d (all-zero for dummy templates) and the marker template_pair_dense_free; for REAL templates it returns the
    per-token keys of ptx_tp.template_real (featurizer_emit_real_rows) from which every rank builds its own rows per block, bit-identical to the
    stock featuriser's dense values. Either way the dense [T,N,N,44] fp32 pair tensors are never built on the host (683 GB at 31k tokens); the
    template update is the row-sharded seam's (ptx_tp.template / the kit binding), never the stock TemplateEmbedder's dense path."""
    import numpy as np
    from protenix.data.template.template_featurizer import Templates
    if getattr(Templates.as_protenix_dict, "_ptx_tp_rows", False):
        return

    def as_protenix_dict(self):
        if np.asarray(self.atom_mask).any():          # REAL template coordinates present
            _am = np.asarray(self.atom_mask); _T = _am.shape[0]
            _n_real = int(_am.reshape(_T, -1).any(axis=1).sum())
            print(f"[ptx_tp] REAL templates loaded: {_n_real} of {_T} template slots have non-zero atom masks (tokens covered: {int(_am.any(axis=-1).sum(axis=-1).max())})", flush=True)
            from ptx_tp.template_real import featurizer_emit_real_rows, NOTICE      # per-token real-template keys; ranks rebuild rows per block (template_real)
            print(f"[ptx_tp] {NOTICE}", flush=True)
            return featurizer_emit_real_rows(self)
        d = dict(self.as_data_dict())
        T, N = np.asarray(self.aatype).shape[:2]
        d["template_pseudo_beta_mask_1d"] = np.zeros((T, N), dtype=np.float32)
        d["template_backbone_frame_mask_1d"] = np.zeros((T, N), dtype=np.float32)
        d["template_pair_dense_free"] = np.array(1, dtype=np.int64)
        return d

    as_protenix_dict._ptx_tp_rows = True
    Templates.as_protenix_dict = as_protenix_dict
    ptx_tp.log("template rows mode installed: featurizer emits per-token template arrays + compact 1-D masks only (no dense [T,N,N,44] on the host)")


def compact_templates_rank0(data: dict) -> None:
    """Before broadcasting features (rank 0): if the dense template pair tensors are present, drop them losslessly with
    ptx_tp.template.compact_template_features; logs the outcome."""
    try:
        feats = data.get("input_feature_dict") if isinstance(data, dict) else None
        if not isinstance(feats, dict) or "template_distogram" not in feats:
            return
        from ptx_tp.template import compact_template_features
        ok = compact_template_features(feats)
        ptx_tp.log(f"template features compacted before broadcast: {ok}")
    except Exception as e:
        ptx_tp.log(f"template compaction skipped: {e!r}")


def _mask_mode(N: int, mc_dropout: bool, tp: bool) -> str:
    if not mc_dropout:
        return "n/a (draw off)"
    if not tp:
        return "full-Philox (exact)"          # stock statement on the full tensor (replicated regime)
    return "full-Philox (exact)" if N <= int(os.environ.get("PTX_TP_DROPOUT_FULL_MAX", "6000")) else "block-seeded (Tier-2-by-mask)"


def _mc1_report(model, N, N_cycle, mc_dropout, tp: bool, layout=None):
    """Every run prints on rank 0 'mc-dropout draw: on|off' and, for TP, the mask mode; both go to the run metadata json."""
    draw = "on" if mc_dropout else "off"
    mask = _mask_mode(N, mc_dropout, tp)
    print(f"mc-dropout draw: {draw}" + (f" | mask: {mask}" if tp else "") +
          f"   [N_token={N} N_cycle={N_cycle} mc_dropout_apply_rate={model.configs.mc_dropout_apply_rate} mc_dropout_rate={model.configs.mc_dropout_rate}]", flush=True)
    try:
        ptx_tp.runmeta_update(kit=f"PTX_TP_ADDON v{ptx_tp.__version__}", prediction={"N_token": int(N), "N_cycle": int(N_cycle), "mc_dropout_draw": draw, "mask": mask, "tp": bool(tp),
                                          "P": (layout.P if layout is not None else 1)},
                              label=os.environ.get("PTX_TP_JOB_LABEL", "job"), seams=(ptx_tp.ledger() if tp else "stock"),
                              dropout_full_max=int(os.environ.get("PTX_TP_DROPOUT_FULL_MAX", "6000")))
    except Exception as e:
        ptx_tp.log(f"runmeta write failed: {e!r}")


def install_bond_mask_drop() -> None:
    """PTX_TP_DROP_BOND_MASK=1 (default in TP mode). The stock featurizer builds feature 'bond_mask' = [N_atom, N_atom] via np.zeros(float64) ->
    torch.Tensor(float32) -> .long() (int64): 20 B per ATOM pair transiently, 8 B resident (quadratic in N_atom: hundreds of GB at the
    largest inputs), and stock then moves it to the GPU with the other features. Its ONLY consumer
    in protenix 2.0.0 is the training bond loss (model/loss.py); inference never reads it. This patch makes Featurizer.get_mask_features skip it
    (no python-random consumption is involved -> the MC-dropout draw is unchanged)."""
    try:                                   # the standalone module (ptx_drop_bond_mask.py at the PTX_TP_ADDON root) when importable
        import ptx_drop_bond_mask
        ptx_drop_bond_mask.apply()
        return
    except ImportError:
        pass
    import numpy as np
    import protenix.data.core.featurizer as FZ
    cls = FZ.Featurizer
    if getattr(cls.get_mask_features, "_ptx_tp_nobond", False):
        return
    _orig = cls.get_mask_features
    _glpbm = FZ.get_ligand_polymer_bond_mask

    def get_mask_features(self):
        # run the stock statements with the two N_atom^2 allocations neutralised: get_ligand_polymer_bond_mask -> empty bond list and
        # np.zeros((num_atoms, num_atoms)) -> a 1x1 dummy; then drop the key.
        n = len(self.cropped_atom_array)
        _np_zeros = np.zeros

        def _zeros(shape, *a, **k):
            if isinstance(shape, tuple) and shape == (n, n) and not a and not k:
                return _np_zeros((1, 1))
            return _np_zeros(shape, *a, **k)

        FZ.get_ligand_polymer_bond_mask = lambda atom_array: np.zeros((0, 3), dtype=np.int64)
        FZ.np.zeros = _zeros
        try:
            feats = _orig(self)
        finally:
            FZ.np.zeros = _np_zeros
            FZ.get_ligand_polymer_bond_mask = _glpbm
        feats.pop("bond_mask", None)
        return feats

    get_mask_features._ptx_tp_nobond = True
    cls.get_mask_features = get_mask_features
    ptx_tp.log("bond_mask drop installed: featurizer no longer builds the [N_atom,N_atom] int64 'bond_mask' (training-loss-only feature)")
