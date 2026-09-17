"""The ``confhoist`` lever (exact-class): the confidence head's sample-invariant prologue (``rf3.model.layers.af3_auxiliary_heads.
ConfidenceHead.forward``, af3_auxiliary_heads.py:97-127 at the pin — the fp32 casts of the trunk's ``S_trunk_I`` / ``Z_trunk_II`` /
``S_inputs_I``, their three whole-tensor layer norms, the two ``process_s_inputs_*`` projections and the outer-sum add into the pair track)
computed ONCE per prediction and reused by every diffusion sample's confidence call, instead of once per sample. Stock calls the head D
times per item (``RF3WithConfidence.forward``'s per-sample loop, D = diffusion_batch_size, 5 shipped) with the SAME three trunk tensors and a
different ``X_pred_L``; everything before the distance embedding does not read ``X_pred_L``. The hoisted statements are upstream's,
verbatim, executed once: the same kernels on the same operands in the same order, so every sample's logits are bit-identical to stock
(the lever's numerics class is exact).

Cache discipline. The prologue's outputs (``S_trunk`` after its layer norm, fp32 [1, I, 384]; ``Z`` after its layer norm plus the outer
sum, fp32 [1, I, I, 128]) are kept under the identity of the call's operands — the head instance and the three input tensor OBJECTS
(``is``) plus their in-place version counters — so a call with other tensors (the next item, a probe) recomputes, never reads a stale
entry; ``S_trunk`` is handed to the pairformer as a fresh ``clone()`` per sample (1.8 MB at 1200 tokens; the cached tensor is never an
operand of an in-place kernel), ``Z`` is consumed by upstream's out-of-place ``Z + process_pred_distances(..)`` first, so the cached pair
tensor is read-only by construction. The entry is dropped when ``RF3WithConfidence.forward`` returns (a wrapper installed on the model
class: the cache never outlives the item, so the next item's trunk and sampler see no held pair tensor) and whenever the operands change.
Memory: one extra fp32 pair tensor (I^2 x 512 B: 2.0 GB at 2000 tokens) is alive during the confidence phase only.

Upstream drift. The per-sample remainder is upstream's statements af3_auxiliary_heads.py:99-102 and 129-218, carried here verbatim; the
lever engages only when the running class's ``forward`` source still hashes to the pin's (``FORWARD_NORM_SHA256``, whitespace-normalised)
— another ``foundry`` refuses it BY NAME (``refused:upstream_bytes:<sha12>``, :func:`problems`), never a silent replacement of code this
module has not read. No CUDA device / the class absent are named the same way. CPU helper processes (no CUDA) are a named no-op like pf.py's.

The companion lever ``confln`` (fast-class, numerics-changing; needs ``confhoist``): the prologue's three WHOLE-TENSOR layer norms
(``F.layer_norm(x, normalized_shape=x.shape)``: no affine, one mean / variance over every element — stock runs them as ONE row of the
row-wise LayerNorm kernel, a single thread block reducing I^2 x 128 elements, which dominates the head's
time) computed as ``(x - mean) * rsqrt(var + eps)`` from ``torch.var_mean`` (a multi-block reduction) — the same formula in fp32, a
different summation order (not bitwise; fast-class numerics), a multi-block kernel instead of a single-block one.

Surface: :func:`arm` (called once at activation with the kit's import-watch installer: patches the class when its module executes, wraps
the model's forward when ``rf3.model.RF3`` executes — or at once when they are imported already), :func:`enable` / :func:`disable`,
:func:`describe` (state + census for the exit tally), :func:`problems` (the pred verdict's failures), :func:`lever_tokens` (the LEVER line).
"""
import hashlib
import inspect
import sys
import threading
from typing import Callable, Dict, List, Optional

NAME = "confhoist"
LN_NAME = "confln"                                        # the companion fast-class lever: the prologue's whole-tensor layer norms by torch.var_mean (needs confhoist)
LN_EPS = 1e-5                                             # F.layer_norm's default eps (upstream passes none)
LN_REQUIRES = f"{LN_NAME} rides {NAME}'s prologue: name {NAME} with it (the whole-tensor layer norms it replaces are the hoisted statements)"
HEADS_MODULE = "rf3.model.layers.af3_auxiliary_heads"      # ConfidenceHead lives here (patched when this module executes)
MODEL_MODULE = "rf3.model.RF3"                            # RF3WithConfidence lives here (its forward is wrapped to drop the cache per item)
FORWARD_NORM_SHA256 = "cfe7bf1001fad48e7aa57964c15745f58aaeb0b6d97fe839673e376a91d26dc9"   # sha256 of ConfidenceHead.forward's source at foundry 4010e3e2e,
                                                          # each line stripped, blank lines dropped, joined by \n (af3_auxiliary_heads.py:86-218)
CPU_PROCESS = "no CUDA device in this process: the confidence head never runs here (a CPU helper process); the hoist is not installed"
CONFLICT_N_GPU = ("the confidence head runs as the row-sharded statement under n_gpu > 1 (rowpair: tp_conf.run_confidence_sharded owns ConfidenceHead.forward); "
                  "the prologue hoist is a single-GPU lever and steps aside by name")

STATE: Dict[str, object] = {"on": False, "armed": False, "installed": False, "model_wrapped": False, "reason": None, "refusal": None,
                            "calls": 0, "prologues": 0, "reused": 0, "cleared": 0, "stale": 0, "passthrough": 0, "upstream_sha12": None,
                            "conflict": None, "fastln": False, "fastln_calls": 0, "fastln_refusal": None}
_PREV: Dict[str, object] = {"forward": None, "cls": None, "model_forward": None, "model_cls": None}
_CACHE: Dict[str, object] = {"entry": None}
_LOCK = threading.Lock()


# ------------------------------------------------------------------------------------------------------------ source pin
def norm_sha256(src: str) -> str:
    """sha256 of a source block with each line stripped and blank lines dropped (indentation / trailing-space / blank-line neutral)."""
    return hashlib.sha256("\n".join(l.strip() for l in src.splitlines() if l.strip()).encode()).hexdigest()


def upstream_digest(cls) -> Optional[str]:
    """The normalised digest of ``cls.forward``'s source, or None when it has no readable source (derived / replaced in this process)."""
    fn = getattr(cls, "forward", None)
    for _ in range(8):                                                        # a report-only wrapper (functools convention) or this module's own forward: look through it
        nxt = getattr(fn, "__wrapped_confhoist__", None) or getattr(fn, "__wrapped__", None)
        if nxt is None:
            break
        fn = nxt
    try:
        return norm_sha256(inspect.getsource(fn))
    except (OSError, TypeError):
        return None


# ------------------------------------------------------------------------------------------------------------ the hoisted forward
def _versions(ts) -> tuple:
    return tuple(int(getattr(t, "_version", 0)) for t in ts)


def _prologue(self, S_inputs_I, S_trunk_I, Z_trunk_II):
    """af3_auxiliary_heads.py:97-98, 101, 104-127 verbatim (the statements that do not read X_pred_L / seq / rep_atoms)."""
    import torch.nn.functional as F
    # stopgrad on S_trunk_I, Z_trunk_II, X_pred_L but not S_inputs_I (4.3.5)
    S_trunk_I = S_trunk_I.detach().float()  # B, L, 384
    Z_trunk_II = Z_trunk_II.detach().float()  # B, L, L, 128
    S_inputs_I = S_inputs_I.detach().float()  # B, L, 384

    if self.layer_norm_along_feature_dimension:
        # do a layer norm on S_trunk_I
        S_trunk_I = F.layer_norm(S_trunk_I, normalized_shape=(S_trunk_I.shape[-1]))
        # do a layer norm on Z_trunk_II
        Z_trunk_II = F.layer_norm(
            Z_trunk_II, normalized_shape=(Z_trunk_II.shape[-1])
        )
        # do a layer norm on S_inputs_I
        S_inputs_I = F.layer_norm(
            S_inputs_I, normalized_shape=(S_inputs_I.shape[-1])
        )
    elif STATE["fastln"]:                                                     # lever confln: the same three whole-tensor normalisations by a multi-block reduction (_whole_layer_norm)
        S_trunk_I = _whole_layer_norm(S_trunk_I)
        Z_trunk_II = _whole_layer_norm(Z_trunk_II)
        S_inputs_I = _whole_layer_norm(S_inputs_I)
    else:
        S_trunk_I = F.layer_norm(S_trunk_I, normalized_shape=(S_trunk_I.shape))
        Z_trunk_II = F.layer_norm(Z_trunk_II, normalized_shape=(Z_trunk_II.shape))
        S_inputs_I = F.layer_norm(S_inputs_I, normalized_shape=(S_inputs_I.shape))

    # embed S_inputs_I twice
    S_inputs_I_right = self.process_s_inputs_right(S_inputs_I)
    S_inputs_I_left = self.process_s_inputs_left(S_inputs_I)
    # add outer product of two linear embeddings of S_inputs_I  to Z_II
    # TODO: check the unsqueezed dimension is the correct one
    Z_trunk_II = Z_trunk_II + (
        S_inputs_I_right.unsqueeze(-2) + S_inputs_I_left.unsqueeze(-3)
    )
    return S_trunk_I, Z_trunk_II


def _whole_layer_norm(x, eps: float = LN_EPS):
    """``F.layer_norm(x, normalized_shape=x.shape)`` (no weight / bias: one mean and one biased variance over EVERY element) as
    ``(x - mean) * rsqrt(var + eps)`` with ``torch.var_mean(x, correction=0)`` — fp32 in, fp32 out, a grid-wide reduction instead of the
    row-wise kernel's single block. Counted (``fastln_calls``)."""
    import torch
    var, mean = torch.var_mean(x, correction=0)
    STATE["fastln_calls"] = int(STATE["fastln_calls"]) + 1
    return (x - mean) * torch.rsqrt(var + eps)


def _tail(self, S_trunk_I, Z_trunk_II, X_pred_L, seq, rep_atoms, frame_atom_idxs):
    """af3_auxiliary_heads.py:99-100, 102, 129-218 verbatim (everything from the distance embedding on), on the prologue's outputs."""
    import torch
    import torch.nn.functional as F
    M = sys.modules[HEADS_MODULE]
    discretize_distance_matrix, calc_Cb_distances = M.discretize_distance_matrix, M.calc_Cb_distances
    if X_pred_L is not None:
        X_pred_L = X_pred_L.detach().float()  # B, n_atoms, 3
    seq = seq.detach()

    # embed distances of representative atom from every token
    #    in the pair representation
    # if no coords are input, skip this connection
    if X_pred_L is not None:
        X_pred_rep_I = X_pred_L.index_select(1, rep_atoms)
        dist = torch.cdist(X_pred_rep_I, X_pred_rep_I)
        if not self.use_af3_style_binning_and_final_layer_norms:
            # bins are 3.375 to 20.375 in 1.75 increments according to pseudocode
            dist_one_hot = F.one_hot(
                discretize_distance_matrix(
                    dist, min_distance=3.375, max_distance=20.875, num_bins=10
                ),
                num_classes=11,
            )
        else:
            # published code is 3.25 to 50.75, with 39 bins
            dist_one_hot = F.one_hot(
                discretize_distance_matrix(
                    dist, min_distance=3.25, max_distance=50.75, num_bins=39
                ),
                num_classes=40,
            )

        Z_trunk_II = Z_trunk_II + self.process_pred_distances(dist_one_hot.float())

        if self.use_Cb_distances:
            # embed difference between observed cb and ideal cb positions
            Cb_distances = calc_Cb_distances(
                X_pred_L, seq, rep_atoms, frame_atom_idxs
            )
            Cb_distances_one_hot = F.one_hot(
                discretize_distance_matrix(
                    Cb_distances,
                    min_distance=0.0001,
                    max_distance=0.25,
                    num_bins=24,
                ),
                num_classes=25,
            )
            Cb_logits = self.process_Cb_distances(Cb_distances_one_hot.float())
            # symmetrize the logits
            if self.symmetrize_Cb_logits:
                Cb_logits = Cb_logits[:, None, :, :] + Cb_logits[:, :, None, :]
            else:
                Cb_logits = Cb_logits[:, None, :, :]

            Z_trunk_II = Z_trunk_II + Cb_logits

    if not self.use_af3_style_binning_and_final_layer_norms:
        S_trunk_residual_I = S_trunk_I.clone()
        Z_trunk_residual_II = Z_trunk_II.clone()

    # process with pairformer stack
    for n in range(len(self.pairformer)):
        S_trunk_I, Z_trunk_II = self.pairformer[n](S_trunk_I, Z_trunk_II)

    # despite doing so in their pseudocode, af3's published code does not add the residual back
    if not self.use_af3_style_binning_and_final_layer_norms:
        S_trunk_I = S_trunk_residual_I + S_trunk_I
        Z_trunk_II = Z_trunk_residual_II + Z_trunk_II

        # linearly project for each prediction task
        pde_logits = self.predict_pde(
            Z_trunk_II + Z_trunk_II.transpose(-2, -3)
        )  # BUG: needs to be symmetrized correctly

        pae_logits = self.predict_pae(Z_trunk_II)

        plddt_logits = self.predict_plddt(S_trunk_I)
        exp_resolved_logits = self.predict_exp_resolved(S_trunk_I)

    # af3's published code does not add the residual back and has some additional layernorms before the linear projections
    # they also do the pde slightly differently, adding the transpose after the linear projection
    else:
        left_distance_logits = self.predict_pde(self.layernorm_pde(Z_trunk_II))
        right_distance_logits = left_distance_logits.transpose(-2, -3)
        pde_logits = left_distance_logits + right_distance_logits

        pae_logits = self.predict_pae(self.layernorm_pae(Z_trunk_II))
        plddt_logits = self.predict_plddt(self.layernorm_plddt(S_trunk_I))
        exp_resolved_logits = self.predict_exp_resolved(
            self.layernorm_exp_resolved(S_trunk_I)
        )

    return dict(
        pde_logits=pde_logits,
        pae_logits=pae_logits,
        plddt_logits=plddt_logits,
        exp_resolved_logits=exp_resolved_logits,
    )


def forward(self, S_inputs_I, S_trunk_I, Z_trunk_II, X_pred_L, seq, rep_atoms, frame_atom_idxs=None):
    """``ConfidenceHead.forward`` with the sample-invariant prologue read from the per-item cache (computed on its first call)."""
    prev = _PREV["forward"]
    if not STATE["on"] or Z_trunk_II is None or not getattr(Z_trunk_II, "is_cuda", False) or (X_pred_L is None and self.use_Cb_distances):
        STATE["passthrough"] = int(STATE["passthrough"]) + 1                  # off, or a call shape the hoist does not take (CPU tensors): the class's own forward
        return prev(self, S_inputs_I, S_trunk_I, Z_trunk_II, X_pred_L, seq, rep_atoms, frame_atom_idxs=frame_atom_idxs)
    STATE["calls"] = int(STATE["calls"]) + 1
    key = (self, S_inputs_I, S_trunk_I, Z_trunk_II)
    ver = _versions(key[1:])
    with _LOCK:
        e = _CACHE["entry"]
        hit = e is not None and len(e["key"]) == 4 and all(a is b for a, b in zip(e["key"], key))
        if hit and e["ver"] != ver:                                            # an operand was modified in place since the prologue ran: recompute, named
            STATE["stale"] = int(STATE["stale"]) + 1
            hit = False
        if hit:
            S_pro, Z_pro = e["out"]
            STATE["reused"] = int(STATE["reused"]) + 1
        else:
            _CACHE["entry"] = None
    if not hit:
        S_pro, Z_pro = _prologue(self, S_inputs_I, S_trunk_I, Z_trunk_II)
        with _LOCK:
            _CACHE["entry"] = {"key": key, "ver": ver, "out": (S_pro, Z_pro)}
        STATE["prologues"] = int(STATE["prologues"]) + 1
    return _tail(self, S_pro.clone(), Z_pro, X_pred_L, seq, rep_atoms, frame_atom_idxs)


forward.__confhoist__ = True


def clear(reason: str = "item") -> bool:
    """Drop the cached prologue (end of an item). Returns whether an entry was held."""
    with _LOCK:
        held = _CACHE["entry"] is not None
        _CACHE["entry"] = None
    if held:
        STATE["cleared"] = int(STATE["cleared"]) + 1
    return held


def _model_forward(self, *a, **k):
    """``RF3WithConfidence.forward`` unchanged, the prologue cache dropped when it returns (or raises): nothing outlives the item."""
    try:
        return _PREV["model_forward"](self, *a, **k)
    finally:
        clear("item")


# ------------------------------------------------------------------------------------------------------------ install
def _cuda_ok():
    try:
        import torch
    except ImportError:
        return None, "torch is not importable"
    if not torch.cuda.is_available():
        return None, CPU_PROCESS
    return torch, None


def enable(heads_module=None) -> dict:
    """Install the hoisted forward on ``ConfidenceHead`` (class-wide: every instance, the model's one). Idempotent. Returns :func:`describe`."""
    if STATE["installed"]:
        STATE["on"] = True
        return describe()
    M = heads_module or sys.modules.get(HEADS_MODULE)
    if M is None:
        try:
            import importlib
            M = importlib.import_module(HEADS_MODULE)
        except Exception as e:                                                  # noqa: BLE001 — upstream absent: named, not installed
            STATE.update(on=False, reason=f"{HEADS_MODULE} is not importable: {type(e).__name__}: {e}", refusal="upstream_absent")
            return describe()
    cls = getattr(M, "ConfidenceHead", None)
    if cls is None:
        STATE.update(on=False, reason=f"{HEADS_MODULE} has no ConfidenceHead", refusal="upstream_absent")
        return describe()
    torch, why = _cuda_ok()
    if torch is None:
        STATE.update(on=False, reason=why, refusal=None if why == CPU_PROCESS else "no_torch")
        return describe()
    dg = upstream_digest(cls)
    STATE["upstream_sha12"] = (dg or "unreadable")[:12]
    if dg != FORWARD_NORM_SHA256:                                              # another foundry: the carried tail is not its code — refused by name, stock's forward stays
        STATE.update(on=False, reason=f"ConfidenceHead.forward's source ({STATE['upstream_sha12']}) is not the pin's ({FORWARD_NORM_SHA256[:12]}): "
                                      f"the hoist carries the pin's statements and does not replace code it has not read", refusal=f"upstream_bytes:{STATE['upstream_sha12']}")
        return describe()
    _PREV["forward"], _PREV["cls"] = cls.forward, cls
    forward.__wrapped_confhoist__ = cls.forward
    cls.forward = forward
    STATE.update(installed=True, on=True, reason=None, refusal=None)
    wrap_model()
    return describe()


def wrap_model(model_module=None) -> bool:
    """Wrap ``RF3WithConfidence.forward`` so the cache is dropped per item (idempotent; a no-op until ``rf3.model.RF3`` has executed)."""
    if STATE["model_wrapped"]:
        return True
    M = model_module or sys.modules.get(MODEL_MODULE)
    cls = getattr(M, "RF3WithConfidence", None) if M is not None else None
    if cls is None or not hasattr(cls, "forward"):
        return False
    _PREV["model_forward"], _PREV["model_cls"] = cls.forward, cls
    cls.forward = _model_forward
    STATE["model_wrapped"] = True
    return True


def disable() -> None:
    """Restore the class's own forward(s); the cache is dropped."""
    clear("disable")
    if STATE["installed"] and _PREV["cls"] is not None:
        _PREV["cls"].forward = _PREV["forward"]
    if STATE["model_wrapped"] and _PREV["model_cls"] is not None and _PREV["model_cls"].forward is _model_forward:
        _PREV["model_cls"].forward = _PREV["model_forward"]
    STATE.update(installed=False, on=False, model_wrapped=False)


def decline(kind: str, why: str) -> dict:
    """Off BY NAME for this process (``conflict:<kind>`` on the LEVER line; the tally block carries the sentence): nothing is installed."""
    STATE.update(on=False, conflict=kind, reason=why)
    return describe()


def arm(rep: dict, install_watch: Callable) -> dict:
    """Activation-time hook: when the row names ``confhoist``, patch ``ConfidenceHead`` as soon as its module executes and wrap the model's
    forward as soon as ``rf3.model.RF3`` executes (``install_watch(trigger, callback, rep)`` is the kit's one import-watch installer, so a
    watch another lever holds on the same module is chained, never replaced); modules imported already are patched at once. Records
    ``rep["confhoist"]`` (:func:`describe`)."""
    levers = rep.get("levers") or []
    if LN_NAME in levers and NAME not in levers:                               # confln without its prologue: refused by name (the tally's ln block carries it; fold's verdict fails)
        STATE.update(fastln=False, fastln_refusal=LN_REQUIRES)
        rep[LN_NAME] = describe_ln()
    if NAME not in levers or STATE["armed"]:
        return rep
    STATE["armed"] = True
    STATE["fastln"] = LN_NAME in levers
    if int(rep.get("n_gpu") or 1) > 1:                                        # the row-sharded line owns the head's forward: off by name, every rank alike
        rep[NAME] = decline("n_gpu", CONFLICT_N_GPU)
        if STATE["fastln"]:
            rep[LN_NAME] = describe_ln()
        return rep

    def on_heads(module):
        rep[NAME] = enable(module)

    def on_model(module):
        wrap_model(module)
        rep[NAME] = describe()

    if HEADS_MODULE in sys.modules:
        on_heads(sys.modules[HEADS_MODULE])
    else:
        install_watch(HEADS_MODULE, on_heads, rep)
    if MODEL_MODULE in sys.modules and getattr(sys.modules[MODEL_MODULE], "RF3WithConfidence", None) is not None:
        on_model(sys.modules[MODEL_MODULE])
    else:
        install_watch(MODEL_MODULE, on_model, rep)
    rep[NAME] = describe()
    if STATE["fastln"]:
        rep[LN_NAME] = describe_ln()
    return rep


def describe_ln() -> dict:
    """The confln lever's tally block: on / calls / ok / reason (a refusal without confhoist, or confhoist's own state when it did not engage)."""
    d = describe()
    on = bool(STATE["fastln"]) and d["installed"]
    out = {"name": LN_NAME, "on": on, "calls": int(STATE["fastln_calls"]), "refusal": STATE["fastln_refusal"], "conflict": d.get("conflict"),
           "impl": __file__, "construction": "torch.var_mean(correction=0); (x-mean)*rsqrt(var+eps)", "eps": LN_EPS}
    if STATE["fastln_refusal"]:
        out.update(ok=False, reason=f"{LN_NAME}:refused:{STATE['fastln_refusal']}")
    elif d.get("conflict"):
        out.update(ok=True, reason=d.get("reason"))
    else:
        out.update(ok=bool(on and d["ok"]), reason=None if (on and d["ok"]) else (d.get("reason") or f"{NAME} did not engage"))
    return out


# ------------------------------------------------------------------------------------------------------------ report surface
def describe() -> dict:
    census = {k: int(STATE[k]) for k in ("calls", "prologues", "reused", "cleared", "stale", "passthrough")}
    d = {"name": NAME, "on": bool(STATE["on"]), "installed": bool(STATE["installed"]), "model_wrapped": bool(STATE["model_wrapped"]),
         "armed": bool(STATE["armed"]), "reason": STATE["reason"], "refusal": STATE["refusal"], "conflict": STATE["conflict"],
         "upstream_sha12": STATE["upstream_sha12"], "pin_sha12": FORWARD_NORM_SHA256[:12], "impl": __file__, "census": census}
    bad = problems(d)
    d["ok"] = not bad and (d["installed"] or d["conflict"] is not None)     # the pred verdict (fold.lever_failures): engaged cleanly or declined by name; the CPU helper's no-op is ok=False with no problems (pf.py's convention)
    if bad:
        d["reason"] = "; ".join(bad)
    elif not d["ok"] and not d["reason"]:
        d["reason"] = f"{HEADS_MODULE} never executed in this process (the hoist was armed, nothing to patch)"
    return d


def problems(desc: Optional[dict] = None) -> List[str]:
    """The pred verdict's failures for this lever: a refusal by name (upstream bytes / absent / no torch), or samples served while the model
    wrapper that bounds the cache to the item is missing. The CPU helper's named no-op and a process that never ran the head are not failures."""
    d = desc or describe()
    out = []
    if d.get("refusal"):
        out.append(f"{NAME}:refused:{d['refusal']}")
    if d.get("installed") and d["census"]["calls"] and not d.get("model_wrapped"):
        out.append(f"{NAME}:cache_unbounded:model_forward_not_wrapped")
    if d.get("installed") and d["census"]["stale"]:
        out.append(f"{NAME}:stale_operands:{d['census']['stale']}")
    return out


def lever_tokens(desc: Optional[dict] = None) -> List[tuple]:
    """(key, value) tokens for the LEVER line: calls / prologues / reused / cleared (+ upstream sha12)."""
    d = desc or describe()
    c = d["census"]
    return [("calls", c["calls"]), ("prologues", c["prologues"]), ("reused", c["reused"]), ("cleared", c["cleared"]), ("upstream", d.get("upstream_sha12"))]
