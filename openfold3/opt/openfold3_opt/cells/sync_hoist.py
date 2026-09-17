"""The `sync_hoist` lever (exact class; the `exact` and `fast` lines): the trunk's recycle loop reads three values back from the GPU on every
recycle pass that are constants of the forward — each read-back is a host synchronisation that drains the launch queue and stalls the host
until the previous pass's kernels (on the fast line: the replayed pairformer graph) have finished, so the next pass's MSA-module launches
never overlap GPU work. This cell keeps the engine's statements and their operands and removes the read-backs (OpenFold3 0.4.x
`projects.of3_all_atom.model.OpenFold3.run_trunk` → `MSAModuleEmbedder.forward` / `_subsample_all_msa`, and the trunk-kernels add-on's template
distinct-evaluation `TemplatePairStack.forward` wrap — served through an instance-level `forward` so the class attribute stays the add-on's), scoped to ONE `run_trunk` call (every memo is filled inside the call and dropped at its end):

  msa_depth  `no_subsampled_all_msa = torch.randint(low=min, high=max + 1, size=(1,), device=…).item()` — the draw is kept exactly as the engine
             makes it (the CUDA generator's offset advances as before: every later draw is the engine's), the read-back is skipped when
             `min_subsampled_all_msa == max_subsampled_all_msa` (the shipped configuration: 1024 == 1024 — the draw can only be `min`); a
             configuration with a range reads the draw back as the engine does, counted `fallback:msa_depth_range`.
  msa_rows   `_subsample_all_msa`: `valid_idx = valid_msa.nonzero()` / `invalid_idx = (~valid_msa).nonzero()` and their element counts are
             functions of `batch["msa_mask"]`, constant through the forward — computed by the engine's statements at the first pass, held with the
             mask they came from (storage / shape / dtype / device key + the tensor itself) and reused by the later passes; `torch.randperm` and the
             `index_select`s run every pass exactly as the engine's (same operands, same generator draws).
  templ      the add-on's distinct-template evaluation compares the T template embeddings BY VALUE (`torch.equal`) at every pass; the verdict is a
             property of the query's template features (the z-dependent term is common to every template), so the add-on decides at the first
             template-stack call of the trunk call (its own statement, its own census) and this cell serves the later calls of the same
             (stack, shapes, dtype) from that verdict: identical → the stack on template 0 expanded ×T (the add-on's statements), distinct → the
             add-on's own path. The verdict is read from the add-on's census delta; an add-on without that census leaves every call to it
             (`fallback:addon_stats_absent|addon_verdict_unread`).

Same kernels on the same operands in the same order, the CUDA generator consumed identically: bitwise the line without it (exact class). Not
served, BY NAME (the engine's statements run): `subsample_main_msa` configurations (`fallback:main_msa`), no subsampling (`no_subsample`), a
batch of more than one query (`batch_gt1`), a call outside a `run_trunk` scope (`no_scope`), a part switched off (OPENFOLD3_OPT_SYNC_HOIST_PARTS).
No CUDA graph is captured here and nothing depends on the attention kernel in use (DS4Sci included). The served bodies are compared with the
engine's source once at install; an engine whose statements differ is refused by name and left unpatched (`state=refused reason=source_differs:<fn>`).

Generic to the OpenFold3 code family (0.4.x / 0.5.x share these statements): a CORE-lift candidate (opt_core.of3_trunk) — kit-namespaced here.

Switches: OPENFOLD3_OPT_SYNC_HOIST=1; OPENFOLD3_OPT_SYNC_HOIST_PARTS = `all` (default) | comma list of msa_depth,msa_rows,templ | `none` (installed,
serving nothing: the ablation arm). Exit line `[openfold3-opt/sync_hoist] LEVER name=sync_hoist state=on parts=… trunk_calls=… msa_depth=served:n
msa_rows=fill:n,hit:n templ=fill:n,hit:n,identical:n,distinct:n fallback=<reason:n,…|none>`."""
from __future__ import annotations

import atexit
import inspect
import os
import sys
from typing import Any, Dict, Optional, Tuple

PREFIX = "[openfold3-opt/sync_hoist]"
ENV = "OPENFOLD3_OPT_SYNC_HOIST"
ENV_PARTS = "OPENFOLD3_OPT_SYNC_HOIST_PARTS"
VALUES = ("1",)
PARTS_ALL: Tuple[str, ...] = ("msa_depth", "msa_rows", "templ")
M_MODEL = "openfold3.projects.of3_all_atom.model"                       # class OpenFold3 (run_trunk: the recycle loop)
MODEL_CLASS = "OpenFold3"
M_IE = "openfold3.core.model.feature_embedders.input_embedders"          # class MSAModuleEmbedder
M_TM = "openfold3.core.model.latent.template_module"                     # class TemplatePairStack
ADDON_MODULE = "of3t_levers"                                             # the trunk-kernels add-on's lever module (STATS: templ_distinct_hits / templ_distinct_miss)
ADDON_KEYS = ("templ_distinct_hits", "templ_distinct_miss")

# the engine statements this cell serves — checked once at install (inspect.getsource); a tree whose statements differ is refused by name
SOURCE_FORWARD = ('batch_dims = batch["msa"].shape[:-3]', 'batch["has_deletion"].unsqueeze(-1)', 'batch["deletion_value"].unsqueeze(-1)', 'msa_mask = batch["msa_mask"]',
                  "if self.subsample_main_msa:", "elif self.subsample_all_msa:", "no_subsampled_all_msa = torch.randint(", "low=self.min_subsampled_all_msa,",
                  "high=int(self.max_subsampled_all_msa + 1),", "size=(1,),", "device=msa_feat.device,", ").item()", "if math.prod(batch_dims) > 1:",
                  "no_subsampled_all_msa=no_subsampled_all_msa,", "m = self.linear_m(msa_feat)", "m = m + self.linear_s_input(s_input).unsqueeze(-3)", "return m, msa_mask")
SOURCE_SUBSAMPLE = ("feat_seq_dim = -3", "mask_seq_dim = -2", "valid_msa = (msa_mask.sum(dim=mask_seq_dim + 1) > 0).squeeze()", "valid_idx = valid_msa.nonzero().squeeze()",
                    "invalid_idx = (~valid_msa).nonzero().squeeze()", "if valid_idx.numel() >= no_subsampled_all_msa:",
                    "permuted_idx = valid_idx[torch.randperm(valid_idx.numel(), device=device)]", "selected = permuted_idx[:no_subsampled_all_msa]",
                    "take_invalid = no_subsampled_all_msa - valid_idx.numel()", "if invalid_idx.numel() > 0:", "torch.randperm(invalid_idx.numel(), device=device)",
                    "selected = torch.cat([valid_idx, permuted_idx[:take_invalid]], dim=0)", "selected = valid_idx",
                    "feat_sub = msa_feat.index_select(feat_seq_dim, selected)", "mask_sub = msa_mask.index_select(mask_seq_dim, selected)")

STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "parts": (), "trunk_calls": 0, "msa_depth_served": 0, "msa_rows_fill": 0, "msa_rows_hit": 0,
                         "templ_fill": 0, "templ_hit": 0, "templ_identical": 0, "templ_distinct": 0, "fallback": {}, "patched": []}
_SCOPE: Dict[str, Any] = {"gen": None, "depth": 0, "rows": None, "templ": {}, "shadowed": set()}


def _log(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")


def _count(d: dict, k, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    parts(environ)                                                       # a mistyped part list is refused with the switch, before install
    return True


def parts(environ=None) -> Tuple[str, ...]:
    """The served parts: OPENFOLD3_OPT_SYNC_HOIST_PARTS unset / `all` = every part; `none` = installed, serving nothing; else a comma list of PARTS_ALL."""
    environ = os.environ if environ is None else environ
    raw = (environ.get(ENV_PARTS) or "").strip().lower()
    if raw in ("", "all"):
        return PARTS_ALL
    if raw == "none":
        return ()
    out = tuple(p.strip() for p in raw.split(",") if p.strip())
    bad = [p for p in out if p not in PARTS_ALL]
    if bad or not out:
        raise ValueError(f"{ENV_PARTS}={raw!r}: unknown part(s) {bad} (known: {', '.join(PARTS_ALL)}, or all / none)")
    return tuple(p for p in PARTS_ALL if p in out)


def serving() -> bool:
    return STATE["state"] == "on"


def release() -> None:
    """Drop every memo (the run_trunk scope's end does this; also callable at an item boundary)."""
    _SCOPE["rows"] = None
    _SCOPE["templ"] = {}


def _tkey(t) -> tuple:
    return (t.data_ptr(), tuple(t.shape), t.dtype, str(t.device))


def _addon_stats() -> Optional[dict]:
    m = sys.modules.get(ADDON_MODULE)
    st = getattr(m, "STATS", None) if m is not None else None
    if isinstance(st, dict) and all(k in st for k in ADDON_KEYS):
        return st
    return None


def subsample_rows(msa_feat, msa_mask, no_subsampled_all_msa: int, memo: Optional[dict]):
    """The engine's `_subsample_all_msa` statements with the row-index memo: `memo` (a dict this cell owns for one run_trunk scope, or None = no
    memo) holds valid_idx / invalid_idx / their counts keyed on the mask tensor they were computed from. Returns (feat_sub, mask_sub, event) with
    event `fill` | `hit` | `nomemo`."""
    import torch
    feat_seq_dim = -3
    mask_seq_dim = -2
    key = _tkey(msa_mask)
    if memo is not None and memo.get("key") == key and memo.get("mask") is msa_mask:
        valid_idx, invalid_idx, n_valid, n_invalid = memo["valid_idx"], memo["invalid_idx"], memo["n_valid"], memo["n_invalid"]
        event = "hit"
    else:
        valid_msa = (msa_mask.sum(dim=mask_seq_dim + 1) > 0).squeeze()  # [N_msa]
        if valid_msa.ndim == 0:
            valid_msa = valid_msa.unsqueeze(0)
        valid_idx = valid_msa.nonzero().squeeze()
        invalid_idx = (~valid_msa).nonzero().squeeze()
        if valid_idx.ndim == 0:
            valid_idx = valid_idx.unsqueeze(0)
        if invalid_idx.ndim == 0:
            invalid_idx = invalid_idx.unsqueeze(0)
        n_valid, n_invalid = valid_idx.numel(), invalid_idx.numel()
        if memo is not None:
            memo.update(key=key, mask=msa_mask, valid_idx=valid_idx, invalid_idx=invalid_idx, n_valid=n_valid, n_invalid=n_invalid)
            event = "fill"
        else:
            event = "nomemo"
    device = msa_feat.device
    if n_valid >= no_subsampled_all_msa:
        permuted_idx = valid_idx[torch.randperm(n_valid, device=device)]
        selected = permuted_idx[:no_subsampled_all_msa]
    else:
        take_invalid = no_subsampled_all_msa - n_valid
        if n_invalid > 0:
            permuted_idx = invalid_idx[torch.randperm(n_invalid, device=device)]
            selected = torch.cat([valid_idx, permuted_idx[:take_invalid]], dim=0)
        else:
            selected = valid_idx
    feat_sub = msa_feat.index_select(feat_seq_dim, selected)
    mask_sub = msa_mask.index_select(mask_seq_dim, selected)
    return feat_sub, mask_sub, event


def _make_embedder_forward(orig):
    import math

    import torch

    def _fb(why, self, batch, s_input):
        _count(STATE["fallback"], why)
        return orig(self, batch, s_input)

    def forward(self, batch, s_input):
        if not serving():
            return orig(self, batch, s_input)
        if _SCOPE["gen"] is None:
            return _fb("no_scope", self, batch, s_input)
        served = STATE["parts"]
        if "msa_depth" not in served and "msa_rows" not in served:
            return _fb("parts_off:msa", self, batch, s_input)
        if self.subsample_main_msa:
            return _fb("main_msa", self, batch, s_input)
        if not self.subsample_all_msa:
            return _fb("no_subsample", self, batch, s_input)
        batch_dims = batch["msa"].shape[:-3]
        if math.prod(batch_dims) > 1:
            return _fb("batch_gt1", self, batch, s_input)
        # [*, N_msa, N_token, 34] — the engine's statements
        msa_feat = torch.cat([batch["msa"], batch["has_deletion"].unsqueeze(-1), batch["deletion_value"].unsqueeze(-1)], dim=-1)
        msa_mask = batch["msa_mask"]
        lo, hi = self.min_subsampled_all_msa, int(self.max_subsampled_all_msa + 1)
        draw = torch.randint(low=lo, high=hi, size=(1,), device=msa_feat.device)      # the engine's draw, kept: the generator advances exactly as before
        if "msa_depth" in served and hi - int(lo) == 1:
            no_subsampled_all_msa = int(lo)                                            # randint(low=min, high=min + 1) can only draw `min`: no read-back
            STATE["msa_depth_served"] += 1
        else:
            if "msa_depth" in served:
                _count(STATE["fallback"], "msa_depth_range")
            no_subsampled_all_msa = draw.item()
        memo = _SCOPE["rows"] if "msa_rows" in served else None
        msa_feat, msa_mask, event = subsample_rows(msa_feat, msa_mask, no_subsampled_all_msa, memo)
        if event == "fill":
            STATE["msa_rows_fill"] += 1
        elif event == "hit":
            STATE["msa_rows_hit"] += 1
        # [*, N_seq, N_token, C_m]
        m = self.linear_m(msa_feat)
        m = m + self.linear_s_input(s_input).unsqueeze(-3)
        return m, msa_mask
    forward.__wrapped__ = orig; forward._of3opt_sync_hoist = True
    return forward


def _make_tps_forward(inst):
    """The instance-level `forward` of one TemplatePairStack (shadowing the class attribute the way the kit's phase timer does, so the class
    attribute stays the trunk-kernels add-on's own wrap and its probe stays truthful): the add-on's by-value verdict once per run_trunk scope and
    (shapes, dtype, device); identical → template 0 through the class forward (the add-on hands T == 1 to the stack unexamined), expanded ×T —
    the add-on's statements. The class attribute is read at call time, so whatever wraps it later still runs."""

    def inner(t, mask, *a, **k):
        return type(inst).forward(inst, t, mask, *a, **k)

    def forward(t, mask=None, *a, **k):
        if not serving() or _SCOPE["gen"] is None or "templ" not in STATE["parts"] or t.dim() < 5 or t.shape[-4] <= 1:
            return inner(t, mask, *a, **k)
        n_templ = t.shape[-4]
        key = (id(inst), tuple(t.shape), t.dtype, str(t.device), None if mask is None else (tuple(mask.shape), mask.dtype))
        verdict = _SCOPE["templ"].get(key)
        if verdict is None:                                                            # the first call of the scope: the add-on decides by value; its census delta is the verdict
            st = _addon_stats()
            if st is None:
                _count(STATE["fallback"], "addon_stats_absent")
                return inner(t, mask, *a, **k)
            h0, m0 = int(st[ADDON_KEYS[0]]), int(st[ADDON_KEYS[1]])
            out = inner(t, mask, *a, **k)
            h1, m1 = int(st[ADDON_KEYS[0]]), int(st[ADDON_KEYS[1]])
            if h1 == h0 + 1 and m1 == m0:
                _SCOPE["templ"][key] = True; STATE["templ_fill"] += 1; STATE["templ_identical"] += 1
            elif m1 == m0 + 1 and h1 == h0:
                _SCOPE["templ"][key] = False; STATE["templ_fill"] += 1; STATE["templ_distinct"] += 1
            else:
                _count(STATE["fallback"], "addon_verdict_unread")
            return out
        if verdict:
            t0 = t.narrow(-4, 0, 1)
            m = mask if (mask is None or mask.shape[-3] == 1) else mask.narrow(-3, 0, 1)
            out = inner(t0, m, *a, **k)                                                  # T == 1: the add-on hands it to the stack unexamined
            STATE["templ_hit"] += 1
            return out.expand(*out.shape[:-4], n_templ, *out.shape[-3:])
        _count(STATE["fallback"], "templ_distinct")
        return inner(t, mask, *a, **k)
    forward._of3opt_sync_hoist = True
    return forward


def _shadow_template_stacks(model) -> None:
    """Give every TemplatePairStack instance of the model this cell's forward at the INSTANCE level (once per instance) — at the first run_trunk
    call, when every add-on has long installed its class-level wrap."""
    import importlib
    try:
        TM = importlib.import_module(M_TM)
    except Exception as e:  # noqa: BLE001
        _count(STATE["fallback"], f"templ_module:{type(e).__name__}")
        return
    n = 0
    for mod in model.modules():
        if isinstance(mod, TM.TemplatePairStack) and not getattr(mod.__dict__.get("forward"), "_of3opt_sync_hoist", False):
            mod.forward = _make_tps_forward(mod); n += 1
    if n:
        STATE["patched"].append(f"TemplatePairStack.forward@instance x{n}")


def _make_run_trunk(orig):
    def run_trunk(self, *a, **k):
        if not serving():
            return orig(self, *a, **k)
        if "templ" in STATE["parts"] and id(self) not in _SCOPE["shadowed"]:
            _SCOPE["shadowed"].add(id(self)); _shadow_template_stacks(self)
        outer = _SCOPE["depth"] == 0
        if outer:
            STATE["trunk_calls"] += 1
            _SCOPE["gen"] = STATE["trunk_calls"]; _SCOPE["rows"] = {}; _SCOPE["templ"] = {}
        _SCOPE["depth"] += 1
        try:
            return orig(self, *a, **k)
        finally:
            _SCOPE["depth"] -= 1
            if outer:
                _SCOPE["gen"] = None
                release()
    run_trunk.__wrapped__ = orig; run_trunk._of3opt_sync_hoist = True
    return run_trunk


def census_line() -> str:
    fb = ",".join("%s:%d" % kv for kv in sorted(STATE["fallback"].items())) or "none"
    return (f"{PREFIX} LEVER name=sync_hoist state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" parts={','.join(STATE['parts']) or 'none'} trunk_calls={STATE['trunk_calls']} msa_depth=served:{STATE['msa_depth_served']}"
            f" msa_rows=fill:{STATE['msa_rows_fill']},hit:{STATE['msa_rows_hit']}"
            f" templ=fill:{STATE['templ_fill']},hit:{STATE['templ_hit']},identical:{STATE['templ_identical']},distinct:{STATE['templ_distinct']} fallback={fb}")


def _source_missing(fn, statements) -> list:
    try:
        src = inspect.getsource(fn)
    except Exception as e:  # noqa: BLE001
        return [f"source unreadable: {type(e).__name__}"]
    return [s for s in statements if s not in src]


def install(environ=None) -> dict:
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    STATE["parts"] = parts(environ)
    import importlib
    MD = importlib.import_module(M_MODEL)
    IE = importlib.import_module(M_IE)
    Model = getattr(MD, MODEL_CLASS)
    Emb = IE.MSAModuleEmbedder
    missing = {"MSAModuleEmbedder.forward": _source_missing(Emb.forward, SOURCE_FORWARD), "MSAModuleEmbedder._subsample_all_msa": _source_missing(Emb._subsample_all_msa, SOURCE_SUBSAMPLE)}
    bad = {k: v for k, v in missing.items() if v}
    if bad:
        which = next(iter(bad))
        STATE.update(installed=True, state="refused", reason=f"source_differs:{which}")
        _log(f"REFUSED: {which} differs from the statements this lever serves (missing {bad[which][:3]}) — the engine's runs as it is")
    else:
        if not getattr(Emb.forward, "_of3opt_sync_hoist", False):
            Emb.forward = _make_embedder_forward(Emb.forward); STATE["patched"].append("MSAModuleEmbedder.forward")
        if not getattr(Model.run_trunk, "_of3opt_sync_hoist", False):
            Model.run_trunk = _make_run_trunk(Model.run_trunk); STATE["patched"].append(f"{MODEL_CLASS}.run_trunk")
        STATE.update(installed=True, state="on")
        _log(f"installed: {MODEL_CLASS}.run_trunk scopes the memos; MSAModuleEmbedder.forward keeps the depth draw without its read-back (min == max) and the MSA row "
             f"indices per trunk call; each TemplatePairStack instance's forward (shadowed at the first trunk call, over the add-on's class-level wrap) serves the add-on's distinct-template verdict per trunk call "
             f"(parts={','.join(STATE['parts']) or 'none'})")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
