"""The row-block confidence head — the `confhead` lever of the `big` mode (the `resident` line at and above `modes.CONF_MIN_TOKENS` tokens).

The stock confidence heads (`openfold3.core.model.heads.prediction_heads.PredictedAlignedErrorHead` / `PredictedDistanceErrorHead`) project the
head pairformer's pair representation zij [*, S, N, N, C_z] to logits [*, S, N, N, 64] — two fp32 tensors of S x N² x 64 x 4 B each,
materialised BEFORE the confidence scorer reads them in row blocks: on the resident line they are the process peak of the
confidence phase (the port's chunked scorer `of3o_confidence.pair_row_pass` streams the logits row block by row block, so the full tensors exist
only to be sliced).

This lever never materialises them: the heads' forward returns a `RowBlockLogits` — the pair representation kept as it is (already resident) and
the head's LayerNorm + Linear evaluated per row block on demand, exactly where the scorer slices (`pde_l[sl, i0:i1]`, `pae_l[sl, i0:i1]`):
    PAE:  logits[s, i0:i1] = linear(ln(z[s, i0:i1, :]))                                                         (stock `_compute_logits`)
    PDE:  logits[s, i0:i1] = linear(ln(z[s, i0:i1, :])) + linear(ln(z[s, :, i0:i1])).transpose(-2, -3)          (stock: f(z) + f(z)ᵀ, the block of it)
under the autocast state the stock head ran in, cast to the dtype the caller asked for (`.to(dtype=)` is honoured lazily). The statement per
(i, j) is the stock one (a LayerNorm over C and a C -> 64 GEMM per pair; the PDE add in stock's operand order); the GEMM's M dimension changes
(cuBLAS may pick a different kernel for a row block than for the full slab), so the lever is tolerance-class: the mode's band decides the word.

Requires the port's chunked scorer (`OF3O_CONF_MODE=chunked`: `of3o_confidence.pair_row_pass` is the only consumer that slices; the stock
scorer indexes the logits as one tensor and would fail by name on the lazy object) — the line exports both switches together and the size
gate (`modes.conf_gate`) removes both together.

Installed by the hook `opt/openfold3_ob0_opt/hooks/confhead/sitecustomize.py` (the resident line's entry hook; `OPENFOLD3_OB0_OPT_CONFHEAD=1`) after
`openfold3.core.model.heads.prediction_heads` executes; the record: `STATE` (installed, blocks evaluated, rows per block, the dtypes) read by the
kit's levers record (`stack` probe `confhead`) and `census()`.

No core code: the row-block statement is the port's scorer loop (one loop); `opt_core.mem.chunk.confidence_head_chunked` takes a materialised
logits tensor (its `statement` runs on blocks of logits) and is the wrong shape for a head whose logits must never exist.
"""
from __future__ import annotations

import sys
import threading
from typing import Optional

import torch

STATE: dict = {"installed": False, "blocks": 0, "rows_max": 0, "objects": 0, "released": 0, "dtype_out": None, "autocast": None, "errors": []}
_LOCK = threading.Lock()
_ORIG: dict = {}


MARKER = "[openfold3_ob0-opt/confhead]"           # the stderr line prefix of the lever's record (registry.LEVERS["confhead"].marker)


def _log(msg: str) -> None:
    sys.stderr.write(f"{MARKER} {msg}\n")
    sys.stderr.flush()


class RowBlockLogits(torch.Tensor):
    """The head's logits [*, S, N, N, C_out] evaluated per row block on demand from the pair representation zij [*, S, N, N, C_z] — a
    torch.Tensor WRAPPER subclass (torch.Tensor._make_wrapper_subclass: no storage; every aten op arrives at __torch_dispatch__), so the port's
    tree maps (`tensor_tree_map(slice_batch)`, `slice_sample`: `isinstance(t, torch.Tensor)` then `t[i]` / `t[j:j+1]`) and the scorer's
    `.reshape(S, n, n, -1)` / `pde_l[sl, i0:i1]` treat it as the logits tensor they expect.

    The op surface (the only ops the port's confidence path issues on the logits): select / slice on a LEAD dim (batch, sample) -> a view
    (the pair representation narrowed, nothing computed); slice on the ROW dim -> the head's block, a real tensor [lead..., r, N, C_out]
    (`pde_l[sl, i0:i1]`, `obj[..., i0:i1, :, :]`); view / reshape that only re-groups the lead dims (the pair axes and the bins stay) -> a
    view; `.to(dtype=)` / `.to(device=)` / `.float()` -> recorded, applied per block; detach / clone / contiguous -> the same lazy object;
    metadata reads (shape, dtype, device, size(), dim(), …) answer from the wrapper. Everything else is refused BY NAME (RuntimeError) at the
    Python level (`__getitem__`, `__torch_function__`) — the refusal does not depend on how a torch version lowers indexing or a method into
    aten calls — with `__torch_dispatch__` as the backstop for calls that arrive below the Python API: the logits are never materialised.
    """

    @staticmethod
    def __new__(cls, box, head, symmetric: bool, name: str, autocast, dtype, device, z_view, released_ok: bool = False):
        n = int(z_view.shape[-2]); c_out = int(head.c_out)
        shape = (*tuple(int(s) for s in z_view.shape[:-3]), n, n, c_out)
        r = torch.Tensor._make_wrapper_subclass(cls, shape, dtype=dtype, device=device, requires_grad=False)
        r._box = box                        # {"zij": the head's input as received, "released": bool} shared by every view of one head call
        r._z = z_view                       # this view's pair representation: [*lead, N, N, C_z] narrowed / re-grouped by the lead ops so far
        r.head = head
        r.symmetric = bool(symmetric)
        r._name = name
        r._autocast = autocast
        r._out_device = torch.device(device)
        return r

    # ---- construction -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def of(cls, zij, head, symmetric: bool, name: str, autocast=None, dtype=None):
        ac = autocast if autocast is not None else (torch.is_autocast_enabled(), torch.get_autocast_dtype("cuda") if hasattr(torch, "get_autocast_dtype") else None)
        box = {"zij": zij, "released": False}
        obj = cls(box, head, symmetric, name, ac, dtype if dtype is not None else zij.dtype, zij.device, zij)
        with _LOCK:
            STATE["objects"] += 1
            STATE["autocast"] = [bool(ac[0]), str(ac[1])]
        return obj

    def _view(self, z_view=None, dtype=None, device=None):
        return RowBlockLogits(self._box, self.head, self.symmetric, self._name, self._autocast, dtype if dtype is not None else self.dtype,
                              device if device is not None else self._out_device, z_view if z_view is not None else self._z)

    # ---- surface ----------------------------------------------------------------------------------------------------------------------------
    @property
    def zij(self):
        return self._box["zij"]

    @property
    def _released(self) -> bool:
        return bool(self._box["released"])

    @property
    def c_out(self) -> int:
        return int(self.head.c_out)

    @property
    def lead(self):
        return tuple(int(s) for s in self._z.shape[:-3])

    def release(self) -> None:
        """Drop the pair representation (after the scorer ran): a later block request refuses by name."""
        self._box["zij"] = None
        self._box["released"] = True
        with _LOCK:
            STATE["released"] += 1

    def __repr__(self) -> str:   # never materialise for a repr
        return f"RowBlockLogits({self._name}, shape={tuple(self.shape)}, dtype={self.dtype}, symmetric={self.symmetric}, released={self._released})"

    __str__ = __repr__

    # ---- the row block --------------------------------------------------------------------------------------------------------------------
    def block(self, i0: int, i1: int):
        """logits[..., i0:i1, :, :] over this view's lead ([*lead, r, N, C_out]) computed now, under the head's autocast state, in the recorded
        dtype and on the recorded device: PAE = linear(ln(z[rows])); PDE = that + linear(ln(z[:, rows])).transpose(-2, -3) (stock's operand order)."""
        if self._released or self._z is None or self._box["zij"] is None:
            raise RuntimeError(f"{self._name}: RowBlockLogits released (the scorer ran): the logits are not materialised by design")
        z = self._z
        n = int(z.shape[-2])
        lead = self.lead
        zf = z.reshape(-1, n, n, z.shape[-1]) if len(lead) != 1 else z          # [S', N, N, C_z]
        head = self.head
        enabled, ac_dtype = self._autocast
        ctx = torch.autocast("cuda", dtype=ac_dtype, enabled=bool(enabled)) if (zf.is_cuda and ac_dtype is not None) else torch.autocast("cuda", enabled=False)
        with torch.no_grad(), ctx:
            rows = zf[:, i0:i1]                                                     # [S', r, N, C_z]
            out = head.linear(head.layer_norm(rows))                                # stock `_compute_logits` on the block
            if self.symmetric:
                cols = zf[:, :, i0:i1]                                              # [S', N, r, C_z]
                out = out + head.linear(head.layer_norm(cols)).transpose(-2, -3)    # stock: logits + logits.transpose(-2, -3), the block of it
        out = out.to(dtype=self.dtype)
        if out.device != self._out_device:
            out = out.to(device=self._out_device)
        out = out.reshape(*lead, i1 - i0, n, self.c_out)
        with _LOCK:
            STATE["blocks"] += 1
            STATE["rows_max"] = max(STATE["rows_max"], int(i1 - i0))
        return out

    # ---- the op surface: ONE implementation per supported op, reached from Python (`__getitem__`, `__torch_function__`) and from aten
    # (`__torch_dispatch__`, the backstop for calls that arrive below the Python API). Every other op refuses BY NAME at the Python level —
    # the refusal does not depend on how a torch version lowers an indexing expression or a method into aten calls.
    def _refuse(self, what: str):
        raise RuntimeError(f"{self._name}: RowBlockLogits: {what} — the logits are never materialised by design (the confidence path reads row blocks only)")

    def _n_lead(self) -> int:
        return self.ndim - 3

    def _select(self, dim: int, index: int):
        """One index on one dim: a lead dim -> the pair representation narrowed (a view, nothing computed); the row dim -> that row's block
        with the row axis dropped; a pair-column / bin dim -> refused by name."""
        nd = self.ndim; dim = dim + nd if dim < 0 else dim
        size = int(self.shape[dim]) if 0 <= dim < nd else 0
        if not 0 <= dim < nd:
            self._refuse(f"select on dim {dim} of a {nd}-dim object")
        index = int(index); index = index + size if index < 0 else index
        if not 0 <= index < size:
            raise IndexError(f"{self._name}: RowBlockLogits index {index} out of range for dim {dim} of size {size}")
        n_lead = self._n_lead()
        if dim < n_lead:
            return self._view(self._z.select(dim, index))
        if dim == n_lead:
            return self.block(index, index + 1).select(n_lead, 0)
        self._refuse(f"select on dim {dim} (a pair-column / bin axis)")

    def _slice(self, dim: int, start, end, step=1):
        """A slice on one dim: a lead dim -> a view; the row dim -> the head's block [*lead, r, N, C_out] computed now; a pair-column / bin dim ->
        refused by name unless it is the whole axis (a no-op)."""
        nd = self.ndim; dim = dim + nd if dim < 0 else dim
        if not 0 <= dim < nd:
            self._refuse(f"slice on dim {dim} of a {nd}-dim object")
        if step is not None and int(step) != 1:
            self._refuse(f"slice with step {step} on dim {dim}")
        size = int(self.shape[dim])
        s0, s1, _ = slice(None if start is None else int(start), None if end is None else int(end)).indices(size)
        n_lead = self._n_lead()
        if dim < n_lead:
            return self._view(self._z.narrow(dim, s0, max(0, s1 - s0)))
        if dim == n_lead:
            return self.block(s0, max(s0, s1))
        if s0 == 0 and s1 == size:
            return self._view()
        self._refuse(f"slice [{s0}:{s1}] on dim {dim} (a pair-column / bin axis)")

    def _regroup(self, shape):
        """view / reshape that re-groups the LEAD dims only ([*lead, N, N, C_out] -> [*lead', N, N, C_out|-1]) -> a view; anything that touches the
        pair axes or the bins is refused by name."""
        shape = [int(s) for s in shape]
        n = int(self.shape[-2]); c = self.c_out
        if len(shape) < 3 or shape[-3] != n or shape[-2] != n or shape[-1] not in (-1, c):
            self._refuse(f"reshape{tuple(shape)}: only a lead re-grouping of [*, {n}, {n}, {c}] is supported")
        lead_new = shape[:-3]
        numel = 1
        for s in self.lead:
            numel *= s
        if lead_new.count(-1) > 1:
            self._refuse(f"reshape{tuple(shape)}: one inferred lead dim at most")
        if -1 in lead_new:
            known = 1
            for s in lead_new:
                if s != -1:
                    known *= s
            lead_new[lead_new.index(-1)] = numel // max(1, known)
        prod = 1
        for s in lead_new:
            prod *= s
        if prod != numel:
            self._refuse(f"reshape{tuple(shape)}: lead {lead_new} != {numel} elements")
        return self._view(self._z.reshape(*lead_new, n, n, self._z.shape[-1]))

    def _cast(self, *args, **kwargs):
        """`.to(...)` / `.float()` / `.cpu()` …: the dtype and device are recorded and applied per block (nothing is copied now)."""
        dtype, device = kwargs.get("dtype"), kwargs.get("device")
        for a in args:
            if isinstance(a, torch.dtype):
                dtype = a
            elif isinstance(a, (str, torch.device)):
                device = a
            elif isinstance(a, torch.Tensor) and not isinstance(a, RowBlockLogits):
                dtype, device = a.dtype, a.device
            elif isinstance(a, bool):                                                   # non_blocking / copy: nothing to do for a lazy object
                pass
            else:
                self._refuse(f".to({type(a).__name__} argument)")
        return self._view(dtype=dtype, device=torch.device(device) if device is not None else None)

    def __getitem__(self, idx):
        """Indexing in Python, independent of how a torch version lowers it: ints / step-1 slices on the lead dims (views), an int or a step-1
        slice on the row dim (the block), `...`, and whole-axis `:` on the pair-column / bin dims; None (newaxis), tensors, lists, bools and
        steps are refused by name."""
        if not isinstance(idx, tuple):
            idx = (idx,)
        for i in idx:
            if i is None or isinstance(i, (bool, list, torch.Tensor)) or not (i is Ellipsis or isinstance(i, (int, slice)) or hasattr(i, "__index__")):
                self._refuse(f"index of type {type(i).__name__}")
        nd = self.ndim
        if idx.count(Ellipsis) > 1:
            self._refuse("more than one Ellipsis in an index")
        if Ellipsis in idx:
            p = idx.index(Ellipsis)
            idx = idx[:p] + (slice(None),) * (nd - (len(idx) - 1)) + idx[p + 1:]
        if len(idx) > nd:
            self._refuse(f"{len(idx)} indices for a {nd}-dim object")
        idx = idx + (slice(None),) * (nd - len(idx))
        n_lead = nd - 3
        cur, d = self, 0                                                                # d: the position in `cur` of the next lead dim
        for i in idx[:n_lead]:
            if isinstance(i, slice):
                cur = cur._slice(d, i.start, i.stop, i.step); d += 1
            else:
                cur = cur._select(d, int(i))                                            # the dim is dropped: d stays
        for k, i in enumerate(idx[n_lead + 1:]):                                        # the pair-column and bin axes: whole-axis only
            size = int(self.shape[n_lead + 1 + k])
            if not (isinstance(i, slice) and (i.step is None or int(i.step) == 1) and slice(i.start, i.stop).indices(size)[:2] == (0, size)):
                cur._refuse(f"index {i!r} on dim {n_lead + 1 + k} (a pair-column / bin axis)")
        i = idx[n_lead]
        row_dim = cur.ndim - 3
        if isinstance(i, slice):
            if (i.step is None or int(i.step) == 1) and slice(i.start, i.stop).indices(int(cur.shape[row_dim]))[:2] == (0, int(cur.shape[row_dim])) and (i.start is None and i.stop is None):
                return cur                                                              # `[:]` on the row dim: still lazy
            return cur._slice(row_dim, i.start, i.stop, i.step)
        return cur._select(row_dim, int(i))

    # the Python-level surface (torch routes every Tensor method / function / property read on a subclass through __torch_function__)
    _TF_META = ("shape", "dtype", "device", "ndim", "is_cuda", "layout", "requires_grad", "is_sparse", "is_meta", "is_quantized", "is_leaf", "grad_fn")   # property reads: no data
    _TF_META_METHODS = ("size", "dim", "ndimension", "numel", "nelement", "__len__", "is_floating_point", "is_complex", "get_device", "element_size", "is_contiguous", "type")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        flat = [a for a in args if isinstance(a, RowBlockLogits)] + [b for a in args if isinstance(a, (list, tuple)) for b in a if isinstance(b, RowBlockLogits)]
        self = flat[0] if flat else next((v for v in kwargs.values() if isinstance(v, RowBlockLogits)), None)
        qual = getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or repr(func)
        if self is None:
            raise RuntimeError(f"RowBlockLogits: {qual} reached RowBlockLogits.__torch_function__ without a RowBlockLogits argument")
        if any(func == getattr(torch.Tensor, p).__get__ for p in cls._TF_META if hasattr(torch.Tensor, p)) or func in tuple(getattr(torch.Tensor, m) for m in cls._TF_META_METHODS if hasattr(torch.Tensor, m)):
            with torch._C.DisableTorchFunctionSubclass():                              # metadata reads: the wrapper's own sizes / dtype / device
                return func(*args, **kwargs)
        if func is torch.Tensor.__getitem__:
            return args[0].__getitem__(args[1])
        if func in (torch.Tensor.reshape, torch.Tensor.view, torch.reshape):
            shape = args[1:] if func is not torch.reshape else (args[1],)
            if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
                shape = tuple(shape[0])
            if "shape" in kwargs:
                shape = tuple(kwargs["shape"])
            return self._regroup(shape)
        if func is torch.Tensor.to:
            return self._cast(*args[1:], **{k: v for k, v in kwargs.items() if k in ("dtype", "device")})
        if func in (torch.Tensor.float, torch.Tensor.half, torch.Tensor.bfloat16, torch.Tensor.double):
            return self._cast({torch.Tensor.float: torch.float32, torch.Tensor.half: torch.float16, torch.Tensor.bfloat16: torch.bfloat16, torch.Tensor.double: torch.float64}[func])
        if func is torch.Tensor.cpu:
            return self._cast(torch.device("cpu"))
        if func is torch.Tensor.cuda:
            dev = kwargs.get("device", args[1] if len(args) > 1 else None)
            return self._cast(torch.device("cuda") if dev is None else (torch.device("cuda", dev) if isinstance(dev, int) else torch.device(dev)))
        if func in (torch.Tensor.detach, torch.Tensor.clone, torch.Tensor.contiguous, torch.clone, torch.detach):
            return self._view()                                                         # the same lazy object (nothing to copy, nothing to lay out)
        self._refuse(f"unsupported op {qual}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """The backstop below the Python API (a call that arrives as an aten op with torch-function handling off): the same supported ops through
        the same implementations, everything else refused by name."""
        kwargs = kwargs or {}
        self = next((a for a in args if isinstance(a, RowBlockLogits)), None) or next((b for a in args if isinstance(a, (list, tuple)) for b in a if isinstance(b, RowBlockLogits)), None)
        if self is None:
            raise RuntimeError(f"RowBlockLogits: {func} reached RowBlockLogits.__torch_dispatch__ without a RowBlockLogits argument")
        aten = torch.ops.aten
        if func in (aten.select.int,):
            return self._select(int(args[1]), int(args[2]))
        if func in (aten.slice.Tensor,):
            return self._slice(int(args[1]) if len(args) > 1 else 0, args[2] if len(args) > 2 else None, args[3] if len(args) > 3 else None, args[4] if len(args) > 4 else 1)
        if func in (aten.view.default, aten.reshape.default, aten._unsafe_view.default, aten._reshape_alias.default):
            return self._regroup(list(args[1]))
        if func in (aten.alias.default, aten.detach.default, aten.clone.default, aten.lift_fresh.default):
            return self._view()
        if func in (aten._to_copy.default,):
            return self._cast(**{k: v for k, v in kwargs.items() if k in ("dtype", "device") and v is not None})
        self._refuse(f"unsupported op {func}")


def _forward_factory(symmetric: bool, name: str):
    def forward(self, zij, apply_per_sample: bool = False):
        return RowBlockLogits.of(zij, self, symmetric=symmetric, name=name,
                                 autocast=(torch.is_autocast_enabled(), torch.get_autocast_dtype("cuda") if hasattr(torch, "get_autocast_dtype") else None))
    forward.__name__ = f"{name}_rowblock_forward"
    forward._of3_confhead = True
    return forward


def install() -> dict:
    """Replace the two heads' forward (class level; idempotent). Returns the record."""
    from openfold3.core.model.heads import prediction_heads as PH
    with _LOCK:
        if STATE["installed"]:
            return dict(STATE)
        _ORIG["pae"] = PH.PredictedAlignedErrorHead.forward
        _ORIG["pde"] = PH.PredictedDistanceErrorHead.forward
        PH.PredictedAlignedErrorHead.forward = _forward_factory(False, "pae")
        PH.PredictedDistanceErrorHead.forward = _forward_factory(True, "pde")
        STATE["installed"] = True
        STATE["sites"] = ["openfold3.core.model.heads.prediction_heads.PredictedAlignedErrorHead.forward", "openfold3.core.model.heads.prediction_heads.PredictedDistanceErrorHead.forward"]
    _log("installed: PredictedAlignedErrorHead.forward / PredictedDistanceErrorHead.forward -> RowBlockLogits (the head evaluated per scorer row block)")
    return dict(STATE)


def installed() -> Optional[bool]:
    """The probe: True when the heads' forward is this module's, False when the heads module is imported and it is not, None before the import."""
    PH = sys.modules.get("openfold3.core.model.heads.prediction_heads")
    if PH is None:
        return None
    return bool(getattr(PH.PredictedAlignedErrorHead.forward, "_of3_confhead", False) and getattr(PH.PredictedDistanceErrorHead.forward, "_of3_confhead", False))


def census() -> dict:
    with _LOCK:
        return dict(STATE)