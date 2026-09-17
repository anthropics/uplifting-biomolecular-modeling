"""engines.enformer.kits.v0 — the trunk-graph wrapper of the exact kit.

``KitV0`` holds a model whose modules ``enformer_fastkit.FastKit`` has patched, one ``TrunkGraph`` per batch size (a CUDA graph of
``model(x, return_only_embeddings=True)`` over static buffers, captured at attach for the requested sizes and at first use for any other),
and runs the STOCK heads eagerly on the replayed embedding — the single-window and the batched head arithmetic kept distinct exactly as the
stock keeps them (a (896, 3072) view for one window, (B, 896, 3072) for a batch; the two take different GEMM paths in torch and differ by
an ulp, so the stock's own choice is reproduced, not unified).

Guards, all refusals by name: every lazily allocated device buffer a captured graph reads (the attention layers' cached ``rel_k``, the
graphs' static input / output, the parameters' device) is recorded at capture and re-checked before each replay — a change would be a
replay over freed memory, so it raises; moving or casting the model (``.to() / .cuda() / .half() …``) while graphs exist is refused for the
same reason; the module patches the levers installed are re-checkable (``patches_effective``: the method snapshot taken across the attach).
"""
from __future__ import annotations

import gc
import hashlib

WARMUPS = 2                     # forwards on a side stream before capture (cuDNN / cuBLAS handles and workspaces must exist before a graph is recorded)
SEQ_LEN_REF = 196_608           # the model's input length; the capture footprint below is stated per window of this length and scaled linearly
CAPTURE_GIB_PER_WINDOW = 4.0    # device memory a trunk capture occupies per window of the batch at SEQ_LEN_REF (warm-up activations + the graph's private
CAPTURE_GIB_FIXED = 4.0         # pool; measured 3.2–4.3 GiB per window at batch 4–16 on H100 and A100) plus a fixed margin. A capture is attempted only when
                                # this fits in the device's free memory; a larger batch runs the same patched modules without replay (UNGRAPHED by the rule)


def capture_fits(batch: int, seq_len: int, device):
    """(fits, need_gib, free_gib): whether a trunk capture at ``batch`` fits in the device's free memory by the stated footprint rule."""
    import torch
    free, _total = torch.cuda.mem_get_info(device)
    free += max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))   # the allocator's cached, unoccupied blocks are returned before a capture
    need = (CAPTURE_GIB_PER_WINDOW * batch * (seq_len / SEQ_LEN_REF) + CAPTURE_GIB_FIXED) * 2**30
    return need <= free, need / 2**30, free / 2**30


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class TrunkGraph:
    """CUDA-graph replay of the patched trunk for one batch size: ``model(x, return_only_embeddings=True)`` recorded over a static input
    buffer (WARMUPS forwards on a side stream first), replayed on the caller's stream -> the static (B, 896, 3072) embedding (valid until
    the next replay of this graph)."""

    def __init__(self, model, batch: int, seq_len: int, device):
        import torch
        self.static_in = torch.zeros(batch, seq_len, 4, device=device)
        s = torch.cuda.Stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(WARMUPS):
                model(self.static_in, return_only_embeddings=True)
        torch.cuda.current_stream(device).wait_stream(s)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()                    # the warm-up's activations sit unused in the allocator's cache; release them so the capture's private pool does not double the footprint
        self.graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self.graph), torch.no_grad():
                self.static_emb = model(self.static_in, return_only_embeddings=True)
        except BaseException:
            self.graph.reset()                      # a failed capture (out of memory near the card's limit) returns its pool at once
            del self.graph, self.static_in
            raise

    def __call__(self, x):
        self.static_in.copy_(x)
        self.graph.replay()
        return self.static_emb

    def release(self):
        """Tear the graph down now: the executable graph destroyed, its private pool handed back to the allocator (returned to the device by
        the next ``torch.cuda.empty_cache()``), the static tensors dropped — deterministic, never left to the collector's timing or a stray
        reference (a traceback frame, a caller's handle)."""
        g = self.__dict__.pop("graph", None)
        if g is not None:
            g.reset()
        self.static_in = self.static_emb = None


def method_snapshot(model) -> dict:
    """Every module's effective ``forward`` (instance override or class attribute) by module name; the levers' patches are the diff of two
    snapshots taken across the attach."""
    snap = {}
    if not hasattr(model, "named_modules"):
        return snap
    for name, m in model.named_modules():
        fwd = m.__dict__.get("forward", None)
        snap[name] = ("inst", id(fwd)) if fwd is not None else ("cls", type(m).__name__, id(type(m).__dict__.get("forward", type(m).forward)))
    return snap


#: the module classes each lever patches (a lever whose patch is absent after attach is a defect, raised by patches_effective)
LEVER_TARGETS = {"poscache": ("Attention",), "fused": ("AttentionPool", "Residual", "GELU", "BatchNorm1d", "Sequential"),
                 "xattn": ("Attention",), "graph": ()}
GRAPH_SCOPE = "trunk (model(x, return_only_embeddings=True)) replayed per batch size; the stock heads run eagerly on the embedding"


def capture_or_none(model, batch: int, seq_len: int, device):
    """(TrunkGraph, None) for ``batch``, or (None, reason) when the capture does not fit: by the footprint rule (capture_fits — no attempt is
    made, so nothing is left behind), or because the attempt itself ran out of device memory (torn down, its pool returned, before returning).
    A graph's private pool cannot release and re-take memory mid-call the way eager execution does, so near the card's limit a batch size the
    patched modules run eagerly does not capture; that size then runs the same modules without replay. Any other error propagates."""
    import torch
    fits, need, free = capture_fits(batch, seq_len, device)
    if not fits:
        return None, f"a capture at batch {batch} takes about {need:.0f} GiB by the kit's footprint rule and {free:.0f} GiB is free"
    try:
        return TrunkGraph(model, batch, seq_len, device), None
    except torch.cuda.OutOfMemoryError:
        pass                                        # the cleanup runs below, once the exception and its frames are released
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return None, f"the capture at batch {batch} ran out of device memory"


class KitV0:
    """The applied kit of one model: ``kit`` (the FastKit holding the module patches), the trunk graphs per batch size, the stock heads.
    ``lazy_capture``: a batch size without a graph is captured at its first ``embed`` (else refused by name). ``ungraphed``: the batch
    sizes whose capture ran out of memory — they run the same patched modules eagerly (same bytes; at such sizes replay saves nothing
    measurable, the forward being GPU-bound), and are not re-captured until the kit is removed and applied again."""

    MOVE_NAMES = ("to", "half", "float", "double", "bfloat16", "cuda", "cpu")

    def __init__(self, kit, model, shapes, seq_len: int, device, pre_snapshot: dict | None = None, levers=None):
        import torch
        self.kit, self.model, self.shapes, self.replays = kit, model, tuple(shapes), 0
        self.levers = tuple(levers or ("poscache", "fused", "graph", "xattn"))
        self.pre_snapshot = pre_snapshot or {}
        self.post_snapshot = method_snapshot(model)
        self.patched_methods = sorted(n for n, v in self.post_snapshot.items() if self.pre_snapshot.get(n) != v) if self.pre_snapshot else []
        self.effective_checks = 0
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.lazy_capture = False
        self.lazy_captures = []
        self.ungraphed = {}
        self.graphs = {}
        if "graph" in self.levers:
            for b in self.shapes:
                g, why = capture_or_none(model, b, seq_len, self.device)
                if g is None:
                    self.ungraphed[b] = why
                else:
                    self.graphs[b] = g
        self.pinned_buffers = self._buffer_state() if "graph" in self.levers else {}
        self._move_overrides = self._install_move_refusal() if "graph" in self.levers else {}

    # ------------------------------------------------------------------------------------------------ guards
    def _install_move_refusal(self) -> dict:
        saved = {}
        for name in self.MOVE_NAMES:
            if name in self.model.__dict__:
                saved[name] = self.model.__dict__[name]

            def _refuse(*a, _n=name, **k):
                raise RuntimeError(f"enformer kit: model.{_n}() after the CUDA-graph capture is refused — the graphs hold device addresses; "
                                   "remove the kit from the model first (enformer_opt.disable())")
            try:
                setattr(self.model, name, _refuse)
            except Exception:
                pass
        return {"installed": [n for n in self.MOVE_NAMES if n in self.model.__dict__], "saved": saved}

    def _restore_move_refusal(self):
        ov = self._move_overrides or {}
        for name in ov.get("installed", []):
            if name in ov.get("saved", {}):
                setattr(self.model, name, ov["saved"][name])
            else:
                self.model.__dict__.pop(name, None)
        self._move_overrides = {}

    def _buffer_state(self) -> dict:
        """data_ptr / shape / device of every lazily allocated buffer a graph reads: the attention layers' cached rel_k (poscache slot and
        xattn cache), the parameters' device, each graph's static buffers."""
        import torch
        st = {}
        if not hasattr(self.model, "named_modules"):
            return st
        for name, m in self.model.named_modules():
            c = getattr(m, "_fastkit_relk", None)
            if c is not None:
                st[f"{name}._fastkit_relk"] = (int(c[1].data_ptr()), tuple(c[1].shape), str(c[1].device), str(c[0]))
            xc = m.__dict__.get("_xattn_relk") if hasattr(m, "__dict__") else None
            if isinstance(xc, dict):
                for k, t in xc.items():
                    if hasattr(t, "data_ptr"):
                        st[f"{name}._xattn_relk[{k}]"] = (int(t.data_ptr()), tuple(t.shape), str(t.device))
        p0 = next(iter(self.model.parameters()), None) if hasattr(self.model, "parameters") else None
        if p0 is not None:
            st["parameters.device"] = (str(p0.device),)
        for b, g in self.graphs.items():
            for nm in ("static_in", "static_emb"):
                t = getattr(g, nm, None)
                if torch.is_tensor(t):
                    st[f"graph[{b}].{nm}"] = (int(t.data_ptr()), tuple(t.shape), str(t.device))
        return st

    def check_buffers(self):
        """Refuse (never replay) when a buffer recorded at capture changed since: a cached rel_k replaced, a parameter move, a re-made static
        buffer. Cache entries added after the capture (another window length's positional basis) are not read by the graphs and pass."""
        if not self.pinned_buffers:
            return
        now = self._buffer_state()
        changed = sorted(k for k in self.pinned_buffers if self.pinned_buffers[k] != now.get(k))   # a recorded buffer replaced, moved or gone; an entry ADDED since (a cache slot for another window length) is not read by the graphs
        if changed:
            raise RuntimeError(f"enformer kit: device buffers a captured graph reads changed since capture (refusing the replay): {changed[:6]}")

    # ------------------------------------------------------------------------------------------------ forward
    def embed(self, x):
        """The trunk embedding of a (B, seq_len, 4) float32 input by graph replay -> the graph's static (B, 896, 3072) buffer (copy it before
        the next replay at this batch size). A batch size without a graph is captured now when ``lazy_capture`` is set, else refused."""
        import torch
        if x.dim() != 3:
            raise RuntimeError(f"enformer kit: input rank {x.dim()} — embed() takes (B, {self.seq_len}, 4)")
        b = int(x.shape[0])
        if "graph" not in self.levers or b in self.ungraphed or int(x.shape[1]) != self.seq_len:   # no graph lever, an ungraphed size, or a window length other than the graphs': the patched modules, eager
            with torch.no_grad():
                return self.model(x, return_only_embeddings=True)
        if b not in self.graphs:
            if not self.lazy_capture:
                raise RuntimeError(f"enformer kit: no captured CUDA graph for batch {b} (captured: {sorted(self.graphs)})")
            g, why = capture_or_none(self.model, b, self.seq_len, self.device)
            if g is None:
                self.ungraphed[b] = why
                with torch.no_grad():
                    return self.model(x, return_only_embeddings=True)
            self.graphs[b] = g
            self.lazy_captures.append(b)
            self.pinned_buffers = self._buffer_state()                    # the new graph's buffers join the record
        self.check_buffers()
        with torch.no_grad():
            emb = self.graphs[b](x)
        self.replays += 1
        if emb.shape[-1] == 3072 and tuple(emb.stride()[1:]) != (1, emb.shape[1]):   # the stock heads read final_pointwise's transposed view (strides (.., 1, 896)); a copy would change their GEMM path
            raise RuntimeError(f"enformer kit: embedding strides {tuple(emb.stride())} are not the stock's transposed view")
        return emb

    def forward_device(self, x):
        """``model(x)`` for a (B, seq_len, 4) input on the device -> {head: (B, 896, n)}: graph replay of the trunk, then the stock heads —
        for B == 1 on the 2-D (896, 3072) view (the stock's arithmetic for a single rank-2 window), else on the batch."""
        import torch
        b = int(x.shape[0])
        try:
            emb = self.embed(x)
            heads = self.model._heads
            with torch.no_grad():
                if b == 1:
                    return {h: heads[h](emb[0]).unsqueeze(0) for h in heads}
                return {h: heads[h](emb) for h in heads}
        except torch.cuda.OutOfMemoryError:
            if b not in self.graphs:
                raise                                                   # out of memory without a graph at this size: the card's limit, as for stock
            emb = None
            self.graphs.pop(b).release()                                # the graph's pool left no room for the rest of the call: this size runs ungraphed from now on
            self.ungraphed[b] = f"device memory ran out beside the graph's pool at batch {b}"
            self.pinned_buffers = {}
        gc.collect()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        self.pinned_buffers = self._buffer_state()
        return self.forward_device(x)                                   # eagerly now (b is ungraphed); a genuine out-of-memory propagates from here

    # ------------------------------------------------------------------------------------------------ records
    def patches_effective(self) -> dict:
        """Every method the attach patched is still the patched object, and every lever with patch targets patched at least one module of a
        target class; raises by name otherwise."""
        now = method_snapshot(self.model)
        lost = [n for n in self.patched_methods if now.get(n) != self.post_snapshot.get(n)]
        missing = []
        if self.pre_snapshot:
            for lv in self.levers:
                targets = LEVER_TARGETS.get(lv, ())
                if targets and not any(self.post_snapshot[n][1] in targets if self.post_snapshot[n][0] == "cls" else True for n in self.patched_methods):
                    missing.append(lv)
        self.effective_checks += 1
        if lost or missing:
            raise RuntimeError(f"enformer kit: lever patches not in effect — lost {lost[:5]} missing {missing}")
        return {"n_patched_methods": len(self.patched_methods), "lost": lost, "levers_without_patch": missing, "checks": self.effective_checks}

    def stamp(self) -> dict:
        st = dict(self.kit.stamp() if hasattr(self.kit, "stamp") else {})
        st.update(graph_scope=GRAPH_SCOPE, captured_shapes=sorted(self.graphs), lazy_captures=list(self.lazy_captures), ungraphed=dict(self.ungraphed), replays=self.replays,
                  n_patched_methods=len(self.patched_methods), patched_methods=self.patched_methods[:64],
                  move_refused=(self._move_overrides or {}).get("installed", []), n_pinned_buffers=len(self.pinned_buffers))
        return st

    def release_graphs(self):
        """Release every trunk graph (pools returned now); the next call at each batch size captures again. The module patches stay."""
        import torch
        for g in list(self.graphs.values()):
            g.release()
        self.graphs = {}
        self.ungraphed = {}
        self.pinned_buffers = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def close(self):
        """Restore the stock modules; release every trunk graph's pool now (TrunkGraph.release, then empty_cache), not at collection time."""
        import torch
        for g in list(self.graphs.values()):
            g.release()
        self.graphs = {}
        self.ungraphed = {}
        self.pinned_buffers = {}
        self._restore_move_refusal()
        out = self.kit.close() if hasattr(self.kit, "close") else None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return out
