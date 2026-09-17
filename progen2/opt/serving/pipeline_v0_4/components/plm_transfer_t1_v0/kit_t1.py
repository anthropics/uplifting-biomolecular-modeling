"""kit_t1: two exact levers on the stock ProGen2 decode step, composable by level.
  t3   : resident rotary tables — the stock function's OWN sin/cos computed once on the device for max_slots positions,
         sliced per call (values identical by construction: elementwise outer product on the CPU + device sin/cos per element).
  t3s  : + static KV cache — K/V slot buffers per (layer, batch size) at max_slots positions (dtypes = the stock's own key/value
         dtypes for the size's regime: K in float32 as the stock rotary hands it over, V in the model's); the per-step torch.cat is
         replaced by a slot write + a view over [0, t]; attention kernels see the same (m, n, k, ld). A batch size's slots are
         allocated when the caller asks for them (allocate(model, B): progen2_decode.py does, for the batch sizes of the work in front
         of the model and for a unit's batch size on first sight) — nothing is sized to the card; kv_bytes_per_sample(model)
         is the arithmetic of one sample's slots, release(keep) drops the slots of the batch sizes not kept.
Every decode step runs the stock's eager kernels; the sampler (warpers, softmax, multinomial) stays the stock generate() loop on the stock's philox stream.
Every patch is a module-global / class-attribute patch of the STOCK module (models.progen.modeling_progen); the originals are kept in _ORIG.
"""
import torch
from models.progen import modeling_progen as mp

_ORIG = dict(fixed_pos_embedding=mp.fixed_pos_embedding, apply_rotary_pos_emb=mp.apply_rotary_pos_emb, attn_forward=mp.ProGenAttention.forward)

class _State:
    def __init__(self):
        self.level = None; self.max_slots = None; self.rot = {}; self.rot_il = {}; self.device = "cuda:0"
        self.kv = {}          # (id(attn_module), B) -> (kc, vc)
        self.batches = set()  # the batch sizes whose slots allocate() made (or adopted) for every layer
        self.stats = dict(rot_hits=0, rot_builds=0, allocations=0)
S = _State()

# ---------------- t3: resident rotary tables ----------------
def _rot_tables(x, dim, seq_len):
    key = (str(x.device), dim)
    ent = S.rot.get(key)
    if ent is None or ent[0].shape[0] < seq_len:
        L = max(seq_len, S.max_slots)
        sin, cos = _ORIG["fixed_pos_embedding"](x, 1, seq_len=L)         # the stock function, once, for L positions
        S.rot[key] = (sin, cos)
        S.rot_il[key] = (sin.repeat_interleave(2, -1), cos.repeat_interleave(2, -1))
        S.stats["rot_builds"] += 1
        ent = S.rot[key]
    S.stats["rot_hits"] += 1
    return ent

def fixed_pos_embedding_cached(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    sin, cos = _rot_tables(x, dim, seq_len)
    return sin[:seq_len], cos[:seq_len]

def apply_rotary_pos_emb_cached(x, sincos, offset=0):
    dim = x.shape[-1]
    key = (str(x.device), dim)
    sin_il, cos_il = S.rot_il[key]
    n = x.shape[1]
    sin = sin_il[None, offset: n + offset, None, :]
    cos = cos_il[None, offset: n + offset, None, :]
    return (x * cos) + (mp.rotate_every_two(x) * sin)

# ---------------- t3s: static KV cache ----------------
def _static_kv(attn, B, H, hd, kdt, vdt, device):
    key = (id(attn), B)
    ent = S.kv.get(key)
    if ent is None:
        kc = torch.zeros((B, H, S.max_slots, hd), dtype=kdt, device=device)
        vc = torch.zeros((B, H, S.max_slots, hd), dtype=vdt, device=device)
        S.kv[key] = (kc, vc); ent = S.kv[key]
    return ent

def attn_forward_static(self, hidden_states, attention_mask=None, layer_past=None, head_mask=None, use_cache=False, output_attentions=False):
    qkv = self.qkv_proj(hidden_states)
    mp_num = 8
    qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))
    local_dim = self.head_dim * self.num_attention_heads // mp_num
    query, value, key = torch.split(qkv_split, local_dim, dim=-1)
    query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
    key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)
    value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
    value = value.permute(0, 2, 1, 3)
    seq_len = key.shape[1]
    offset = 0
    if layer_past is not None:
        offset = layer_past[0].shape[-2]
        seq_len += offset
    if self.rotary_dim is not None:
        k_rot = key[:, :, :, : self.rotary_dim]
        k_pass = key[:, :, :, self.rotary_dim :]
        q_rot = query[:, :, :, : self.rotary_dim]
        q_pass = query[:, :, :, self.rotary_dim :]
        sincos = mp.fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
        k_rot = mp.apply_rotary_pos_emb(k_rot, sincos, offset=offset)
        q_rot = mp.apply_rotary_pos_emb(q_rot, sincos, offset=offset)
        key = torch.cat([k_rot, k_pass], dim=-1)
        query = torch.cat([q_rot, q_pass], dim=-1)
    else:
        sincos = mp.fixed_pos_embedding(key, 1, seq_len=seq_len)
        key = mp.apply_rotary_pos_emb(key, sincos, offset=offset)
        query = mp.apply_rotary_pos_emb(query, sincos, offset=offset)
    key = key.permute(0, 2, 1, 3)
    query = query.permute(0, 2, 1, 3)
    B, H, T, hd = key.shape
    assert offset + T <= S.max_slots, f"static KV cache overflow: offset {offset} + T {T} > max_slots {S.max_slots}"
    kc, vc = _static_kv(self, B, H, hd, KEY_SLOT_DTYPE, self.qkv_proj.weight.dtype, key.device)   # the slot dtypes are the model's regime (K float32 off the
    # float32 rotary tables, V the parameter dtype), never the dtypes of the call that happens to size them first: sample.py's --sanity pass runs under autocast and
    # would otherwise leave a float32-parameter size's batch-1 V slots in half for the fp32 sample() units that follow (allocate() sizes them the same way)
    kc[:, :, offset: offset + T].copy_(key)
    vc[:, :, offset: offset + T].copy_(value)
    if layer_past is None:
        k_use, v_use = key, value                       # the stock's own prefill tensors
    else:
        k_use, v_use = kc[:, :, : offset + T], vc[:, :, : offset + T]
    if use_cache is True:
        present = (kc[:, :, : offset + T], vc[:, :, : offset + T])
    else:
        present = None
    attn_output, attn_weights = self._attn(query, k_use, v_use, attention_mask, head_mask)
    attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)
    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)
    return outputs


def install(model, level="t3s", max_slots=1024, batches=(), device="cuda:0"):
    """Patch the stock module at `level`; for t3s also allocate the static K/V slots of the batch sizes in `batches` (allocate(); none by
    default: a batch size is allocated when the caller asks for it)."""
    assert level in ("t3", "t3s"), level
    S.level = level; S.max_slots = max_slots; S.device = str(device)
    mp.fixed_pos_embedding = fixed_pos_embedding_cached
    mp.apply_rotary_pos_emb = apply_rotary_pos_emb_cached
    info = dict(level=level, max_slots=max_slots)
    if level == "t3s":
        import time
        mp.ProGenAttention.forward = attn_forward_static
        t0 = time.perf_counter()
        for B in batches:
            allocate(model, int(B))
        info["alloc_s"] = time.perf_counter() - t0
        info["batches"] = allocated_batches()
        info["kv_bytes_per_sample"] = kv_bytes_per_sample(model)
    info["stats"] = dict(S.stats)
    return info


# ---------------- the static K/V slots of a batch size: arithmetic, allocation, release ----------------
KEY_SLOT_DTYPE = torch.float32   # the dtype the stock hands the key cache: apply_rotary_pos_emb multiplies by float32 sin/cos tables, so K is float32 whatever the model's dtype; V keeps the model's


def kv_bytes_per_sample(model):
    """Bytes of static K/V ONE sample (one unit of batch) holds at S.max_slots positions: per layer a (n_head, max_slots, head_dim) K slot
    buffer in KEY_SLOT_DTYPE and a V buffer in the model's parameter dtype — the shapes and dtypes _static_kv allocates (n_head * head_dim =
    n_embd). The estimate a caller checks against the free device memory BEFORE allocate(); kv_bytes_held() is the count after."""
    cfg = model.config
    vbytes = next(model.parameters()).element_size()
    return int(cfg.n_layer) * int(S.max_slots) * int(cfg.n_embd) * (torch.empty((), dtype=KEY_SLOT_DTYPE).element_size() + vbytes)


def kv_bytes_held(B=None):
    """Bytes of static K/V allocated now (every batch size, or batch size B only), counted from the buffers themselves."""
    return sum(kc.numel() * kc.element_size() + vc.numel() * vc.element_size() for (_, b), (kc, vc) in S.kv.items() if B is None or b == B)


def allocated_batches():
    """The batch sizes whose static K/V slots are held, ascending."""
    return sorted(S.batches | {b for (_, b) in S.kv})


def allocated(B):
    """True when batch size B has its static K/V slots for every layer by allocate(); a batch size whose slots the attention lever made on its
    own (the load's sanity forward at batch 1) counts once allocate() has adopted them."""
    return int(B) in S.batches


def allocate(model, B):
    """Allocate the static K/V slots of batch size B for every layer: one eager prefill (context length 1) and one eager decode step through the
    stock forward, which sizes each layer's slot buffers in the stock's own dtypes (_static_kv) and warms the kernels at B. Idempotent; returns
    True when it allocated. The caller checks kv_bytes_per_sample(model) * B against the free memory first."""
    B = int(B)
    if B in S.batches:
        return False
    assert S.level == "t3s", f"allocate: the static K/V lever is not installed (level {S.level})"
    device = next(model.parameters()).device
    if not all((id(blk.attn), B) in S.kv for blk in model.transformer.h):   # a batch size whose slots a forward already made (the sanity pass at batch 1) is only adopted
        with torch.no_grad():
            ctx = torch.full((B, 1), 3, dtype=torch.long, device=device)
            out = model(input_ids=ctx, use_cache=True, return_dict=True)
            nxt = torch.full((B, 1), 5, dtype=torch.long, device=device)
            model(input_ids=nxt, past_key_values=out.past_key_values, attention_mask=torch.ones((B, 2), dtype=torch.long, device=device), use_cache=True, return_dict=True)
            del out
    if device.type == "cuda":
        torch.cuda.synchronize()
    S.batches.add(B)
    S.stats["allocations"] = S.stats.get("allocations", 0) + 1
    return True


def release(keep=()):
    """Drop the static K/V slots of every batch size NOT in `keep` (an exact lever: memory only; a dropped batch size is allocated again when
    asked for). Returns the counts."""
    keep = {int(b) for b in keep}
    n_kv = 0; dropped = set()
    for key in [k for k in S.kv if k[1] not in keep]:
        del S.kv[key]; n_kv += 1; dropped.add(key[1])
    for b in [b for b in S.batches if b not in keep]:
        S.batches.discard(b); dropped.add(b)
    if n_kv or dropped:
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"evicted_kv_entries": n_kv, "released_batches": sorted(dropped)}


def stats():
    return dict(S.stats)


def rotary_table_check(model, max_len, device="cuda:0"):
    """For every seq_len in [1, max_len]: the stock function's fresh (sin, cos) vs the cached slices — bit-pattern counts."""
    from compare.bit_equal import bit_equal
    attn = model.transformer.h[0].attn
    dim = attn.rotary_dim
    x = torch.zeros((1, 1, 1, dim), device=device, dtype=next(model.parameters()).dtype)
    ok = 0; bad = []
    for n in range(1, max_len + 1):
        s0, c0 = _ORIG["fixed_pos_embedding"](x, 1, seq_len=n)
        s1, c1 = fixed_pos_embedding_cached(x, 1, seq_len=n)
        if bit_equal(s0, s1) and bit_equal(c0, c1):
            ok += 1
        else:
            bad.append(n)
    return dict(n=max_len, equal=ok, mismatch_lengths=bad[:20], n_mismatch=len(bad))
