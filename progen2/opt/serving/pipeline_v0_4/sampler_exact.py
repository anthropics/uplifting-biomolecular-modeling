"""Lever `sampler_exact`: the sampler + loop bookkeeping of transformers 4.16.2
generate(do_sample=True) — the eager part of every decode step around the forward — with the SAME float ops in the SAME
order on the SAME philox stream (torch.multinomial is the only RNG consumer, called exactly as the stock does), and the INTEGER
bookkeeping fused or cached:

  stock (generation_utils.sample + ProGenForCausalLM.prepare_inputs_for_generation + _update_model_kwargs_for_generation), per step:
    position_ids  = attention_mask.long().cumsum(-1) - 1; masked_fill_(mask == 0, 1); [:, -1]     → 5 launches   → a static arange buffer (0)
    attention_mask = cat([mask, ones(B, 1)])                                                      → 2 launches   → a static ones buffer, sliced (0)
    next_tokens   = next_tokens * unfinished + pad * (1 - unfinished)                              → 4 launches   → torch.where (1)
    input_ids     = cat([input_ids, next_tokens[:, None]])                                         → 1 launch     → out[:, cur] = next_tokens (1)
    unfinished    = unfinished.mul((next_tokens != eos).long())                                    → 3 launches   → unfinished &= next_tokens != eos (2)
    stop          = unfinished.max() == 0  (a host sync)                                           → 2 + sync     → unfinished.any() (1 + sync)
    TopKLogitsWarper(50) on a 32-wide vocabulary: k = min(50, 32) = 32 → nothing removed          → 3 launches   → skipped iff k == vocab (0)
  unchanged (bit for bit, the same torch ops on the same tensors): TemperatureLogitsWarper, TopKLogitsWarper where it is live (the
  51,200-wide sizes), TopPLogitsWarper, softmax, torch.multinomial(probs, 1), the forward call (the stock forward, through whatever the decode component patched onto it).

Exactness by construction: every changed op is an integer/bool identity (the mask is all ones — asserted once per call; sample.py
never pads), and the RNG stream sees the same multinomial calls in the same order.  The check is the token tensor, bit for bit, +
the printed block vs the stock's model.generate on the same seed.  The stop step is the stock's (one host sync per
step remains — it is the exactness of the extent).
"""
import torch


def _skip_topk(warper, vocab):
    """TopKLogitsWarper is the identity iff min(max(k, min_keep), vocab) == vocab (nothing is < the minimum of everything)."""
    from transformers.generation_logits_process import TopKLogitsWarper
    if not isinstance(warper, TopKLogitsWarper):
        return False
    k = min(max(warper.top_k, warper.min_tokens_to_keep), vocab)
    return k == vocab


@torch.no_grad()
def generate_exact(model, input_ids, *, max_length, temperature, top_p, num_return_sequences, pad_token_id, eos_token_id=None, top_k=None):
    """model.generate(input_ids, do_sample=True, temperature=, max_length=, top_p=, num_return_sequences=, pad_token_id=) — the
    tokens tensor (B, <= max_length) bit for bit, cheaper per step."""
    device = input_ids.device
    eos = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    B0, T0 = input_ids.shape
    # generate(): the attention mask (all ones: pad not in the inputs — sample.py's contexts never contain <|pad|>), then the
    # num_return_sequences expansion (index_select with the interleaved index, as _expand_inputs_for_generation does)
    assert not bool((input_ids == pad_token_id).any()), "pad token inside the context: the stock would build a non-trivial mask (not this lever's case)"
    idx = torch.arange(B0, device=device).view(-1, 1).repeat(1, num_return_sequences).view(-1)
    input_ids = input_ids.index_select(0, idx)
    B = input_ids.shape[0]
    warpers = model._get_logits_warper(top_k=top_k, top_p=top_p, temperature=temperature, num_beams=1)
    vocab = model.config.vocab_size
    warpers = [w for w in warpers if not _skip_topk(w, vocab)]
    # static buffers: the output tensor (pad after eos, as the stock's cat + pad arithmetic leaves it), the all-ones mask, the positions
    out = torch.full((B, max_length), pad_token_id, dtype=input_ids.dtype, device=device)
    out[:, :T0] = input_ids
    ones = torch.ones((B, max_length), dtype=torch.long, device=device)
    pos_all = torch.arange(max_length, device=device).unsqueeze(0).expand(B, max_length)
    unfinished = torch.ones(B, dtype=torch.bool, device=device)
    pad_vec = torch.full((B,), pad_token_id, dtype=input_ids.dtype, device=device)
    past = None
    cur = T0
    while True:
        if past is None:                       # the prefill: the stock's position_ids = cumsum(ones) - 1 = arange(T0)
            ids_in = out[:, :cur]; pos = pos_all[:, :cur]
        else:                                  # the decode step: the last token, position cur - 1
            ids_in = out[:, cur - 1:cur]; pos = pos_all[:, cur - 1:cur]
        outputs = model(input_ids=ids_in, past_key_values=past, attention_mask=ones[:, :cur], position_ids=pos, use_cache=True, return_dict=True)
        past = outputs.past_key_values
        scores = outputs.logits[:, -1, :]
        for w in warpers:                      # the stock's warper chain, the same objects, the same order
            scores = w(None, scores)
        probs = torch.nn.functional.softmax(scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)      # THE RNG draw — as the stock's
        next_tokens = torch.where(unfinished, next_tokens, pad_vec)   # == next * unf + pad * (1 - unf)
        out[:, cur] = next_tokens
        cur += 1
        unfinished &= next_tokens != eos       # == unfinished.mul((next_tokens != eos).long())
        if cur >= max_length or not bool(unfinished.any()):     # MaxLengthCriteria | unfinished.max() == 0 (the stock's stop, one sync)
            break
    return out[:, :cur]


def install(model):
    """Shadow model.generate on the instance with generate_exact for sample.py's one call form (do_sample=True, temperature, max_length,
    top_p, num_return_sequences, pad_token_id); any other form goes to the stock's generate.  Returns the restore function."""
    orig = model.generate

    def generate(input_ids, **kw):
        keys = set(kw)
        if kw.get("do_sample") is True and keys <= {"do_sample", "temperature", "max_length", "top_p", "num_return_sequences", "pad_token_id", "top_k"}:
            return generate_exact(model, input_ids, max_length=kw["max_length"], temperature=kw["temperature"], top_p=kw["top_p"],
                                  num_return_sequences=kw.get("num_return_sequences", 1), pad_token_id=kw["pad_token_id"], top_k=kw.get("top_k"))
        return orig(input_ids, **kw)

    model.generate = generate

    def restore():
        del model.generate
    return restore
