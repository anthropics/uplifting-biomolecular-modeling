"""Kit ``tokenmemo`` for Profluent-E1 — host side: `E1BatchPreparer.prepare_multiseq` without re-tokenising a context for every row.

Upstream tokenises every row string of every batch from characters (`prepare_singleseq`: a Python dict lookup per residue) and decodes
the context's token ids back to a string (the KV cache key) — per ROW. In retrieval-augmented scoring every row of a job is
"<context members>,<query>" with few distinct contexts and many rows per context, so the same context is tokenised and decoded
once per row — and for the cached batches upstream then slices the context tokens away again (KVCache.before_forward).

LEVER (exact: the same integer tensors, the same strings, the same exceptions):
    tokenmemo   a row with a context ("," in it) = memo(context part) ++ memo(query member): the context part's input ids, labels
                (masked to <pad> unless preserve_context_labels, as the stock does after concatenation), within-sequence and global
                position ids, sequence ids, the decoded context string and the global-position offset are computed ONCE per (context
                string, preparer configuration) with the stock's own `prepare_singleseq` calls in the stock's order (so a miss raises
                what the stock raises), the query member once per string; the row's tensors are then two concatenations each. Rows
                without a context take the stock call untouched. The memo lives on the preparer instance, keyed with the configuration
                fields the result depends on (remove_X_tokens, max_num_positions_within_seq, preserve_context_labels), bounded in size.
Counters: rows_stock (no context), rows_memo, ctx_miss, ctx_hit, query_miss, query_hit.
"""
from __future__ import annotations

import collections

import torch

KIT = "tokenmemo"
LEVERS = ("tokenmemo",)
MAX_CONTEXTS = 64
MAX_QUERIES = 1 << 16
CTR = collections.Counter()
_R: dict = {"installed": False, "orig": None}


class TokenmemoRefused(RuntimeError):
    pass


def _cfg_key(self):
    c = self.data_prep_config
    return (bool(c.remove_X_tokens), int(c.max_num_positions_within_seq), bool(self.preserve_context_labels), int(self.pad_token_id))


def _memo(self):
    m = self.__dict__.get("_kit_tokenmemo")
    if m is None:
        m = {"ctx": collections.OrderedDict(), "query": collections.OrderedDict()}
        self.__dict__["_kit_tokenmemo"] = m
    return m


def _context_entry(self, ctx_str: str, members: list):
    """Everything of the row that depends on the context part only — computed with the stock's per-member calls, in order."""
    encs = [_R["orig_single"](self, s) for s in members]                      # raises what the stock raises, member by member
    num_tokens = [len(x["input_ids"]) for x in encs]
    input_ids = torch.cat([x["input_ids"] for x in encs])
    within = torch.cat([x["position_ids"] for x in encs])
    global_ids, off = [], 0
    for x in encs:                                                             # the stock's offset recurrence, verbatim
        global_ids.append(x["position_ids"] + off)
        off = max(off, x["position_ids"].max().item() + off + 1)
    global_ids = torch.cat(global_ids)
    sequence_ids = torch.repeat_interleave(torch.tensor(num_tokens))
    context_len = sum(num_tokens)
    context = self.tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
    if self.preserve_context_labels:
        labels = input_ids.clone()
    else:
        labels = torch.full_like(input_ids, self.pad_token_id)
    return {"input_ids": input_ids, "labels": labels, "within": within, "global": global_ids, "sequence_ids": sequence_ids,
            "n_seq": len(members), "offset": off, "context_len": context_len, "context": context}


def _prepare_multiseq_memo(self, sequence: str):
    if "," not in sequence:
        CTR["rows_stock"] += 1
        return _R["orig"](self, sequence)
    single_sequences = sequence.split(",")
    if len(single_sequences) > self.data_prep_config.max_num_sequences:      # the stock's first check, its message
        raise ValueError(
            f"Number of sequences {len(single_sequences)} exceeds max number of sequences {self.data_prep_config.max_num_sequences}"
            " in the provided multi-sequence instance. Please remove some homologous sequences before trying again."
        )
    memo = _memo(self)
    ck = _cfg_key(self)
    ctx_str = sequence.rsplit(",", 1)[0]
    ent = memo["ctx"].get((ck, ctx_str))
    if ent is None:
        ent = _context_entry(self, ctx_str, single_sequences[:-1])
        memo["ctx"][(ck, ctx_str)] = ent
        if len(memo["ctx"]) > MAX_CONTEXTS:
            memo["ctx"].popitem(last=False)
        CTR["ctx_miss"] += 1
    else:
        CTR["ctx_hit"] += 1
    q_str = single_sequences[-1]
    q = memo["query"].get((ck, q_str))
    if q is None:
        q = _R["orig_single"](self, q_str)
        memo["query"][(ck, q_str)] = q
        if len(memo["query"]) > MAX_QUERIES:
            memo["query"].popitem(last=False)
        CTR["query_miss"] += 1
    else:
        CTR["query_hit"] += 1
    q_tokens, q_pos = q["input_ids"], q["position_ids"]
    input_ids = torch.cat([ent["input_ids"], q_tokens])
    labels = torch.cat([ent["labels"], q["labels"]])
    within_seq_position_ids = torch.cat([ent["within"], q_pos])
    global_position_ids = torch.cat([ent["global"], q_pos + ent["offset"]])
    sequence_ids = torch.cat([ent["sequence_ids"], torch.full((len(q_tokens),), ent["n_seq"], dtype=ent["sequence_ids"].dtype)])
    context_len = ent["context_len"]
    assert (
        input_ids.shape == sequence_ids.shape == within_seq_position_ids.shape == global_position_ids.shape == labels.shape
    ), "Input ids, sequence ids, within seq position ids, global position ids, and labels must have the same shape"
    assert input_ids.shape[0] >= context_len, "Input ids must have at least as many tokens as the context length"
    CTR["rows_memo"] += 1
    return {
        "input_ids": input_ids,
        "sequence_ids": sequence_ids,
        "within_seq_position_ids": within_seq_position_ids,
        "global_position_ids": global_position_ids,
        "labels": labels,
        "context": ent["context"],
        "context_len": context_len,
    }


def install() -> dict:
    from E1.batch_preparer import E1BatchPreparer as P
    if _R["installed"]:
        raise TokenmemoRefused(f"kit {KIT}: already installed in this process")
    _R["orig"] = P.prepare_multiseq
    _R["orig_single"] = P.prepare_singleseq
    P.prepare_multiseq = _prepare_multiseq_memo
    _R["installed"] = True
    CTR.clear()
    return {"kit": KIT, "levers": list(LEVERS), "max_contexts": MAX_CONTEXTS, "max_queries": MAX_QUERIES}


def uninstall() -> None:
    if not _R["installed"]:
        return
    from E1.batch_preparer import E1BatchPreparer as P
    P.prepare_multiseq = _R["orig"]
    _R.update(installed=False, orig=None)


def in_force() -> bool:
    from E1.batch_preparer import E1BatchPreparer as P
    return bool(_R["installed"]) and P.prepare_multiseq is _prepare_multiseq_memo


def counters() -> dict:
    return {k: int(v) for k, v in CTR.items()}


def is_pristine() -> dict:
    from E1.batch_preparer import E1BatchPreparer as P
    return {"prepare_multiseq": P.prepare_multiseq is not _prepare_multiseq_memo}
