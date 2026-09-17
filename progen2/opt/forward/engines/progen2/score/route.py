"""The ProGen2 LIKELIHOOD route — the constants and the unit grammar of the authors' likelihood.py (stock/src/progen2/likelihood.py,
salesforce/progen @ c27a419c — this repo's own vendored copy) shared by the scoring kits: ``ll()`` L174-200 and main L285-299 —
the context string AS GIVEN (terminals included, e.g. '1'+seq+'2'), ``model(target, labels=target)`` on the 1-D ids under
autocast(fp16), shift, drop a trailing '1'/'2' target, logits[:, 5:30], -cross_entropy sum AND mean per direction (a forward per
call), the reversed STRING as the second direction, ll = .5*(lr+rl).
"""
from __future__ import annotations

FIRST_TOKEN, LAST_TOKEN = 5, 29                  # likelihood.py L194: the 25 amino-acid token ids A..Y (tokenizer.json ids 5..29)
N_AA = LAST_TOKEN - FIRST_TOKEN + 1
BOS_TOKEN, EOS_TOKEN = 3, 4                      # likelihood.py L185: the '1' / '2' terminal ids (the names are the file's own)
DEFAULT_CONTEXT = ('1MGHGVSRPPVVTLRPAVLDDCPVLWRWRNDPETRQASVDEREIPVDTHTRWFEETLKRFDRKLFIVSADGVDAGMVRLDIQDRDAAVSVNIAPEWRGRGVGPRALGCLSREAFGPLALLRM'
                   'SAVVKRENAASRIAFERAGFTVVDTGGPLLHSSKARLHVVAAIQARMGSTRLPGKVLVSIAGRPTIQRIAERLAVCQELDAVAVSTSVENRDDAIADLAAHLGLVCVRGSETDLIERLGRT'
                   'AARTGADALVRITADCPLVDPALVDRVVGVWRRSAGRLEYVSNVFPPTFPDGLDVEVLSRTVLERLDREVSDPFFRESLTAYVREHPAAFEIANVEHPEDLSRLRWTMDYPEDLAFVEAV'
                   'YRRLGNQGEIFGMDDLLRLLEWSPELRDLNRCREDVTVERGIRGTGYHAALRARGQAP2')   # likelihood.py L127 --context default (422 tokens)


def load_tokenizer(path: str = "tokenizer.json"):
    """likelihood.py L60-62 create_tokenizer_custom: Tokenizer.from_str(open(file).read())."""
    from tokenizers import Tokenizer
    with open(path, "r") as f:
        return Tokenizer.from_str(f.read())


def encode(tokenizer, s: str) -> list:
    return list(tokenizer.encode(s).ids)


def drop_trailing_terminal(target_ids: list) -> bool:
    """likelihood.py L186: the LAST target is dropped iff it is a terminal ('1' or '2')."""
    return len(target_ids) > 0 and target_ids[-1] in (BOS_TOKEN, EOS_TOKEN)


def assert_no_terminal(target_ids: list) -> None:
    """likelihood.py L190-191: no terminal may remain among the scored targets."""
    if any(t in (BOS_TOKEN, EOS_TOKEN) for t in target_ids):
        raise AssertionError("a terminal token remains among the scored targets")


def reverse_string(s: str) -> str:
    """likelihood.py L287 ``reverse = lambda s: s[::-1]``: the STRING is reversed."""
    return s[::-1]


def rows_T(tokenizer, seq: str) -> list:
    """The two forward rows of a T unit: (direction, input_ids) — the ids the model sees (the full encoded string)."""
    return [("lr", encode(tokenizer, seq)), ("rl", encode(tokenizer, reverse_string(seq)))]


def scored_targets_T(ids: list) -> tuple:
    """likelihood.py L181-197 on the ids: (n_logit_rows_kept, targets - FIRST_TOKEN) for the row's [L, V] logits."""
    target = ids[1:]
    n = len(target)
    if drop_trailing_terminal(target):
        target = target[:-1]
        n -= 1
    assert_no_terminal(target)
    return n, [t - FIRST_TOKEN for t in target]


def rows_for_unit(tokenizer, unit: dict) -> list:
    """The forward rows of a unit {unit_key, seq}: one dict per direction {unit_key, part ('lr' | 'rl'), input_ids, targets (shifted
    by -FIRST_TOKEN), n_keep, L} (the row grammar is the route's, not a kit's)."""
    key = unit["unit_key"]
    out = []
    for d, ids in rows_T(tokenizer, unit["seq"]):
        n_keep, tg = scored_targets_T(ids)
        out.append({"unit_key": key, "part": d, "input_ids": ids, "targets": tg, "n_keep": n_keep, "L": len(ids)})
    return out


def bucket_rows(rows: list, max_rows: int, max_tokens: int = None) -> list:
    """EXACT-LENGTH batches: rows grouped by (L, n_keep) in first-seen order, cut at max_rows (and max_tokens = rows*L). No
    padding, no mask — every batch is an unpadded [B, L] forward. Returns [{'L', 'n_keep', 'rows': [...]}, ...]."""
    groups, order = {}, []
    for r in rows:
        k = (r["L"], r["n_keep"])
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(r)
    batches = []
    for k in order:
        L, n_keep = k
        cap = max_rows if max_tokens is None else max(1, min(max_rows, max_tokens // max(1, L)))
        g = groups[k]
        for i in range(0, len(g), cap):
            batches.append({"L": L, "n_keep": n_keep, "rows": g[i:i + cap]})
    return batches


def combine_T(lr: dict, rl: dict) -> dict:
    """likelihood.py L295-296: ll_sum = .5*(ll_lr_sum + ll_rl_sum); ll_mean = .5*(ll_lr_mean + ll_rl_mean) — Python float64."""
    return {"ll_sum": .5 * (lr["ll_sum"] + rl["ll_sum"]), "ll_mean": .5 * (lr["ll_mean"] + rl["ll_mean"])}

