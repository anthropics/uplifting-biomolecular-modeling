"""ef2_feats (FZ-EF2) — vectorised MSA featurisation for ESMFold2's input builder (host side; exact by construction).

    import ef2_feats as FZ
    FZ.enable()                 # patches esm.models.esmfold2.paired_msa.msa_to_res_type_and_deletions AND .construct_paired_msa (module
                                # attributes, resolved at call time by upstream's compute_msa_features; idempotent)
    FZ.disable()                # restores upstream's two functions
    FZ.stats()                  # calls / rows / chars / seconds / fallbacks; pair_calls / pair_rows / pair_seconds / pair_selfcheck_* / pair_fallback

Lever (one word, `fz`: both halves of the host MSA featurisation upstream's `fold()` runs INSIDE the timed window, before the network)
-----
FZ  upstream's `paired_msa.msa_to_res_type_and_deletions(msa, letter_to_res_type)` converts an MSA to `(res_type[M, L] int64,
    deletions[M, L] float32)` with a pure-Python double loop over every row and every CHARACTER of the a3m text (one
    `is_a3m_insertion` call + dict.get + str.upper per character: ~13.6 M calls for a 1400-token 6-chain input, ~85-90 % of
    `prepare_input`, which upstream's `fold()` runs inside the timed window). This module computes the same two arrays over one bytes view
    of the whole MSA: an insertion mask (lowercase a-z or '.', = is_a3m_insertion) and per-row match counts; when every row holds exactly L
    match columns (always so for MSA.from_a3m, which rejects ragged a3m) the insertion characters are dropped with one bytes.translate and
    the [M, L] letter grid is mapped through a 256-entry table built from the SAME `letter_to_res_type` dict the caller passes ('-' -> gap,
    unknown -> UNK: the loop's branches); ragged rows take a general cumulative-rank path with the loop's cut-at-L / default-fill
    semantics. Deletions: `msa.deletions.astype(float32)` when stored (what upstream returns then), else insertion-run lengths by
    cumulative-sum differences. Rows that are not pure ASCII take upstream's own function (counted; never expected from a3m files).
    Outputs are integer ids and float32 counts of small integers: identical arrays, not merely close ones.
FZ (pairing)  upstream's `paired_msa.construct_paired_msa(...)` then builds the taxonomy-paired row table with pure-Python loops: one dict
    per output row per chain (up to 16384 rows x chains), `list.pop(0)` for every unpaired draw (quadratic in the MSA depth) and one list
    comprehension per chain over the row dicts — at 800 tokens / 4 chains / ~7.7 k rows per chain that is ~0.25 s of a fold's ~0.45 s
    host featurisation (cProfile of the fold window), and its ~100 k short-lived containers per fold are what trips the
    interpreter's full (generation-2) garbage collections inside fold windows (0.25-0.5 s each, every second or third fold).
    `construct_paired_msa_vec` computes the SAME row table with array operations: the taxonomy groups by np.unique (kept: groups of more
    than one entry; ordered by distinct-chain count descending, then first appearance — the dict-insertion order Python's stable sort
    keeps), each member chain's row values by cycling its group entries (i mod n), each non-member chain's draws as the running rank of its
    non-member rows into its ascending `available` list (exhausted -> -1), the trailing unpaired block as the continuation of those ranks,
    the max_pairs / max_total / max_seqs cuts at the same row counts; the per-chain gather into the [M, T] arrays is upstream's own
    (np.ix_ on rows x columns).  The first call of a process runs upstream's function too and compares all three arrays (np.array_equal):
    identical -> `pair_selfcheck_identical`; a difference (never expected) -> `pair_selfcheck_mismatch`, upstream's arrays are returned and
    the pairing half steps aside for the rest of the process (counted, named on the LEVERFOLD / EXIT lines).  A taxonomy id outside int64
    takes upstream's function for that call (`pair_fallback`).
Tier: exact (bitwise) — the model's input features are the same arrays.
"""
import collections
import time

import numpy as np

STATS = collections.Counter()
_ORIG = {}

_INS = np.zeros(256, dtype=bool)
_INS[ord("a"):ord("z") + 1] = True          # str.islower() on an ASCII character
_INS[ord(".")] = True                       # is_a3m_insertion: ch == "." or ch.islower()
_NL = 10                                    # rows are joined with '\n' (never inside a parsed sequence)


def _paired_msa():
    import esm.models.esmfold2.paired_msa as P
    return P


def _res_lut(letter_to_res_type, gap_id, unk_id):
    """res_type id for every byte value as the loop assigns it: '-' -> gap; else letter_to_res_type.get(ch.upper(), UNK)."""
    lut = np.empty(256, dtype=np.int64)
    for c in range(256):
        lut[c] = letter_to_res_type.get(chr(c).upper(), unk_id)
    lut[ord("-")] = gap_id
    return lut


_DELETE = bytes(range(ord("a"), ord("z") + 1)) + b"."  # the insertion characters, for bytes.translate(None, delete)


def _rows_general(seqs, L, lut, gap_id, stored_deletions):
    """Any row lengths / match counts: per-character column ranks by cumulative sums over one '\n'-joined bytes view (rows with more than L
    match columns are cut at L like the loop's break; shorter rows keep the gap / zero defaults)."""
    M = len(seqs)
    b = np.frombuffer(("\n" + "\n".join(seqs)).encode("ascii"), dtype=np.uint8)
    nl = b == _NL                                                 # one '\n' before every row: row starts
    ins = _INS[b]
    match = ~(ins | nl)                                           # every other character occupies a match column (letters, '-', anything else)
    res_type = np.full((M, L), gap_id, dtype=np.int64)
    nl_pos = np.flatnonzero(nl)                                   # [M] position of each row's leading '\n'
    row_of = np.cumsum(nl, dtype=np.int32) - 1                    # row index of every position
    cm = np.cumsum(match, dtype=np.int32)                          # match characters up to and including each position
    mp = np.flatnonzero(match)                                    # positions of match characters, in text order
    rows = row_of[mp]
    cols = cm[mp] - cm[nl_pos[rows]] - 1                          # 0-based rank of the match character inside its row = its column
    keep = cols < L                                               # the loop breaks at the first column >= L: later characters of the row are ignored
    rk, ck = rows[keep], cols[keep]
    res_type[rk, ck] = lut[b[mp[keep]]]
    if stored_deletions is not None:
        deletions = stored_deletions.astype(np.float32)
    else:
        deletions = np.zeros((M, L), dtype=np.float32)
        cins = np.cumsum(ins, dtype=np.int32)                      # insertion characters up to and including each position
        bpos = np.flatnonzero(match | nl)                          # boundaries: row starts and match characters (bpos[0] is row 0's '\n')
        d = cins[bpos[1:]] - cins[bpos[:-1]]                       # insertions strictly between consecutive boundaries
        dmatch = d[match[bpos[1:]]]                                # ... for boundaries that are match characters, in text order == mp order
        deletions[rk, ck] = dmatch[keep].astype(np.float32)        # the run preceding each kept column (0 where none, as np.zeros leaves it)
    return res_type, deletions


def msa_to_res_type_and_deletions_vec(msa, letter_to_res_type):
    """Drop-in for paired_msa.msa_to_res_type_and_deletions: same arguments, same return `(res_type int64 [M, L], deletions float32 [M, L])`."""
    P = _paired_msa()
    t0 = time.perf_counter()
    entries = msa.entries
    M = len(entries)
    if M == 0:
        return _ORIG["fn"](msa, letter_to_res_type)               # upstream indexes entries[0]: its own error
    seqs = [e.sequence for e in entries]
    try:
        raw = "".join(seqs).encode("ascii")
    except UnicodeEncodeError:                                   # a non-ASCII row: upstream's per-character semantics (str.islower / upper) decide
        STATS["fallback_non_ascii"] += 1
        return _ORIG["fn"](msa, letter_to_res_type)
    q = np.frombuffer(seqs[0].encode("ascii"), dtype=np.uint8)
    L = int(q.size - np.count_nonzero(_INS[q]))                   # query length after stripping insertions from row 0
    if msa.deletions is not None:                                # upstream's assert (after its loop; the outcome is the same exception)
        msg = f"stored deletions {msa.deletions.shape} != expected {(M, L)}"
        assert msa.deletions.shape == (M, L), msg
    lut = _res_lut(letter_to_res_type, P.MSA_GAP_TOKEN_ID, P.PROTEIN_UNK_RES_TYPE)
    lens = np.fromiter((len(x) for x in seqs), dtype=np.int64, count=M)
    flat = np.frombuffer(raw, dtype=np.uint8)
    insf = _INS[flat]
    n_ins = int(np.count_nonzero(insf))
    fast = bool((lens > 0).all())
    if fast and n_ins:
        starts = np.zeros(M, dtype=np.int64); np.cumsum(lens[:-1], out=starts[1:])
        match_cnt = lens - np.add.reduceat(insf, starts)          # per row: characters minus insertion characters
        fast = bool((match_cnt == L).all())
    elif fast:
        fast = bool((lens == L).all())
    if not fast:                                                 # ragged rows (never produced by MSA.from_a3m, which rejects them): the general ranks
        STATS["general_path"] += 1
        res_type, deletions = _rows_general(seqs, L, lut, P.MSA_GAP_TOKEN_ID, msa.deletions)
    else:
        # every row holds exactly L match columns: dropping the insertion characters (one C pass) leaves the [M, L] letter grid in row order
        codes = np.frombuffer(raw.translate(None, _DELETE), dtype=np.uint8).reshape(M, L) if n_ins else flat.reshape(M, L)
        res_type = lut[codes]                                    # '-' -> gap, letters by the caller's map, anything else -> UNK (the loop's branches)
        if msa.deletions is not None:
            deletions = msa.deletions.astype(np.float32)
        elif n_ins == 0:
            deletions = np.zeros((M, L), dtype=np.float32)       # no insertion character anywhere: every run length is 0
        else:
            cins = np.cumsum(insf, dtype=np.int32)                # insertions up to and including each position (= strictly before, at a match position)
            c = cins[~insf].reshape(M, L)
            before_row = np.zeros(M, dtype=np.int32)
            before_row[1:] = cins[starts[1:] - 1]                 # insertions in all earlier rows
            deletions = np.diff(c, axis=1, prepend=before_row[:, None]).astype(np.float32)   # the run preceding each match column of its own row
    STATS["calls"] += 1; STATS["rows"] += M; STATS["chars"] += int(flat.size)
    STATS["seconds_x1000"] += int(1000 * (time.perf_counter() - t0))
    return res_type, deletions


# ----------------------------------------------------------------------------------------------------------------- taxonomy pairing
_PAIR = {"checked": False, "off": False}     # first-call self-check state; off = stepped aside after a mismatch (upstream's function serves)


def _pairing_tables(taxs, max_pairs, max_total, max_seqs):
    """The row table of construct_paired_msa for chains 0..C-1 with per-row taxonomy arrays ``taxs`` (int64, row 0 = query):
    returns (M, [chain_pairing int64 [M]], [chain_paired float32 [M]]) — row r of chain c reads MSA row chain_pairing[c][r] (-1: gap row),
    chain_paired[c][r] is upstream's is_paired flag."""
    nC = len(taxs)
    n_c = [int(t.size) for t in taxs]
    e_ci, e_seq, e_tax = [], [], []
    for ci, t in enumerate(taxs):                                    # entries in upstream's traversal order: chains ascending, rows ascending,
        if t.size > 1:                                               # the query row and taxon -1 skipped
            s = np.flatnonzero(t[1:] != -1) + 1
            if s.size:
                e_ci.append(np.full(s.size, ci, dtype=np.int64)); e_seq.append(s.astype(np.int64)); e_tax.append(t[s])
    K = 0
    if e_ci:
        E_ci, E_seq, E_tax = np.concatenate(e_ci), np.concatenate(e_seq), np.concatenate(e_tax)
        u_tax, first_idx, inv, cnt = np.unique(E_tax, return_index=True, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1).astype(np.int64)
        kept_ids = np.flatnonzero(cnt > 1)                           # taxonomy_map keeps groups of more than one ENTRY (any chains)
        K = int(kept_ids.size)
    if K:
        uk = np.unique(inv * nC + E_ci)                               # distinct (group, chain) pairs
        n_distinct = np.bincount(uk // nC, minlength=u_tax.size)     # chains per group
        order = kept_ids[np.lexsort((first_idx[kept_ids], -n_distinct[kept_ids]))]   # distinct chains descending, then first appearance (stable sort of dict order)
        rank = np.full(u_tax.size, -1, dtype=np.int64); rank[order] = np.arange(K, dtype=np.int64)
        e_rank = rank[inv]
        sel = np.flatnonzero(e_rank >= 0)
        o = sel[np.argsort(e_rank[sel], kind="stable")]              # kept entries in (group rank, chain, row) order — per_chain's lists, concatenated
        S_k, S_ci, S_seq = e_rank[o], E_ci[o], E_seq[o]
        n_kc = np.zeros((K, nC), dtype=np.int64); np.add.at(n_kc, (S_k, S_ci), 1)
        off_kc = (np.cumsum(n_kc.ravel()) - n_kc.ravel()).reshape(K, nC)   # start of each (group, chain) list inside S_seq
        max_occ = n_kc.max(axis=1)                                    # rows each group emits
        R = int(max_occ.sum())
        P_rows = min(R, max(1, int(max_pairs) - 1))                   # rows appended until len(pairing) (query row included) reaches max_pairs
        row_k = np.repeat(np.arange(K, dtype=np.int64), max_occ)[:P_rows]
        start_k = np.cumsum(max_occ) - max_occ
        row_j = np.arange(P_rows, dtype=np.int64) - start_k[row_k]
        visited = [S_seq[S_ci == ci] for ci in range(nC)]
    else:
        P_rows = 0
        row_k = row_j = np.zeros(0, dtype=np.int64)
        visited = [np.zeros(0, dtype=np.int64) for _ in range(nC)]
    avail = [np.setdiff1d(np.arange(1, n_c[ci], dtype=np.int64), visited[ci]) for ci in range(nC)]   # ascending, as upstream builds it
    vals, flags, consumed = [], [], []
    for ci in range(nC):
        val = np.full(P_rows, -1, dtype=np.int64); member = np.zeros(P_rows, dtype=bool)
        if K:
            n_r = n_kc[row_k, ci]
            member = n_r > 0
            if member.any():
                val[member] = S_seq[off_kc[row_k[member], ci] + (row_j[member] % n_r[member])]
        nm = np.flatnonzero(~member)                                  # rows where this chain draws its next unpaired row (FIFO), or -1 when exhausted
        A = avail[ci]
        take = min(int(nm.size), int(A.size))
        val[nm[:take]] = A[:take]
        vals.append(val); flags.append(member.astype(np.float32)); consumed.append(take)
    len_after = 1 + P_rows
    max_left = max((int(avail[ci].size) - consumed[ci] for ci in range(nC)), default=0)
    n_left = max(0, min(int(max_total) - len_after, max_left))      # the trailing unpaired block
    M = min(len_after + n_left, int(max_seqs)) if int(max_seqs) >= 0 else max(0, len_after + n_left + int(max_seqs))   # pairing[:max_seqs]
    pairings, paireds = [], []
    for ci in range(nC):
        A = avail[ci]
        pos = consumed[ci] + np.arange(n_left, dtype=np.int64)              # this chain's remaining unpaired rows continue in order; -1 once exhausted
        left = np.where(pos < A.size, A[np.minimum(pos, A.size - 1)], -1) if (n_left and A.size) else np.full(n_left, -1, dtype=np.int64)
        pairings.append(np.concatenate([np.zeros(1, dtype=np.int64), vals[ci], left])[:M])
        paireds.append(np.concatenate([np.ones(1, dtype=np.float32), flags[ci], np.zeros(n_left, dtype=np.float32)])[:M])
    return M, pairings, paireds


def _construct_paired_msa_arrays(P, chain_msas, chain_query_res_types, token_asym_ids, token_res_ids, letter_to_res_type, max_pairs, max_total, max_seqs):
    chain_ids = sorted(chain_msas.keys())
    rts, dls, taxs = [], [], []
    for c in chain_ids:
        m = chain_msas.get(c)
        if m is None or m.depth == 0:                                # a chain without an MSA: the one-row dummy (its query), taxonomy -1
            qres = chain_query_res_types[c]
            rts.append(qres[None, :]); dls.append(np.zeros((1, qres.shape[0]), dtype=np.float32)); taxs.append(np.full(1, -1, dtype=np.int64))
            continue
        rt, dl = P.msa_to_res_type_and_deletions(m, letter_to_res_type)   # the module attribute: the vectorised converter above under fz
        rts.append(rt); dls.append(dl)
        taxs.append(np.fromiter((P._taxonomy_from_header(e.header) for e in m.entries), dtype=np.int64, count=len(m.entries)))  # upstream's own header rule; OverflowError beyond int64 (caller falls back)
    M, pairings, paireds = _pairing_tables(taxs, max_pairs, max_total, max_seqs)
    T = len(token_asym_ids)
    msa_residues = np.full((M, T), P.MSA_GAP_TOKEN_ID, dtype=np.int64)
    deletion_value = np.zeros((M, T), dtype=np.float32)
    paired_mask = np.zeros((M, T), dtype=np.float32)
    for ci, c in enumerate(chain_ids):                                # upstream's per-chain gather (rows x this chain's token columns): the same values
        rt, dl = rts[ci], dls[ci]
        Lc = rt.shape[1]
        chain_pairing, chain_paired = pairings[ci], paireds[ci]
        token_mask = token_asym_ids == c
        if not token_mask.any():
            continue
        token_idx = np.flatnonzero(token_mask)
        t0, t1 = int(token_idx[0]), int(token_idx[-1]) + 1
        block = int(token_idx.size) == t1 - t0                        # this chain's tokens are one contiguous run of the token axis
        cols = np.minimum(token_res_ids[token_mask], Lc - 1)          # modified-residue tokens past the query length read the last column
        valid_rows = chain_pairing >= 0                               # -1 rows stay gap / zero
        if valid_rows.any():
            if block and int(cols.size) == Lc and np.array_equal(cols, np.arange(Lc)):   # the tokens are the chain's residues 0..Lc-1 in order: whole
                take = np.maximum(chain_pairing, 0)                   # MSA rows, gathered contiguously and written as one block (a -1 row reads row 0
                g_rt = rt[take]; g_dl = dl[take]                       # and is reset to the arrays' initial gap / 0 — the block is untouched before)
                if not valid_rows.all():
                    inv = ~valid_rows
                    g_rt[inv] = P.MSA_GAP_TOKEN_ID; g_dl[inv] = 0.0
                msa_residues[:, t0:t1] = g_rt; deletion_value[:, t0:t1] = g_dl
            else:                                                     # any other token layout: rows x columns fancy gather / scatter (upstream's form)
                rows = chain_pairing[valid_rows]
                valid_idx = np.flatnonzero(valid_rows)
                msa_residues[np.ix_(valid_idx, token_idx)] = rt[np.ix_(rows, cols)]
                deletion_value[np.ix_(valid_idx, token_idx)] = dl[np.ix_(rows, cols)]
        if block:
            paired_mask[:, t0:t1] = chain_paired[:, None]
        else:
            paired_mask[:, token_mask] = chain_paired[:, None]
    return msa_residues, deletion_value, paired_mask


def construct_paired_msa_vec(chain_msas, chain_query_res_types, token_asym_ids, token_res_ids, letter_to_res_type=None, *,
                             max_pairs=8192, max_total=16384, max_seqs=16384):
    """Drop-in for paired_msa.construct_paired_msa: same arguments, same return `(msa_residues int64 [M, T], deletion_value float32 [M, T],
    is_paired float32 [M, T])` — identical arrays (self-checked against upstream's function on the first call of the process)."""
    P = _paired_msa()
    if _PAIR["off"]:
        return _ORIG["pair"](chain_msas, chain_query_res_types, token_asym_ids, token_res_ids, letter_to_res_type,
                             max_pairs=max_pairs, max_total=max_total, max_seqs=max_seqs)
    t0 = time.perf_counter()
    if letter_to_res_type is None:
        letter_to_res_type = P.protein_letter_to_res_type()
    try:
        out = _construct_paired_msa_arrays(P, chain_msas, chain_query_res_types, token_asym_ids, token_res_ids, letter_to_res_type,
                                           max_pairs, max_total, max_seqs)
    except OverflowError:                                            # a `key=` beyond int64: upstream's Python ints decide that input
        STATS["pair_fallback"] += 1
        return _ORIG["pair"](chain_msas, chain_query_res_types, token_asym_ids, token_res_ids, letter_to_res_type,
                             max_pairs=max_pairs, max_total=max_total, max_seqs=max_seqs)
    STATS["pair_calls"] += 1; STATS["pair_rows"] += int(out[0].shape[0])
    STATS["pair_seconds_x1000"] += int(1000 * (time.perf_counter() - t0))
    if not _PAIR["checked"]:                                         # once per process: upstream's own function on the same arguments, all three arrays compared
        _PAIR["checked"] = True
        ref = _ORIG["pair"](chain_msas, chain_query_res_types, token_asym_ids, token_res_ids, letter_to_res_type,
                            max_pairs=max_pairs, max_total=max_total, max_seqs=max_seqs)
        same = all(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b) for a, b in zip(out, ref))
        if same:
            STATS["pair_selfcheck_identical"] += 1
        else:
            STATS["pair_selfcheck_mismatch"] += 1; _PAIR["off"] = True
            return ref
    return out


def enable():
    """Route paired_msa.msa_to_res_type_and_deletions AND paired_msa.construct_paired_msa through the vectorised implementations (module
    attributes: compute_msa_features imports construct_paired_msa from the module inside the call, and construct_paired_msa resolves the
    converter at call time). Idempotent."""
    import ef2_srcguard                                                       # both functions reproduce upstream's statements: refuse by name on another upstream source
    ef2_srcguard.check("fz")
    P = _paired_msa()
    if "fn" not in _ORIG:
        _ORIG["fn"] = P.msa_to_res_type_and_deletions
    if "pair" not in _ORIG:
        _ORIG["pair"] = P.construct_paired_msa
    P.msa_to_res_type_and_deletions = msa_to_res_type_and_deletions_vec
    P.construct_paired_msa = construct_paired_msa_vec
    return describe()


def disable():
    P = _paired_msa()
    if "fn" in _ORIG:
        P.msa_to_res_type_and_deletions = _ORIG["fn"]
    if "pair" in _ORIG:
        P.construct_paired_msa = _ORIG["pair"]
    return True


def active():
    P = _paired_msa()
    return ("fn" in _ORIG and P.msa_to_res_type_and_deletions is msa_to_res_type_and_deletions_vec
            and "pair" in _ORIG and P.construct_paired_msa is construct_paired_msa_vec)


def pairing_state():
    """`on` (vectorised pairing serving), `aside` (stepped aside after a self-check mismatch: upstream's function serves), `off` (not installed)."""
    if "pair" not in _ORIG or _paired_msa().construct_paired_msa is not construct_paired_msa_vec:
        return "off"
    return "aside" if _PAIR["off"] else "on"


def describe():
    on = "fn" in _ORIG and _paired_msa().msa_to_res_type_and_deletions is msa_to_res_type_and_deletions_vec
    return f"ef2_feats: msa_to_res_type_and_deletions vectorised={'on' if on else 'off'} construct_paired_msa vectorised={pairing_state()}"


def stats():
    d = dict(STATS)
    d["seconds"] = d.pop("seconds_x1000", 0) / 1000.0
    d["pair_seconds"] = d.pop("pair_seconds_x1000", 0) / 1000.0
    return d
