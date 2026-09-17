"""Byte -> row lookup-table encoders: one-hot featurisation of nucleotide / residue strings and single-character tokenisation, as one
numpy gather instead of a per-character Python loop.

Contract. Both encoders build a 256-row table once from the kit's own alphabet data and encode by indexing it with the input's bytes,
so the output for every byte is exactly the table row for that byte: one-hot values that are exactly representable in the requested
dtype (0, 1, 0.25, ...) reproduce a per-character loop bit for bit, and token ids are the kit's own integers. What a table cannot prove
it does not guess: :class:`OneHotLUT` gives every byte outside the alphabet the kit's declared ``unknown`` row; :class:`TokenLUT` marks a
sequence containing any byte without an id (or any non-ASCII character) as out of domain and either hands it to the kit's ``fallback``
encoder (counted in ``stats()["fallback"]``) or raises :class:`OutOfDomain` naming the sequences — never a silent substitution. numpy
is imported on first use; where it is missing the call raises :class:`Unavailable` (a sentence a report can carry). :func:`line_fields`
gives the ``{key: value}`` fields of the kit's activation line (``opt_core.report.kv(**fields)``).
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


class Unavailable(RuntimeError):
    """numpy cannot be imported in this interpreter; the message names the module and the encoder that asked for it."""


class OutOfDomain(ValueError):
    """One or more sequences contain characters the table has no id for and no fallback encoder was given; ``.rows`` lists their
    positions in the batch."""

    def __init__(self, message: str, rows: Sequence[int]):
        super().__init__(message)
        self.rows = list(rows)


def _np(who: str):
    try:
        import numpy
    except Exception as e:  # noqa: BLE001 — any import failure is the same refusal
        raise Unavailable("opt_core.seq.lut.{who} needs numpy, which did not import: {err!r}".format(who=who, err=e))
    return numpy


def _codes(np, seq: Union[str, bytes]) -> Tuple[object, bool]:
    """uint8 codes of a str / bytes and whether every character was a single byte (ASCII for str)."""
    if isinstance(seq, str):
        try:
            b = seq.encode("ascii")
        except UnicodeEncodeError:
            return np.frombuffer(seq.encode("utf-8"), dtype=np.uint8), False
    else:
        b = bytes(seq)
    return np.frombuffer(b, dtype=np.uint8), True


class OneHotLUT:
    """One-hot rows by table lookup.

    ``alphabet``: the symbols in column order (``"ACGT"`` gives A -> column 0 ... T -> column 3). ``unknown`` (required): the value of
    EVERY column for a byte that is not a symbol (``0.25`` for a uniform background, ``0`` for an all-zero row) — the kit's stock rule,
    never assumed here. ``dtype`` (required): the numpy dtype name the kit's consumer reads. ``on`` / ``off``: the values of a symbol's own
    column and of the other columns (1 / 0). ``fold_case``: lower-case letters get their upper-case
    symbol's row. ``aliases``: extra ``{character: symbol}`` pairs that share a symbol's row (e.g. ``{"U": "T"}``).

    ``encode(seq)`` -> ``(len(seq), width)`` array; ``encode_codes(codes)`` -> ``codes.shape + (width,)`` for uint8 codes the kit gathered
    itself (a memory-mapped genome, a pre-encoded batch); ``encode_into(out, seq, start)`` writes a sequence's rows into a caller-owned
    zero-initialised window (centre padding / trimming stay the kit's arithmetic). ``stats()`` counts calls and symbols for the record."""

    def __init__(self, alphabet: str, *, unknown: float, dtype: str, on: float = 1, off: float = 0,
                 fold_case: bool = True, aliases: Optional[Mapping[str, str]] = None):
        np = _np("OneHotLUT")
        if len(set(alphabet)) != len(alphabet) or not alphabet:
            raise ValueError("alphabet must be non-empty with distinct symbols: {a!r}".format(a=alphabet))
        self.alphabet, self.width = str(alphabet), len(alphabet)
        self.on, self.off, self.unknown, self.dtype, self.fold_case = on, off, unknown, np.dtype(dtype).name, bool(fold_case)
        self.aliases = dict(aliases or {})
        table = np.full((256, self.width), unknown, dtype=self.dtype)
        cols = {}
        for j, sym in enumerate(self.alphabet):
            if len(sym) != 1 or ord(sym) > 255:
                raise ValueError("alphabet symbols must be single bytes: {s!r}".format(s=sym))
            cols[sym] = j
        for ch, sym in self.aliases.items():
            if sym not in cols or len(ch) != 1 or ord(ch) > 255:
                raise ValueError("alias {c!r} -> {s!r}: the target must be an alphabet symbol and the alias a single byte".format(c=ch, s=sym))
        for ch, j in list(cols.items()) + [(c, cols[s]) for c, s in self.aliases.items()]:
            keys = {ch, ch.lower(), ch.upper()} if self.fold_case else {ch}
            for k in keys:
                table[ord(k)] = off
                table[ord(k), j] = on
        for v in (on, off, unknown):                       # the exactness premise: every table value survives the dtype round trip
            if float(np.asarray(v, dtype=self.dtype)) != float(v):
                raise ValueError("value {v!r} is not exactly representable in {d}".format(v=v, d=self.dtype))
        self.table = table
        self._n_calls = 0
        self._n_symbols = 0

    def encode_codes(self, codes):
        """Rows for uint8 codes of any shape -> ``codes.shape + (width,)``, C-contiguous."""
        np = _np("OneHotLUT")
        codes = np.asarray(codes)
        if codes.dtype != np.uint8:
            raise TypeError("codes must be uint8, got {d}".format(d=codes.dtype))
        self._n_calls += 1
        self._n_symbols += int(codes.size)
        return np.ascontiguousarray(self.table[codes])

    def encode(self, seq: Union[str, bytes]):
        """Rows for one sequence -> ``(len, width)``. A str with a non-ASCII character has no single-byte codes: ValueError by name."""
        np = _np("OneHotLUT")
        codes, single = _codes(np, seq)
        if not single:
            raise ValueError("OneHotLUT.encode: the sequence has non-ASCII characters (no byte -> row mapping); length {n}".format(n=len(seq)))
        return self.encode_codes(codes)

    def encode_into(self, out, seq: Union[str, bytes], start: int = 0):
        """Write ``encode(seq)`` into ``out[start:start + len(seq)]`` (``out``: a caller-owned ``(L, width)`` array of this dtype) and
        return ``out``; rows outside that span are left as the caller initialised them."""
        rows = self.encode(seq)
        n = rows.shape[0]
        if start < 0 or start + n > out.shape[0] or tuple(out.shape[1:]) != (self.width,):
            raise ValueError("encode_into: span [{s}, {e}) does not fit out{sh}".format(s=start, e=start + n, sh=tuple(out.shape)))
        out[start:start + n] = rows
        return out

    def stats(self) -> dict:
        return {"encoder": "onehot_lut", "alphabet": self.alphabet, "dtype": self.dtype, "unknown": self.unknown, "calls": self._n_calls, "symbols": self._n_symbols}


class TokenLUT:
    """Token ids by table lookup for vocabularies whose sequence tokens are single characters.

    ``tok_to_idx``: the kit's own token -> id mapping; only single-character, non-whitespace, single-byte tokens enter the table (multi-
    character specials such as ``<mask>`` cannot appear inside a plain sequence and are left to the fallback). ``pad_idx``: the padding
    id of a batch. ``bos_idx`` / ``eos_idx``: written before / after every sequence when given. ``fallback``: the kit's stock encoder
    ``str -> sequence of ids`` for out-of-domain sequences (counted); without it those sequences raise :class:`OutOfDomain`.

    ``encode(seq)`` -> ``(ids int64, in_domain)``; ``encode_batch(seqs, truncation=None)`` -> ``(tokens (B, T) int64, in_domain list)``
    padded to the longest (truncation applies to the encoded sequence before bos/eos, as a converter that truncates ids does).
    ``stats()`` counts sequences, tokens and fallbacks for the record."""

    def __init__(self, tok_to_idx: Mapping[str, int], pad_idx: int, bos_idx: Optional[int] = None, eos_idx: Optional[int] = None,
                 fallback: Optional[Callable[[str], Sequence[int]]] = None):
        np = _np("TokenLUT")
        table = np.full(256, -1, dtype=np.int64)
        n_single = 0
        for tok, idx in tok_to_idx.items():
            if isinstance(tok, str) and len(tok) == 1 and ord(tok) < 256 and not tok.isspace():
                if int(idx) < 0:
                    raise ValueError("token {t!r} has a negative id {i}; -1 is the table's no-id mark".format(t=tok, i=idx))
                table[ord(tok)] = int(idx)
                n_single += 1
        if n_single == 0:
            raise ValueError("tok_to_idx has no single-character tokens; nothing to build a table from")
        self.table, self.n_single = table, n_single
        self.pad_idx, self.bos_idx, self.eos_idx, self.fallback = int(pad_idx), bos_idx, eos_idx, fallback
        self._n_seqs = 0
        self._n_tokens = 0
        self._n_fallback = 0

    def encode(self, seq: str):
        """``(ids, in_domain)``: the table's ids when every character has one; else the fallback's ids (counted) with ``in_domain``
        False, or :class:`OutOfDomain` when there is no fallback."""
        np = _np("TokenLUT")
        codes, single = _codes(np, seq)
        ids = self.table[codes] if single else None
        ok = bool(single and not (ids.size and (ids < 0).any()))
        self._n_seqs += 1
        if ok:
            self._n_tokens += int(ids.size)
            return ids, True
        if self.fallback is None:
            raise OutOfDomain("TokenLUT.encode: sequence of length {n} has characters without a single-character id and no fallback encoder was given".format(n=len(seq)), [0])
        out = np.asarray(list(self.fallback(seq)), dtype=np.int64)
        self._n_fallback += 1
        self._n_tokens += int(out.size)
        return out, False

    def encode_batch(self, seqs: Iterable[str], truncation: Optional[int] = None):
        """``(tokens, in_domain)``: ``tokens`` is ``(B, T)`` int64 filled with ``pad_idx``, ``T`` = longest encoded length (after
        truncation) + bos + eos; row ``i`` holds ``[bos] ids [eos]`` from column 0. ``in_domain[i]`` is False for a row the fallback
        encoded. Without a fallback, out-of-domain rows raise one :class:`OutOfDomain` listing every such row."""
        np = _np("TokenLUT")
        enc = []      # type: List[object]
        ok = []       # type: List[bool]
        bad = []      # type: List[int]
        for i, s in enumerate(seqs):
            try:
                ids, good = self.encode(s)
            except OutOfDomain:
                bad.append(i)
                continue
            if truncation:
                ids = ids[: int(truncation)]
            enc.append(ids)
            ok.append(good)
        if bad:
            raise OutOfDomain("TokenLUT.encode_batch: {n} sequence(s) out of domain with no fallback encoder: rows {r}".format(n=len(bad), r=bad), bad)
        if not enc:
            raise ValueError("encode_batch: no sequences")
        bos, eos = (0 if self.bos_idx is None else 1), (0 if self.eos_idx is None else 1)
        width = max(int(e.shape[0]) for e in enc) + bos + eos
        tokens = np.full((len(enc), width), self.pad_idx, dtype=np.int64)
        if bos:
            tokens[:, 0] = int(self.bos_idx)
        for i, e in enumerate(enc):
            n = int(e.shape[0])
            tokens[i, bos:bos + n] = e
            if eos:
                tokens[i, bos + n] = int(self.eos_idx)
        return tokens, ok

    def stats(self) -> dict:
        return {"encoder": "token_lut", "single_char_tokens": self.n_single, "sequences": self._n_seqs, "tokens": self._n_tokens,
                "fallback": self._n_fallback, "has_fallback": self.fallback is not None}


def line_fields(encoder, key: str = "lut") -> Dict[str, str]:
    """The activation-line fields of an encoder (or its ``stats()`` dict): ``{<key>: "onehot_lut", "symbols": <n>, "calls": <n>}`` or
    ``{<key>: "token_lut", "seqs": <n>, "tokens": <n>, "fallback": <n>}`` — the fallback count is the census a run record needs (0 = every
    sequence went through the table); every value a str. Feed to ``opt_core.report.kv(**fields)``."""
    s = encoder.stats() if hasattr(encoder, "stats") else dict(encoder)
    if s.get("encoder") == "onehot_lut":
        return {key: "onehot_lut", "symbols": str(s["symbols"]), "calls": str(s["calls"])}
    return {key: "token_lut", "seqs": str(s["sequences"]), "tokens": str(s["tokens"]), "fallback": str(s["fallback"])}
