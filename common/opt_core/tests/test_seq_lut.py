"""opt_core.seq.lut: the tables reproduce per-character reference loops bit for bit; out-of-domain input is counted or refused by name."""
import pytest

np = pytest.importorskip("numpy")

from opt_core.seq import lut
from opt_core import report


# ---------------------------------------------------------------- references: the per-character loops the tables replace

def ref_onehot_uniform(seq, dtype="float16"):
    """A/C/G/T -> unit row, every other character -> 0.25 in every column (the uniform-background form)."""
    out = np.zeros((len(seq), 4), dtype=dtype)
    for i, ch in enumerate(seq.upper()):
        j = "ACGT".find(ch)
        if j >= 0:
            out[i, j] = 1
        else:
            out[i, :] = 0.25
    return out


def ref_onehot_zero(seq, dtype="int8"):
    """A/C/G/T (either case) -> unit row, anything else -> all-zero row."""
    out = np.zeros((len(seq), 4), dtype=dtype)
    for i, ch in enumerate(seq):
        j = "ACGT".find(ch.upper())
        if j >= 0:
            out[i, j] = 1
    return out


class StubAlphabet:
    """The shape of a residue alphabet with single-letter tokens plus multi-character specials."""
    def __init__(self):
        toks = ["<cls>", "<pad>", "<eos>", "<unk>"] + list("LAGVSERTIDPKQNFYMHWCXBUZO.-") + ["<null_1>", "<mask>"]
        self.tok_to_idx = {t: i for i, t in enumerate(toks)}
        self.padding_idx, self.cls_idx, self.eos_idx, self.unk_idx = 1, 0, 2, 3

    def encode(self, seq):                                   # the per-token loop: KeyError for an unknown character, like a strict tokenizer
        return [self.tok_to_idx[ch] for ch in seq]


def ref_batch(alphabet, seqs, truncation=None):
    enc = [np.asarray(alphabet.encode(s), dtype=np.int64) for s in seqs]
    if truncation:
        enc = [e[:truncation] for e in enc]
    width = max(len(e) for e in enc) + 2
    toks = np.full((len(enc), width), alphabet.padding_idx, dtype=np.int64)
    toks[:, 0] = alphabet.cls_idx
    for i, e in enumerate(enc):
        toks[i, 1:1 + len(e)] = e
        toks[i, 1 + len(e)] = alphabet.eos_idx
    return toks


# ---------------------------------------------------------------- one-hot

SEQS = ["ACGTNacgtn", "NNNNACGTRYKM", "", "T" * 37 + "n" + "A" * 3, "acgt" * 100]


def test_onehot_uniform_float16_bitwise():
    enc = lut.OneHotLUT("ACGT", on=1, off=0, unknown=0.25, dtype="float16")
    for s in SEQS:
        got = enc.encode(s)
        ref = ref_onehot_uniform(s)
        assert got.dtype == np.float16 and got.shape == (len(s), 4)
        assert got.tobytes() == ref.tobytes(), s
    assert enc.stats()["symbols"] == sum(len(s) for s in SEQS) and enc.stats()["calls"] == len(SEQS)


def test_onehot_zero_int8_bitwise_and_codes_path():
    enc = lut.OneHotLUT("ACGT", on=1, off=0, unknown=0, dtype="int8")
    for s in SEQS:
        assert enc.encode(s).tobytes() == ref_onehot_zero(s).tobytes(), s
    codes = np.frombuffer(b"ACGTn" * 6, dtype=np.uint8).reshape(2, 3, 5)          # a pre-gathered code block (genome memmap form)
    got = enc.encode_codes(codes)
    assert got.shape == (2, 3, 5, 4) and got.flags["C_CONTIGUOUS"]
    assert got.reshape(-1, 4).tobytes() == ref_onehot_zero("ACGTn" * 6).tobytes()
    with pytest.raises(TypeError):
        enc.encode_codes(np.arange(5))                                               # not uint8: refused by name


def test_onehot_aliases_case_and_into():
    enc = lut.OneHotLUT("ACGT", unknown=0, dtype="float32", aliases={"U": "T"}, fold_case=True)
    assert enc.encode("uU").tolist() == [[0, 0, 0, 1], [0, 0, 0, 1]]
    strict = lut.OneHotLUT("ACGT", unknown=0, dtype="float32", fold_case=False)
    assert strict.encode("a").tolist() == [[0, 0, 0, 0]]
    out = np.zeros((8, 4), dtype="float32")
    enc.encode_into(out, "CG", start=3)
    assert out[:3].sum() == 0 and out[5:].sum() == 0 and out[3].tolist() == [0, 1, 0, 0] and out[4].tolist() == [0, 0, 1, 0]
    with pytest.raises(ValueError):
        enc.encode_into(out, "ACGTACGT", start=2)                                    # span does not fit: refused by name
    with pytest.raises(ValueError):
        enc.encode("ACGT\u00e9")                                                      # non-ASCII str: no byte -> row mapping


def test_onehot_refuses_inexact_values_and_bad_alphabets():
    with pytest.raises(ValueError):
        lut.OneHotLUT("ACGT", unknown=0.3, dtype="float16")                          # 0.3 is not exact in float16
    with pytest.raises(ValueError):
        lut.OneHotLUT("AACG", unknown=0, dtype="float32")
    with pytest.raises(ValueError):
        lut.OneHotLUT("ACGT", unknown=0, dtype="float32", aliases={"U": "X"})


# ---------------------------------------------------------------- tokens

PROTS = ["MKTAYIAKQR", "LAGVSERTIDPKQNFYMHWCXBUZO.-", "M", "MKT" * 300]


def test_tokens_match_reference_converter():
    a = StubAlphabet()
    t = lut.TokenLUT(a.tok_to_idx, pad_idx=a.padding_idx, bos_idx=a.cls_idx, eos_idx=a.eos_idx)
    toks, ok = t.encode_batch(PROTS)
    assert ok == [True] * len(PROTS) and toks.dtype == np.int64
    assert toks.tobytes() == ref_batch(a, PROTS).tobytes()
    toks5, _ = t.encode_batch(PROTS, truncation=5)
    assert toks5.tobytes() == ref_batch(a, PROTS, truncation=5).tobytes()
    s = t.stats()
    assert s["fallback"] == 0 and s["sequences"] == 2 * len(PROTS) and s["single_char_tokens"] == 27


def test_tokens_out_of_domain_refused_or_counted():
    a = StubAlphabet()
    strict = lut.TokenLUT(a.tok_to_idx, pad_idx=a.padding_idx, bos_idx=a.cls_idx, eos_idx=a.eos_idx)
    with pytest.raises(lut.OutOfDomain) as ei:
        strict.encode_batch(["MKT", "MK T", "MKJ", "MKT"])                           # whitespace and 'J' have no id
    assert ei.value.rows == [1, 2]
    calls = []
    def fallback(seq):                                                               # the kit's stock encoder: here 'J' -> <unk>, ' ' dropped
        calls.append(seq)
        return [a.tok_to_idx.get(ch, a.unk_idx) for ch in seq if not ch.isspace()]
    lenient = lut.TokenLUT(a.tok_to_idx, pad_idx=a.padding_idx, bos_idx=a.cls_idx, eos_idx=a.eos_idx, fallback=fallback)
    toks, ok = lenient.encode_batch(["MKT", "MK T", "MKJ"])
    assert ok == [True, False, False] and calls == ["MK T", "MKJ"] and lenient.stats()["fallback"] == 2
    assert toks[1].tolist() == toks[0].tolist()                                       # fallback ids land in the same layout
    assert toks[2].tolist()[:5] == [a.cls_idx, a.tok_to_idx["M"], a.tok_to_idx["K"], a.unk_idx, a.eos_idx]
    ids, good = lenient.encode("MKT\u00e9")                                           # non-ASCII goes to the fallback too, counted
    assert good is False and lenient.stats()["fallback"] == 3


def test_tokens_no_bos_eos_and_line_fields():
    a = StubAlphabet()
    t = lut.TokenLUT(a.tok_to_idx, pad_idx=a.padding_idx)
    toks, _ = t.encode_batch(["MK", "M"])
    assert toks.tolist() == [[a.tok_to_idx["M"], a.tok_to_idx["K"]], [a.tok_to_idx["M"], a.padding_idx]]
    assert report.kv(**lut.line_fields(t, key="tok")) == "tok=token_lut seqs=2 tokens=3 fallback=0"
    enc = lut.OneHotLUT("ACGT", unknown=0, dtype="float32")
    enc.encode("ACG")
    f = lut.line_fields(enc)
    assert all(isinstance(v, str) for v in f.values()) and report.kv(**f) == "lut=onehot_lut symbols=3 calls=1"
    with pytest.raises(ValueError):
        lut.TokenLUT({"<cls>": 0, "<pad>": 1}, pad_idx=1)                            # no single-character tokens


def test_onehot_requires_unknown_and_dtype():
    with pytest.raises(TypeError):
        lut.OneHotLUT("ACGT")                                                         # the kit states its stock's unknown rule and dtype
    with pytest.raises(TypeError):
        lut.OneHotLUT("ACGT", 1, 0, 0.25, "float16")                                  # keyword-only
