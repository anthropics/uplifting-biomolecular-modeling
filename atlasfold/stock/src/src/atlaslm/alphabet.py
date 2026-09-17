# fmt: off
# ESM-3's sequence vocabulary.
AMINO_ACIDS = [
    'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
    'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
    'X', 'B', 'U', 'Z', 'O', '.', '-', '|',
]
VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    *AMINO_ACIDS,
    "<mask>",
]
# fmt: on
NUM_VOCAB: int = len(VOCAB)


class Alphabet:
    def __init__(self, tokens: list[str] = VOCAB) -> None:
        self.tokens: list[str] = list(tokens)
        self.tok_to_idx: dict[str, int] = {tok: i for i, tok in enumerate(tokens)}
        self.unk_idx: int = self.tok_to_idx["<unk>"]
        self.bos_idx: int = self.tok_to_idx["<cls>"]
        self.eos_idx: int = self.tok_to_idx["<eos>"]
        self.pad_idx: int = self.tok_to_idx["<pad>"]
        self.mask_idx: int = self.tok_to_idx["<mask>"]
        self.aa_idxs: list[int] = [
            self.tok_to_idx[tok] for tok in AMINO_ACIDS if tok in self.tok_to_idx
        ]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, sequence: str, add_special_tokens: bool = True) -> list[int]:
        encoded = [self.tok_to_idx.get(tok, self.unk_idx) for tok in sequence]
        if add_special_tokens:
            encoded = [self.bos_idx] + encoded + [self.eos_idx]
        return encoded
