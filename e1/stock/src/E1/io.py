import biotite.sequence.io.fasta as fasta


def read_fasta_sequences(path: str) -> dict[str, str]:
    sequences: dict[str, str] = {}
    with open(path, "r") as f:
        for header, sequence in fasta.FastaFile.read(f).items():
            sequences[header] = sequence
    return sequences


def write_fasta_sequences(path: str, sequences: dict[str, str]) -> None:
    with open(path, "w") as f:
        for header, sequence in sequences.items():
            f.write(f">{header}\n{sequence}\n")
