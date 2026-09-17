"""Default MMseqs2 commands"""

from pathlib import Path


def write_fasta(sequences: list[tuple[str, str]] | dict[str, str], path: str | Path):
    """Write sequences to a fasta file.

    Parameters
    ----------
    sequences : list[tuple[str, str]] | dict[str, str]
        List of tuples (ID, sequence) or a dictionary mapping IDs to sequences.
    path : str | Path
        Path to the output fasta file.
    """
    if isinstance(sequences, dict):
        sequences = sorted(sequences.items())
    with open(path, "w") as f:
        for seq_id, seq in sequences:
            f.write(f">{seq_id}\n{seq}\n")


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Read sequences from a fasta file.

    Parameters
    ----------
    path : str | Path
        Path to the input fasta file.

    Returns
    -------
    list[tuple[str, str]]
        List of tuples (ID, sequence) read from the fasta file.
    """
    sequences: list[tuple[str, str]] = []
    with open(path) as f:
        seq_id: str | None = None
        seq_lines: list[str] = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    sequences.append((seq_id, "".join(seq_lines)))
                seq_id = line[1:]  # Remove '>'
                seq_lines = []
            else:
                seq_lines.append(line)
        # Add the last sequence
        if seq_id is not None:
            sequences.append((seq_id, "".join(seq_lines)))

    return sequences
