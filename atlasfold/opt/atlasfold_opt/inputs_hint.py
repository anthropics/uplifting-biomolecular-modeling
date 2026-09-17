"""The run's inputs, read ONCE at start-up from the stock argv (``--input-fasta``) before activation: record count, token counts and the padded
length buckets the stock runner will use (``atlasfold.common.featurize.DEFAULT_BUCKETS``).  A start-up fact for levers whose policy is process-level
and must be decided before the first CUDA allocation (``alloc_expandable`` beside the CUDA-graph lever: the allocator configuration cannot change per
item).  Never raises: an unreadable / absent FASTA gives ``{}`` and every lever keeps its hint-free behaviour."""
import os
from typing import Dict, List, Optional, Sequence

BUCKETS = (32, 64, 128, 192, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048)   # atlasfold.common.featurize.DEFAULT_BUCKETS (read from the stock tree when importable)


def buckets() -> Sequence[int]:
    try:
        from atlasfold.common import featurize as F                         # a light module (numpy): no CUDA initialisation
        return tuple(int(b) for b in F.DEFAULT_BUCKETS)
    except Exception:  # noqa: BLE001
        return BUCKETS


def bucket_of(tokens: int, table: Sequence[int] = None) -> int:
    """The smallest bucket >= tokens (the stock rule); above the largest bucket the token count itself."""
    for b in (table or buckets()):
        if tokens <= b:
            return int(b)
    return int(tokens)


def fasta_path(argv: Sequence[str]) -> Optional[str]:
    """``--input-fasta X`` / ``--input-fasta=X`` of the stock argv, or None."""
    for i, w in enumerate(argv or ()):
        if w == "--input-fasta" and i + 1 < len(argv):
            return argv[i + 1]
        if w.startswith("--input-fasta="):
            return w.split("=", 1)[1]
    return None


def read_records(path: str) -> List[Dict]:
    """[{name, tokens}] — one per FASTA record; tokens = residues over all ':'-joined chains (the multimer form; a monomer record has one chain)."""
    out, name, seq = [], None, []
    def flush():
        if name is not None:
            s = "".join(seq).replace(" ", "")
            out.append({"name": name, "tokens": sum(len(c) for c in s.split(":"))})
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush(); name = (line[1:].split() or ["unnamed"])[0]; seq = []
            else:
                seq.append(line)
    flush()
    return out


def from_argv(argv: Sequence[str]) -> Dict:
    """{path, records, max_tokens, min_bucket, max_bucket} for the run's --input-fasta, {} when absent / unreadable / empty."""
    path = fasta_path(list(argv or ()))
    if not path or not os.path.isfile(path):
        return {}
    try:
        recs = read_records(path)
    except Exception:  # noqa: BLE001
        return {}
    if not recs:
        return {}
    table = buckets()
    bks = [bucket_of(r["tokens"], table) for r in recs]
    return {"path": path, "records": len(recs), "max_tokens": max(r["tokens"] for r in recs), "min_bucket": min(bks), "max_bucket": max(bks)}
