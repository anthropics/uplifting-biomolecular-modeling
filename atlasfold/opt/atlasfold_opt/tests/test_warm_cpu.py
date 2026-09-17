"""warm — the fold's size covers every lever floor whose kernel compiles at first use (CPU: the FASTA text only)."""
from atlasfold_opt import cli


def test_warm_fasta_default_is_the_640_bucket(monkeypatch):
    monkeypatch.delenv(cli.WARM_TOKENS_ENV, raising=False)
    txt = cli.warm_fasta_text()
    name, seq, tail = txt.split("\n")
    assert name == ">warm" and tail == "" and len(seq) == cli.WARM_TOKENS_DEFAULT == 640
    assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")


def test_warm_tokens_word_and_floor(monkeypatch):
    monkeypatch.setenv(cli.WARM_TOKENS_ENV, "1024")
    assert len(cli.warm_fasta_text().split("\n")[1]) == 1024
    monkeypatch.setenv(cli.WARM_TOKENS_ENV, "3")
    assert len(cli.warm_fasta_text().split("\n")[1]) == 16          # never an empty / degenerate record
    monkeypatch.setenv(cli.WARM_TOKENS_ENV, "not-a-number")
    assert len(cli.warm_fasta_text().split("\n")[1]) == 640
    assert len(cli.warm_fasta_text(64).split("\n")[1]) == 64
