"""inputs_hint — the run's start-up facts from the stock argv (--input-fasta): records, token counts, padded buckets (CPU, tmp files)."""
from atlasfold_opt import inputs_hint as IH


def test_fasta_records_tokens_and_buckets(tmp_path):
    fa = tmp_path / "in.fasta"
    fa.write_text(">a some header\n" + "A" * 300 + ":" + "C" * 320 + "\n>b\n" + "D" * 1100 + "\n" + "E" * 100 + "\n\n>c\nKK\n")
    recs = IH.read_records(str(fa))
    assert [(r["name"], r["tokens"]) for r in recs] == [("a", 620), ("b", 1200), ("c", 2)]
    h = IH.from_argv(["multimer", "--input-fasta", str(fa), "--out-dir", "x"])
    assert h["records"] == 3 and h["max_tokens"] == 1200 and h["min_bucket"] == 32 and h["max_bucket"] == 1280
    assert IH.from_argv(["multimer", f"--input-fasta={fa}"])["records"] == 3
    assert IH.bucket_of(1201, IH.BUCKETS) == 1280 and IH.bucket_of(2049, IH.BUCKETS) == 2049 and IH.bucket_of(1, IH.BUCKETS) == 32


def test_absent_or_unreadable_is_empty(tmp_path):
    assert IH.from_argv(["multimer", "--out-dir", "x"]) == {}
    assert IH.from_argv(["multimer", "--input-fasta", str(tmp_path / "missing.fa")]) == {}
    (tmp_path / "empty.fa").write_text("\n")
    assert IH.from_argv(["--input-fasta", str(tmp_path / "empty.fa")]) == {}
