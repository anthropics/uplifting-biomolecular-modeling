"""Public inputs for the package's `warm` verb: barnase-barstar (PDB 1BRS) tiled 1:1 .. 6:6 (199 / 398 / 597 / 796 / 995 / 1194 tokens), single-sequence a3m per chain.
No network access needed: the two sequences are embedded below with their provenance and a sha256 self-check.
Writes an OpenFold3 inference query JSON (one query) + the single-sequence a3m directories.

usage: python public_inputs.py <tiling[,tiling…]: 1to1|2to2|3to3|4to4|5to5|6to6> <outdir>   -> prints one line per query (name, token count);
       several tilings = several queries of ONE query JSON, so one `run_openfold predict` process folds every size in turn
"""
import hashlib, json, os, sys

PROVENANCE = ("RCSB PDB entry 1BRS (barnase-barstar complex): entity 1 barnase (chains A-C, 110 aa), entity 2 barstar C40A/C82A "
              "(chains D-F, 89 aa); sequences from https://www.rcsb.org/fasta/entry/1BRS (fetched 2026-08-23)")
BARNASE = "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"
BARSTAR = "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDALTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS"
EXPECTED_SHA256 = {"barnase": "de6ee3cdb0fd76b4756d3c3bacbc6b22567b1cee43248044f4385c3b9af675c2", "barstar": "a55f4f6f8034de8530775b266225c907bb62c3a97b6e4d2716da47ed535c73bb"}


def single_seq_dir(seq, d):
    """OpenFold3's main-MSA layout: a directory holding <source>.a3m, 'mmseqs_colabfold' being one of upstream's default source names."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "mmseqs_colabfold.a3m"), "w") as fh:
        fh.write(f">101\n{seq}\n")
    return d


MAX_TILING = 6                                          # 6:6 = 1194 tokens (chain ids A..L)
CHAIN_IDS = "ABCDEFGHIJKL"


def tilings_of(spec):
    """'2to2' -> ['2to2']; '1to1,4to4,6to6' -> the list (order kept, duplicates refused by name)."""
    out = [t.strip() for t in str(spec).split(",") if t.strip()]
    assert out, f"no tiling in {spec!r}"
    for t in out:
        n = int(t[0]) if t[:1].isdigit() else 0
        assert t == f"{n}to{n}" and 1 <= n <= MAX_TILING, f"unknown tiling {t!r} (1to1 .. {MAX_TILING}to{MAX_TILING})"
    assert len(set(out)) == len(out), f"duplicate tiling in {spec!r}"
    return out


def _query(tiling, a_dir, b_dir):
    n = int(tiling[0])
    chains = []
    for i in range(n):
        chains.append({"molecule_type": "protein", "chain_ids": [CHAIN_IDS[i]], "sequence": BARNASE, "main_msa_file_paths": [a_dir]})
    for i in range(n):
        chains.append({"molecule_type": "protein", "chain_ids": [CHAIN_IDS[n + i]], "sequence": BARSTAR, "main_msa_file_paths": [b_dir]})
    meta = dict(query=f"brs_{tiling}", n_tokens=n * (110 + 89), target_chains=list(CHAIN_IDS[:n]), binder_chains=list(CHAIN_IDS[n:2 * n]), provenance=PROVENANCE,
                msa_mode="single-sequence (a warm-up input; no MSA search, not an accuracy setting)")
    return meta, {"chains": chains, "use_msas": True, "use_paired_msas": False}


def build_many(tilings, outdir):
    """One query JSON holding one query per tiling (`brs_<tiling>`), the two single-sequence a3m directories shared; returns the metas in order."""
    tilings = tilings_of(tilings) if isinstance(tilings, str) else list(tilings)
    assert len(BARNASE) == 110 and len(BARSTAR) == 89
    for name, seq in (("barnase", BARNASE), ("barstar", BARSTAR)):
        got = hashlib.sha256(seq.encode()).hexdigest()
        assert got == EXPECTED_SHA256[name], f"embedded {name} sequence corrupted ({got})"
    outdir = os.path.abspath(outdir); os.makedirs(outdir, exist_ok=True)
    a_dir = single_seq_dir(BARNASE, os.path.join(outdir, "msas", "barnase")); b_dir = single_seq_dir(BARSTAR, os.path.join(outdir, "msas", "barstar"))
    metas, queries = [], {}
    for t in tilings:
        meta, q = _query(t, a_dir, b_dir)
        metas.append(meta); queries[meta["query"]] = q
    json.dump({"queries": queries}, open(os.path.join(outdir, "q.json"), "w"), indent=1)
    record = metas[0] if len(metas) == 1 else dict(queries=metas, n_tokens=[m["n_tokens"] for m in metas], provenance=PROVENANCE, msa_mode=metas[0]["msa_mode"])
    json.dump(record, open(os.path.join(outdir, "meta.json"), "w"), indent=1)
    return metas


def build(tiling, outdir):
    """One tiling, one query: returns its meta."""
    ts = tilings_of(tiling); assert len(ts) == 1, tiling
    return build_many(ts, outdir)[0]


if __name__ == "__main__":
    for m in build_many(sys.argv[1], sys.argv[2]):
        print(f"[public_inputs] {m['query']} n_tokens={m['n_tokens']} | provenance: {PROVENANCE}")
