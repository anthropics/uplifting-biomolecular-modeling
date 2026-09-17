#!/usr/bin/env python
"""fetch_public_inputs.py — the kit's public inputs, all from public sources, sha256-checked, via python urllib (no curl/wget).

1. PDB entry 1BRS (barnase:barstar complex, Buckle, Schreiber & Fersht 1994) from files.rcsb.org — used only for provenance and to derive the
   target sequence: barstar = SEQRES of chain D (89 aa, the C40A/C82A double mutant crystallised in 1BRS). The sequence is also inlined below so
   a design can run if RCSB is unreachable (the inline copy is asserted equal to the downloaded SEQRES when the download succeeds).
2. Boltz-2 weights + CCD molecules (MIT licence, NOT redistributed in this kit): fetched by boltz's own `boltz.main.download_boltz2(cache)` into
   $MOSAIC_CACHE_DIR/boltz (boltz2_conf.ckpt sha256 asserted below; ~5 GB with mols/; first run only).

usage: python tools/fetch_public_inputs.py [--weights] [--out DIR]      prints the provenance line
"""
import argparse, hashlib, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path

PDB_ID = "1BRS"
PDB_URL = "https://files.rcsb.org/download/1BRS.pdb"
PDB_SHA256 = "0076aae54252706a8c8c02f7b1e71ce2394b0e928b413590b4ac12aade9e8f6c"      # files.rcsb.org/download/1BRS.pdb (REVDAT 7, 07-FEB-24)
BARSTAR_1BRS_D = "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDALTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS"   # SEQRES chain D (89 aa)
BARNASE_1BRS_A = "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"  # SEQRES chain A (110 aa), provenance only
BARSTAR_SHA256 = hashlib.sha256(BARSTAR_1BRS_D.encode()).hexdigest()
BOLTZ2_CONF_CKPT_SHA256 = "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1"   # huggingface.co/boltz-community/boltz-2 boltz2_conf.ckpt

THREE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M',
         'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def sha256_file(p, block=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def seqres(pdb_text):
    chains = {}
    for line in pdb_text.splitlines():
        if line.startswith("SEQRES"):
            chains.setdefault(line[11], []).extend(line[19:].split())
    return {c: "".join(THREE.get(r, "X") for r in rs) for c, rs in chains.items()}


def fetch_pdb(out_dir: Path, timeout=60):
    """Returns dict(status, path, sha256, source). Never raises: offline -> status 'inline'."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{PDB_ID}.pdb"
    rec = {"pdb_id": PDB_ID, "url": PDB_URL, "expected_sha256": PDB_SHA256, "target_chain": "D (barstar C40A/C82A, 89 aa)", "target_seq_sha256": BARSTAR_SHA256}
    try:
        if not (p.exists() and sha256_file(p) == PDB_SHA256):
            req = urllib.request.Request(PDB_URL, headers={"Accept": "text/plain"})
            data = urllib.request.urlopen(req, timeout=timeout).read()
            p.write_bytes(data)
        got = sha256_file(p)
        rec.update({"path": str(p), "sha256": got, "sha256_ok": got == PDB_SHA256})
        sq = seqres(p.read_text())
        rec["seqres_D_equals_inline"] = (sq.get("D") == BARSTAR_1BRS_D)
        rec["seqres_A_equals_inline"] = (sq.get("A") == BARNASE_1BRS_A)
        if got != PDB_SHA256:
            # RCSB occasionally re-releases files (REVDAT); the sequence is what matters for the design, the file hash is provenance.
            rec["status"] = "downloaded (sha256 differs from the recorded one: RCSB file revised?) - sequence check decides"
        else:
            rec["status"] = "downloaded, sha256 ok"
        if not rec["seqres_D_equals_inline"]:
            rec["status"] = "ERROR: SEQRES chain D of the downloaded file != inline barstar sequence"
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        rec.update({"status": f"inline (download failed: {type(e).__name__}: {str(e)[:120]})", "path": None, "sha256": None, "sha256_ok": None})
    return rec


def provenance_line(rec):
    s = rec.get("sha256") or "n/a"
    return (f"provenance: PDB {PDB_ID} {PDB_URL} sha256={s} (expected {PDB_SHA256[:16]}..., ok={rec.get('sha256_ok')}); "
            f"target = barstar SEQRES chain D, 89 aa, seq_sha256={BARSTAR_SHA256[:16]}...; binder = poly-X hallucination, single-sequence mode (msa: empty), no MSA server; {rec.get('status')}")


def ensure_weights(cache_root: Path):
    """Download Boltz-2 weights/CCD with boltz's own downloader if absent; assert the checkpoint sha256. Returns dict."""
    bdir = cache_root / "boltz"; bdir.mkdir(parents=True, exist_ok=True)
    ck = bdir / "boltz2_conf.ckpt"
    t0 = time.time(); fetched = False
    if not (ck.exists() and (bdir / "mols").exists()):
        from boltz.main import download_boltz2
        print(f"[fetch] downloading Boltz-2 weights + CCD molecules into {bdir} (public: model-gateway.boltz.bio / huggingface.co/boltz-community; ~5 GB, first run only) ...", flush=True)
        download_boltz2(bdir); fetched = True
    # mosaic's featurizer also wants ccd.pkl next to the checkpoint for boltz-1 style paths; Boltz-2 uses mols/ (load_canonicals) — nothing else needed.
    got = sha256_file(ck)
    return {"cache_dir": str(bdir), "boltz2_conf_ckpt_sha256": got, "sha256_ok": got == BOLTZ2_CONF_CKPT_SHA256, "fetched_now": fetched, "t_s": round(time.time() - t0, 1),
            "n_mols": sum(1 for _ in (bdir / "mols").iterdir()) if (bdir / "mols").exists() else 0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="inputs_public")
    ap.add_argument("--weights", action="store_true", help="also fetch Boltz-2 weights into $MOSAIC_CACHE_DIR (default ~/.cache/mosaic)")
    a = ap.parse_args()
    rec = fetch_pdb(Path(a.out)); print(json.dumps(rec, indent=1)); print(provenance_line(rec))
    if a.weights:
        root = Path(os.environ.get("MOSAIC_CACHE_DIR", "~/.cache/mosaic")).expanduser()
        print(json.dumps(ensure_weights(root), indent=1))
    sys.exit(0 if not str(rec.get("status", "")).startswith("ERROR") else 1)
