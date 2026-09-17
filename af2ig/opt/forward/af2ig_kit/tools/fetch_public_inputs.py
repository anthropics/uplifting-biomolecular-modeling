#!/usr/bin/env python3
"""fetch_public_inputs.py — rebuild the public binder/target complex pack from the RCSB PDB with the Python standard library only
(no curl/wget; urllib with retries; sha256 check; provenance line printed per file).

usage:  python fetch_public_inputs.py [--out DIR] [--only ID,ID,...] [--no-check]
result: DIR/pdbs/<id>.pdb (binder = chain A first, target chains B.. after, residues renumbered from 1 per chain, ATOM records of standard
        residues only, first alt-loc), DIR/index.csv, and a final line 'public inputs: N/N files sha256-checked'.
Entries: 1BRS barnase–barstar tiled k:k (k=1..4; copies displaced 70 Å along x), 5JDS nanobody–PD-L1, 6M0J RBD–ACE2 (full and ACE2 19-330),
1YY9 EGFR domain III vs cetuximab Fab (2 target chains).  PDB data: wwPDB/RCSB usage policy (CC0).
"""
import argparse, collections, csv, hashlib, os, sys, time, urllib.request

RCSB = "https://files.rcsb.org/download/{}.pdb"
# sha256 of the PROCESSED files produced by this script (raw PDB entries may be re-released by the wwPDB; if a hash changes, the script
# prints MISMATCH with both hashes so the kit's expected table can be updated deliberately).
EXPECTED_SHA256 = {
    "1brs_bb_1to1": "a2fac360b1c166912e4c31b97543d7f8ff05ec79a155020d1ccb5775bd26e933",
    "1brs_bb_2to2": "a7508f5d25dd1667ac4fe1ee5d2e9bc5397e30575a95fabc573b343400cc19d7",
    "1brs_bb_3to3": "10ef154cd3b09c650adb90c829de9f79c4bf846f99553d2262950d9a5e92c14a",
    "1brs_bb_4to4": "05ef08e1fc99648db56ae812edc875706113b9a400e0366aff75ceba7ca0542b",
    "1yy9_egfrd3_fab": "35d5a73c01a7c2c428b9fce2c215d6d2fdab715e226dbc5ac1ab044972c56a9f",
    "5jds_nb_pdl1": "2b9feedbe16bda5d2e753bf910268faebcff84c1af945dc6572fe25caad8288f",
    "6m0j_rbd_ace2": "1167ef829349633730842bbe3605a19628092d5894c8efe0822f9c6081329b09",
    "6m0j_rbd_ace2n": "4c974ecbc3fa51420ff676b42d41931af1aee134812127d8b30f6e9f450a754d",
}
AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M',
       'PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}


def fetch(pdb_id, cache_dir, retries=5):
    os.makedirs(cache_dir, exist_ok=True)
    fn = os.path.join(cache_dir, pdb_id + ".pdb")
    if os.path.exists(fn) and os.path.getsize(fn) > 0:
        return fn
    url = RCSB.format(pdb_id)
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=60) as r, open(fn + ".part", "wb") as f:
                f.write(r.read())
            os.replace(fn + ".part", fn)
            return fn
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


def parse_chains(fn):
    chains = collections.OrderedDict()
    for line in open(fn):
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        if line[16] not in (" ", "A"):
            continue
        if line[17:20] not in AA3:
            continue
        line = line[:16] + " " + line[17:]
        chains.setdefault(line[21], []).append(line.rstrip("\n"))
    return chains


def seq_of(lines):
    seen = collections.OrderedDict()
    for l in lines:
        seen[(l[22:26], l[26])] = AA3[l[17:20]]
    return "".join(seen.values())


def renumber(lines, chain_id):
    out, cur, resi = [], None, 0
    for l in lines:
        key = (l[22:26], l[26])
        if key != cur:
            cur = key
            resi += 1
        out.append(l[:21] + chain_id + f"{resi:4d}" + " " + l[27:])
    return out


def translate(lines, dx):
    out = []
    for l in lines:
        x, y, z = float(l[30:38]) + dx, float(l[38:46]), float(l[46:54])
        out.append(l[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + l[54:])
    return out


def slice_res(lines, lo, hi):
    return [l for l in lines if lo <= int(l[22:26]) <= hi]


def write_complex(fn, chain_blocks):
    serial, out = 1, []
    for cid, lines in chain_blocks:
        for l in renumber(lines, cid):
            out.append(l[:6] + f"{serial % 100000:5d}" + l[11:])
            serial += 1
        out.append("TER")
    out.append("END")
    with open(fn, "w") as f:
        f.write("\n".join(out) + "\n")


def build(out_dir, only=None):
    raw = os.path.join(out_dir, "raw")
    pdbs = os.path.join(out_dir, "pdbs")
    os.makedirs(pdbs, exist_ok=True)
    ids = "ABCDEFGHIJ"
    rows = []

    def want(x):
        return only is None or x in only

    if any(want(f"1brs_bb_{k}to{k}") for k in (1, 2, 3, 4)):
        c = parse_chains(fetch("1BRS", raw))
        barnase, barstar = c["A"], c["D"]
        for k in (1, 2, 3, 4):
            cid = f"1brs_bb_{k}to{k}"
            if not want(cid):
                continue
            blocks = [barstar] + [translate(barnase, 70.0 * i) for i in range(k)] + [translate(barstar, 70.0 * i) for i in range(1, k)]
            named = [(ids[j], b) for j, b in enumerate(blocks)]
            write_complex(os.path.join(pdbs, cid + ".pdb"), named)
            rows.append(dict(id=cid, pdb="1BRS", binder="barstar chain D (first copy) as chain A",
                             target=f"barnase x{k} + barstar x{k-1}, copies displaced 70 A along x",
                             binder_len=len(seq_of(barstar)), n_target_chains=len(named) - 1,
                             total_residues=sum(len(seq_of(b)) for _, b in named)))
    if want("5jds_nb_pdl1"):
        c = parse_chains(fetch("5JDS", raw))
        nb, pdl1 = c["B"], c["A"]
        write_complex(os.path.join(pdbs, "5jds_nb_pdl1.pdb"), [("A", nb), ("B", pdl1)])
        rows.append(dict(id="5jds_nb_pdl1", pdb="5JDS", binder="nanobody KN035 (chain B) as chain A", target="PD-L1 IgV domain (chain A) as chain B",
                         binder_len=len(seq_of(nb)), n_target_chains=1, total_residues=len(seq_of(nb)) + len(seq_of(pdl1))))
    if want("6m0j_rbd_ace2") or want("6m0j_rbd_ace2n"):
        c = parse_chains(fetch("6M0J", raw))
        rbd, ace2 = c["E"], c["A"]
        if want("6m0j_rbd_ace2"):
            write_complex(os.path.join(pdbs, "6m0j_rbd_ace2.pdb"), [("A", rbd), ("B", ace2)])
            rows.append(dict(id="6m0j_rbd_ace2", pdb="6M0J", binder="SARS-CoV-2 RBD (chain E) as chain A", target="ACE2 peptidase domain (chain A) as chain B",
                             binder_len=len(seq_of(rbd)), n_target_chains=1, total_residues=len(seq_of(rbd)) + len(seq_of(ace2))))
        if want("6m0j_rbd_ace2n"):
            a = slice_res(ace2, 19, 330)
            write_complex(os.path.join(pdbs, "6m0j_rbd_ace2n.pdb"), [("A", rbd), ("B", a)])
            rows.append(dict(id="6m0j_rbd_ace2n", pdb="6M0J", binder="SARS-CoV-2 RBD (chain E) as chain A", target="ACE2 residues 19-330 (chain A) as chain B",
                             binder_len=len(seq_of(rbd)), n_target_chains=1, total_residues=len(seq_of(rbd)) + len(seq_of(a))))
    if want("1yy9_egfrd3_fab"):
        c = parse_chains(fetch("1YY9", raw))
        d3 = slice_res(c["A"], 310, 501)
        write_complex(os.path.join(pdbs, "1yy9_egfrd3_fab.pdb"), [("A", d3), ("B", c["C"]), ("C", c["D"])])
        rows.append(dict(id="1yy9_egfrd3_fab", pdb="1YY9", binder="EGFR domain III residues 310-501 (chain A) as chain A",
                         target="cetuximab Fab light (C) + heavy (D) as chains B,C", binder_len=len(seq_of(d3)), n_target_chains=2,
                         total_residues=len(seq_of(d3)) + len(seq_of(c["C"])) + len(seq_of(c["D"]))))
    rows.sort(key=lambda r: r["total_residues"])
    with open(os.path.join(out_dir, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="public_pack")
    ap.add_argument("--only", default=None, help="comma-separated ids")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    rows = build(a.out, only)
    ok = 0
    for r in rows:
        p = os.path.join(a.out, "pdbs", r["id"] + ".pdb")
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        exp = EXPECTED_SHA256.get(r["id"])
        status = "OK" if (exp == h) else ("MISMATCH expected " + str(exp)) if exp else "NO-EXPECTED"
        if exp == h or a.no_check:
            ok += 1
        print(f"provenance: {r['id']} <- RCSB PDB {r['pdb']} ({RCSB.format(r['pdb'])}); {r['total_residues']} residues; sha256 {h} {status}")
    print(f"public inputs: {ok}/{len(rows)} files sha256-checked")
    sys.exit(0 if ok == len(rows) else 1)


if __name__ == "__main__":
    main()
