#!/usr/bin/env python
"""fast_parse.py — output-identical replacement for helper_scripts/parse_multiple_chains.py (ProteinMPNN @ 8907e66).
The stock script re-reads every PDB once per chain letter (52 letters + 300 digits = 352 reads per file); this reads each
file ONCE, buckets ATOM lines by chain and runs the stock per-chain logic (verbatim) on the bucket. Same dict keys in the
same order, same float parsing, same glob order -> byte-identical parsed.jsonl."""
import argparse, glob, json, os
import numpy as np
alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
aa_3_N = {a: n for n, a in enumerate(alpha_3)}
aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
def N_to_AA(x):
    x = np.array(x)
    if x.ndim == 1: x = x[None]
    return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]
def parse_lines(lines, atoms):
    """stock parse_PDB_biounits body applied to the ATOM lines of ONE chain (already filtered)."""
    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    for line in lines:
        atom = line[12:12+4].strip(); resi = line[17:17+3]; resn = line[22:22+5].strip()
        x, y, z = [float(line[i:(i+8)]) for i in [30, 38, 46]]
        if resn[-1].isalpha(): resa, resn = resn[-1], int(resn[:-1])-1
        else: resa, resn = "", int(resn)-1
        if resn < min_resn: min_resn = resn
        if resn > max_resn: max_resn = resn
        if resn not in xyz: xyz[resn] = {}
        if resa not in xyz[resn]: xyz[resn][resa] = {}
        if resn not in seq: seq[resn] = {}
        if resa not in seq[resn]: seq[resn][resa] = resi
        if atom not in xyz[resn][resa]: xyz[resn][resa][atom] = np.array([x, y, z])
    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn+1):
            if resn in seq:
                for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k], 20))
            else: seq_.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
                        else: xyz_.append(np.full(3, np.nan))
            else:
                for atom in atoms: xyz_.append(np.full(3, np.nan))
        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_))
    except TypeError:
        return 'no_chain', 'no_chain'
init_alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
chain_alphabet = init_alphabet + [str(i) for i in range(300)]
def parse_pdb(path, ca_only=False):
    buckets = {}
    for line in open(path, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()
        if line[:6] == "HETATM" and line[17:17+3] == "MSE":
            line = line.replace("HETATM", "ATOM  "); line = line.replace("MSE", "MET")
        if line[:4] == "ATOM":
            buckets.setdefault(line[21:22], []).append(line)
    atoms = ['CA'] if ca_only else ['N', 'CA', 'C', 'O']
    my_dict = {}; s = 0; concat_seq = ''
    for letter in chain_alphabet:
        if letter not in buckets: continue
        xyz, seq = parse_lines(buckets[letter], atoms)
        if type(xyz) != str:
            concat_seq += seq[0]
            my_dict['seq_chain_'+letter] = seq[0]
            coords_dict_chain = {}
            if ca_only:
                coords_dict_chain['CA_chain_'+letter] = xyz.tolist()
            else:
                coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
            my_dict['coords_chain_'+letter] = coords_dict_chain
            s += 1
    fi = path.rfind("/")
    my_dict['name'] = path[(fi+1):-4]
    my_dict['num_of_chains'] = s
    my_dict['seq'] = concat_seq
    return my_dict
def parse_folder(folder, ca_only=False):
    if folder[-1] != '/': folder = folder + '/'
    out = []
    for biounit in glob.glob(folder + '*.pdb'):      # same glob order as the stock script
        d = parse_pdb(biounit, ca_only)
        if d['num_of_chains'] < len(chain_alphabet): out.append(d)
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--input_path", required=True); ap.add_argument("--output_path", required=True); ap.add_argument("--ca_only", action="store_true")
    a = ap.parse_args()
    with open(a.output_path, 'w') as f:
        for entry in parse_folder(a.input_path, a.ca_only): f.write(json.dumps(entry) + '\n')
