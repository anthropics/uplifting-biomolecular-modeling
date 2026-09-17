# Public example inputs
Sequences: PDB entry 1BRS (barnase, chains A-C, 110 aa; barstar C40A/C82A, chains D-F, 89 aa), Bacillus amyloliquefaciens; source https://www.rcsb.org/fasta/entry/1BRS (wwPDB, public domain).
Inputs p199/p398/p597/p796/p995.json = barnase:barstar with count n:n for n=1..5 (199*n tokens), single-sequence MSAs (ssmsa/*/pairing.a3m, non_pairing.a3m = the query only) so `--use_msa true` runs with no MSA search and no network access.
The placeholder __KIT__ in the JSONs stands for this kit's directory (opt/forward/flashpairformer); the kit README's `sed` line fills it in.
