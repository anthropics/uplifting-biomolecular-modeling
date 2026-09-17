# Sources of the example inputs

`variants.parquet` — 8 single-nucleotide variants on hg38, columns `chrom`, `pos` (one-based), `ref`, `alt` and nothing else:
`1:1425822 C>G`, `1:1615869 C>T`, `2:5718743 A>G`, `2:5718745 C>T`, `3:213423 A>G`, `3:384380 A>C`, `5:420942 A>C`, `5:850655 C>T`.
They are the first two rows for each of chromosomes 1, 2, 3 and 5 of `test.parquet` in the Hugging Face dataset
`songlab/omim_traitgym` at revision `9317562efb8c61f31bb5fc62a19f731b2f8b4384` — the table and revision that upstream GPN's notebook
`colabs/gpn_star_precomputed_scores.ipynb` (inside `../stock/gpn-6f28c81b.tar.gz`) reads, described there as 338 OMIM regulatory
variants and 3,042 controls. All eight rows carry `label = false` in that table (control variants of the matched set) and an empty
`OMIM` field; the table's other columns (`OMIM`, `consequence`, `label`, `tss_dist`, `match_group`) are not carried, so no OMIM
record content is present here. Dataset card licence field: `license: mit` (no separate licence text is published with the dataset).
What was done to them: row selection and column selection only; the values are unchanged.

`msa/100/all.zarr/` — a sparse copy of the GPN-Star 100-vertebrate alignment store `99.zarr.zip` of the Hugging Face dataset
`songlab/multiz100way` (dataset card licence field: `license: mit`): the root and per-chromosome zarr metadata files and the eight
chunk files `1/2784.0`, `1/3155.0`, `1/3156.0`, `2/11169.0`, `3/416.0`, `3/750.0`, `5/822.0`, `5/1661.0` (512 alignment columns by
100 species each), every byte as published; `README.md` beside this file gives the cutting recipe. The dataset card names the
source data: the UCSC Genome Browser hg38 `multiz100way` alignment (Multiz alignment of 100 vertebrate genome assemblies,
http://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/), with the human row replaced by the GRCh38 soft-masked primary
assembly of Ensembl release 107. UCSC's README.txt in that download directory states: "All the files in this directory are
freely usable for any purpose. For data use restrictions regarding the individual genome assemblies, see
http://genome.ucsc.edu/goldenPath/credits.html." What was done to them: selection of whole files; nothing re-encoded.
