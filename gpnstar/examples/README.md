# gpnstar example — 8 variants and the alignment chunks they need

`variants.parquet` — eight illustrative hg38 SNVs used to demonstrate the input format and the reference-allele check (columns chrom, pos, ref, alt; pos is 1-based; chrom is named as the alignment store
names its arrays — here `1`, `2`, `3`, `5`, no `chr` prefix), two per chromosome; ref equals the human base of the alignment at each
position (checked when this example was cut).

`msa/100/all.zarr/` — a sparse zarr (v2) directory store: the root and per-chromosome metadata files of the public 100-vertebrate
alignment store plus ONLY the 8 chunk files the eight 128-bp windows touch, every byte copied verbatim (nothing re-encoded). Source:
Hugging Face dataset songlab/multiz100way, file 99.zarr.zip (dataset card: license mit; alignment origin: UCSC hg38 multiz100way).
Recipe: read the zip's central directory with HTTP range requests, select the chunk entries covering each window from the array's chunk
grid (512 rows per chunk), fetch those entries by range, write them under the same keys. Total 0.14 MiB. Every other region of this store
is absent and reads as fill bytes — it serves these eight variants only.

Run: README.md 'Run' uses these two paths (`--input-path examples/variants.parquet --msa-path examples/msa`) once as stock
(`gpn star vep … --tf32`) and once with `GPNSTAR_OPT=exact` in front, and compares the two output files with `cmp` — identical bytes on the same card.

Scores computed against this sparse store are identical to scores computed against a larger cut of the same store holding these windows
(same chunk bytes, same reader): the sparse store changes what is present, not what is read.
