# Source of the test fixture

`fixture_v100.npz` is a byte-identical copy of `tests/fixtures/hg38_chr6_31575665_31575793_multiz100way.npz` from upstream GPN
(https://github.com/songlab-cal/gpn at commit `6f28c81bcbfe7d65cb6d8ece9ce88f87ca583791`, MIT, `Copyright (c) 2022 Song Lab @ Cal`;
the member of that name inside `../../../../stock/gpn-6f28c81b.tar.gz`; licence text `../../../../third_party_licenses/gpn.MIT.txt`).
Its digest is pinned in `../../ACCEL_MANIFEST.json`.

Content, as upstream's `tests/fixtures/README.md` and `tests/fixtures/published_model_baseline.json` record it: the hg38 interval
`chr6:31575665-31575793` (zero-based, half-open; 128 bp) of the 100-vertebrate whole-genome alignment, tokenised with the
vocabulary `-ACGT?` (0..5), as two arrays — `gpn_star_v100_tokens` (128 x 100: hg38 plus 99 vertebrates, the column order of
the GPN-Star `v100` checkpoint) and `gpn_msa_tokens` (128 x 90: the same columns minus the ten closest primates). Upstream read
the slice once from a local store equivalent to the Hugging Face dataset `songlab/multiz100way-pigz` (`99.zarr.tar.gz`, revision
`6a9d42a35e7debbba845979dea6064f14d5cb3f9`; dataset card `license: mit`), a repackaging of `songlab/multiz100way`, whose card
names the source data: the UCSC Genome Browser hg38 `multiz100way` alignment
(http://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/; UCSC's README.txt there: "All the files in this directory are
freely usable for any purpose. For data use restrictions regarding the individual genome assemblies, see
http://genome.ucsc.edu/goldenPath/credits.html."), with the human row replaced by the GRCh38 soft-masked primary assembly of
Ensembl release 107.

Use in this kit: `accel/inputs.py` tiles or crops these real alignment tokens into input batches for the identity self-check
and the engage tests; inputs for other species counts are synthesised and labelled synthetic.
