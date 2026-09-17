# Third-party notices — `borzoi` kit

The kit's own files under `opt/` adapt, restate or wrap code from the two upstream projects listed here. The projects
themselves are carried verbatim as `stock/borzoi-5c93582.tar.gz` and `stock/baskerville-544073b.tar.gz`, and each archive
contains that project's `LICENSE` member; plain copies of those licence texts sit in `third_party_licenses/`. The kit's
own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file
concerns the third-party portions only.

## Borzoi

- Upstream: https://github.com/calico/borzoi at commit `5c9358222b5026abb733ed5fb84f3f6c77239b37`.
- Licence: Apache License, Version 2.0 — the upstream `LICENSE` file. No `NOTICE` file is published upstream at this commit.
- Copyright line: the upstream `LICENSE` file carries no copyright line; the upstream source-file headers read
  `Copyright 2017 Calico LLC` (as in `src/scripts/borzoi_sad.py`).
- Kit files concerned:
  - `opt/datapath/pipeline_tf/v17/borzoi_sad.py` — a modified copy of upstream `src/scripts/borzoi_sad.py` (Apache-2.0): the whole
    upstream file with its copyright and licence header retained, plus the kit's inserted blocks, each bracketed
    `# ---- kit … ----` (insertions only; no upstream line is deleted or edited). `opt/datapath/pipeline_tf/build.py` renders the
    file from the archive member and the kit blocks, so the inserted text is the complete statement of the changes (Apache-2.0
    §4(b): this file was modified; every other Borzoi file is used unmodified from the archive).
  - `opt/datapath/pipeline_tf/v17/kitlib/writer.py` — restates the statistics of `write_snp_len` from upstream
    `src/scripts/borzoi_sad.py`, computing only the intermediates the requested statistics use.
  - `opt/borzoi_opt/tests/_fixtures.py` — reproduces the option grammar of upstream `src/scripts/borzoi_sad.py` in a
    stand-in script used by the CPU tests.
- Licence text: `third_party_licenses/Borzoi-Calico.Apache-2.0.txt`, byte-identical to the `LICENSE` member of
  `stock/borzoi-5c93582.tar.gz`.

## Baskerville

- Upstream: https://github.com/calico/baskerville at commit `544073b87245d9f43ba63442c75bdbf32a9f8720` (tag `v1.0.0`).
- Licence: Apache License, Version 2.0 — the upstream `LICENSE` file. No `NOTICE` file is published upstream at this commit.
- Copyright line: the upstream `LICENSE` file carries no copyright line; the upstream source-file headers read
  `Copyright 2023 Calico LLC` (as in `src/baskerville/dna.py`).
- Kit files concerned:
  - `opt/datapath/pipeline_tf/v17/kitlib/onehot.py` — re-implements `dna_1hot` from upstream `src/baskerville/dna.py` as a
    lookup table; its sequence trim/pad lines restate upstream's.
  - `opt/datapath/pipeline_tf/tests/test_pipeline_tf.py` — `stock_dna_1hot` transcribes `dna_1hot` from upstream
    `src/baskerville/dna.py` as the test's reference implementation.
  - `opt/datapath/pipeline_tf/v17/kitlib/forward.py`, `opt/datapath/pipeline_tf/v17/kitlib/sad_post.py` and
    `opt/borzoi_opt/_autoload.py` — wrap `baskerville.seqnn.SeqNN.__call__` and `baskerville.dna.dna_1hot` at run time;
    they contain no upstream code.
- Licence text: `third_party_licenses/Baskerville-Calico.Apache-2.0.txt`, byte-identical to the `LICENSE` member of
  `stock/baskerville-544073b.tar.gz`.

## Data files

- Outside `stock/`: `tests/inputs/example_snvs.vcf` — six single-nucleotide variants in hg38 coordinates (chromosome, position,
  reference and alternate allele with GTEx-style identifiers; no sample, genotype or phenotype fields), the example input of the
  README commands. No other data file sits outside `stock/`.
- Inside `stock/baskerville-544073b.tar.gz`, as committed upstream for that project's unit tests and carried unmodified:
  `tests/data/eval/model.h5` (a small trained Keras test model — not the Borzoi weights, and never read by this kit),
  `tests/data/hg38_1m.fa.gz` with its indexes (a one-megabase excerpt of the public human reference assembly hg38), `tests/data/tiny/{hg38,mm10}/`
  (TFRecord shards and BED intervals of a miniature training set on the public hg38 / mm10 assemblies) and the VCF / BED files under
  `tests/data/{ism,snp}/` (a handful of variant and interval lines). They are part of the Apache-2.0-licensed repository at the pin.
- Inside `stock/borzoi-5c93582.tar.gz`, as committed upstream and carried unmodified: `data/sequences_{human,mouse}.bed.gz` and
  `data/targets_{human,mouse}.txt.gz` (the model's training-interval lists and target-track tables, which name public ENCODE, GTEx,
  FANTOM5 and GEO datasets by accession but contain none of their data), `examples/*.gtf` and `tutorials/**/HBE1_example.gtf` (single-gene
  excerpts of the public GENCODE annotation), the tutorial VCF / CSV files under `tutorials/` (a few variant lines each), `examples/params.json`
  and `examples/targets_*.txt` (model configuration and target lists the README commands use) and `borzoi_logo.png`. They are part of the
  Apache-2.0-licensed repository at the pin.
- Neither archive and no file of this tree contains the Borzoi model weights (`STOCK.md` says where `run.sh install --weights` obtains
  them), a genome assembly beyond that test excerpt, or any sequence database.
