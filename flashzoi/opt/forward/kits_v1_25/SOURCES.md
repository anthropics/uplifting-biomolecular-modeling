# Sources of the data files in `opt/forward/kits_v1_25`

## `canary_window_0.npz`

- Content: one model input window and nothing else — a NumPy archive with a single array under the key `x`, shape `(4, 524288)`,
  dtype `uint8`: the one-hot encoding (rows A, C, G, T; one 1 per column) of a genomic sequence. It carries no model outputs and no
  expected results; `flashzoi_opt`'s `warm` runs one forward on it to fill the kernel cache.
- Source: the human reference genome assembly GRCh38 (UCSC `hg38`), chr1:143,315,785-143,840,073 as a 0-based, end-exclusive
  interval (524,288 bp; its central 196,608 bp are chr1:143,479,625-143,676,233, the window `README.md` names). The encoded bases
  equal the assembly sequence over the whole interval; soft-masking is not represented. Nothing else went into the file.
- What was done: the bases of that interval were one-hot encoded and stored as a compressed NumPy `.npz` archive; no base was
  altered.
- Terms: GRCh38 is public reference data of the Genome Reference Consortium, released through GenBank; the `README.txt` UCSC
  publishes beside its `hg38` sequence files states that NCBI places no restrictions on the use or distribution of the GenBank
  data, while some submitters may claim rights in portions of the data they submitted. No licence text accompanies the assembly.
  The kit uses no UCSC annotation, conservation or alignment track. `../../../THIRD_PARTY_NOTICES.md` carries the same entry.

## `engines/flashzoi/kits/v1_25/class_records/a100.json`

- Content: a GPU device-class record of the kit (class name, compute capability, memory size, device name strings, the digest of
  the kit's kernel module). It contains no third-party data.
