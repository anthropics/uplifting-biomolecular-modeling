"""mosaic.fast — the kit's lever modules, installed beside stock `mosaic` (nothing here is imported by stock).
repro_cache (P1) persistent XLA compilation cache + autotune pin · fastload (P2) Boltz2 weight load without the discarded random init ·
frozen (P3) save/load featurized inputs · memlevers (P5) pairformer rematerialisation + chunked triangle attention ·
dead_template (E1) · precision (P6) · halfpair (P7) · flashattn (K1) · layer_unroll (E10) · trimul_layout (F6) · trimul_fused + trimul_provider (F8) ·
transition_fused (F9) · numstate (the numeric-state observer). Classes, switches and numerics per lever: the kit's CHANGES.md."""
