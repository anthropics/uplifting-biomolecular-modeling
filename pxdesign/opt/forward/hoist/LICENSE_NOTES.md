# LICENSE_NOTES.md
* See also `LICENSE` (Apache-2.0 full text, the licence of the upstream code this add-on derives from) and `NOTICE`.
* PXDesign (https://github.com/bytedance/PXDesign) — Apache-2.0.
* Protenix (https://github.com/bytedance/Protenix, tag v0.5.0+pxd) — Apache-2.0. `pxd_xattempt/hoist.py` re-implements the control flow of
  `DiffusionModule.f_forward`, `DiffusionConditioning.forward`, `AtomAttentionEncoder.forward`, `AttentionPairBias.forward` and
  `ConditionedTransitionBlock.forward` (Copyright 2024 ByteDance and/or its affiliates, Apache-2.0) by calling the unmodified upstream sub-modules; the
  portions that mirror upstream lines are identified in comments.
* Model weights: the PXDesign diffusion checkpoint (`pxdesign_v0.1.0.pt`) is downloaded by PXDesign's own scripts from the same project and carries no
  separate weights licence statement; the Protenix checkpoints PXDesign pulls are released under Apache-2.0 including parameters. No weights are included here.
* `inputs/tasks_3targets.json` references three public PDB entries (5O45, 1TNF, 3DI3); their mmCIF files are vendored under `inputs/targets/`
  (`SOURCES.json`: entry, URL, size). PDB archive data are free of all copyright restrictions (CC0 1.0).
