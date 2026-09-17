# Skip the unused confidence head at run time (`agk.enable_skip_unused_confidence(model)`)

**Applies to:** the pinned Biohub transformers fork (`models/esmfold2`) and the esm cookbook's `binder_design` (STOCK.md §Pin). Applied at run
time by patching bound methods of the loaded model; no source edits.

## What
`ESMFold2ExperimentalModel.forward` accepts `calculate_confidence` only through **kwargs and ignores it, so the confidence head (a full
pairformer stack under no_grad) runs on every design step although the design loss never reads plddt / pae / iptm. The lever wraps forward so
that `calculate_confidence=False` temporarily detaches the confidence head; confidence steps (`calculate_confidence=True`) are unchanged.

## Numerics class
exact: bitwise (distogram logits, atom coordinates, loss and the gradient w.r.t. the soft sequence), with fewer launches and host
synchronisations per step.

## Equivalent source change (one line, `modeling_esmfold2_experimental.py`, `ESMFold2ExperimentalModel.forward`)
```diff
-            if self.confidence_head is not None:
+            if self.confidence_head is not None and kwargs.get("calculate_confidence", True):
```
