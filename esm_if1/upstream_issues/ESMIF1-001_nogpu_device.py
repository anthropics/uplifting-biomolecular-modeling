"""ESMIF1-001 — ``sample_sequences.py --nogpu`` (single-chain route) samples on ``cuda`` regardless and fails.

Applied only on request: ``run.sh design --upstream-fix ESMIF1-001`` (every mode, ``off`` included; opt/esm_if1_opt/upstream_fix.py loads
this file by path in the process that runs the model — the proven stock child under ``off``, the kit child under ``fast`` — after the
stock proof and before the model is used, and prints ``[esm_if1-opt] UPSTREAM-FIX ESMIF1-001 applied …``; opt_manifest.json
``kit.upstream_fix`` records it). Nothing imports this file otherwise; no lever, mode or config references it. Without the flag every mode
runs upstream's code exactly as shipped.

WHAT UPSTREAM DOES. ``examples/inverse_folding/sample_sequences.py`` ``sample_seq_singlechain`` (l.20-41) keeps the model on the CPU when
``--nogpu`` is given (l.21-23) but calls ``model.sample(coords, temperature=args.temperature, device=torch.device('cuda'))`` (l.34) with the
device hard-wired; ``GVPTransformerModel.sample`` (``esm/inverse_folding/gvp_transformer.py`` l.104-131) moves the batch and the token
matrix to that device and runs the CPU-resident encoder on CUDA tensors. The same file's multichain route does it right:
``multichain_util.sample_sequence_in_complex`` l.95 takes ``device = next(model.parameters()).device``.

EVIDENCE. On the pinned stack (torch 2.4.0+cu121, H100): ``run.sh design --mode off 5YH2.pdb --chain C --num-samples 1 --nogpu`` ends the
script with ``RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (… index_select)``
before the first sequence; on a host without CUDA the same call fails at ``.to(device)``. Without ``--nogpu`` the route is unaffected.

WHAT THE FIX CHANGES. One method rebinding: ``GVPTransformerModel.sample`` is wrapped so that its ``device`` argument is the model's own
parameter device (upstream's multichain idiom, l.95) whatever the caller passed. No file is edited. Under ``fast`` the batched driver already
samples on the model's device, so the rebinding changes nothing there (the line is printed all the same: the flag applies to every mode).

EXPECTED EFFECT ON OUTPUTS. None on a GPU run without ``--nogpu`` (the model is on ``cuda`` and so was the hard-wired device: the same
statements on the same device). With ``--nogpu`` the single-chain route now runs, on the CPU, instead of raising.
"""
ID = "ESMIF1-001"
WORDS = "model.sample() samples on the model's own device (upstream multichain_util l.95's idiom) instead of a hard-wired cuda"


def apply():
    """Rebind ``GVPTransformerModel.sample``; returns the record ``{id, target, applied}``. Idempotent."""
    from esm.inverse_folding.gvp_transformer import GVPTransformerModel
    if getattr(GVPTransformerModel.sample, "_esmif1_001", False):
        return {"id": ID, "target": "esm.inverse_folding.gvp_transformer.GVPTransformerModel.sample", "applied": True}
    original = GVPTransformerModel.sample

    def sample(self, coords, partial_seq=None, temperature=1.0, confidence=None, device=None):
        return original(self, coords, partial_seq=partial_seq, temperature=temperature, confidence=confidence, device=next(self.parameters()).device)

    sample._esmif1_001 = True
    sample.__wrapped__ = original
    GVPTransformerModel.sample = sample
    return {"id": ID, "target": "esm.inverse_folding.gvp_transformer.GVPTransformerModel.sample", "applied": True}
