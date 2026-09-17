"""chai1_eager.msa_kernels — MSA-module statements for the structured eager trunk (plugged via ``chai1_eager.trunk.CFG['msa_impl']``).

``msa_pad`` (exact-class).  Upstream pads every MSA context to 16384 rows (``msa_mask`` False on every position of a padding row) and the export's
MSA module runs its row-wise work on all of them: the outer-product mean in 4096-row slices (``OuterProductMean.CH``), the pair-weighted averaging
in 8192-row slices (``MSAPairWeightedAveraging.CH``), the MSA transition and the ``linear_s2m`` broadcast on every row.  A padding row never reaches
the module's only output (the pair representation): the outer-product mean masks its operand rows to exact zeros (a slice whose every row is
masked contributes a tensor of signed zeros to the running sum ``acc``, and ``acc = op_1 + 0`` then ``acc + op_k`` never holds a negative zero, so
``acc + (±0) == acc`` bit for bit), the pair-weighted averaging masks ``v`` (its output on a padding row is never read by a kept row: every
MSA-track statement is row-local), and the MSA track itself is dropped at the module's end.  ``msa_pad`` runs the SAME statement on the leading
rows only, cut on the export's own slice grid: the module on the first ``ceil(S_eff / 8192) * 8192`` rows and its outer-product means on the first
``ceil(S_eff / 4096) * 4096`` rows (``S_eff`` = the last row that carries any mask position, + 1; at least one slice each) — every GEMM, LayerNorm,
softmax and SDPA call of a kept slice is then the very call the whole statement makes on that slice (same operands, shapes and kernels), the
transition's ``ceil(numel * 4 / 2**30)`` token chunking lands on the same per-GEMM row count at chai1's crops 384 / 512 / 1024 / 1536 / 2048, and the
skipped slices are the all-masked ones.  Nothing here is bitwise by assumption: the first call of every shape class ``(S, S_mod, S_opm, N, dtype)``
runs the whole statement AND the cut one and compares the returned pair tensors with ``torch.equal`` (the reference held on the host meanwhile, so
the device peak is the whole statement's own); a class that differs is served by the whole statement from then on and booked
``fallback:not_bitwise`` — an undeclared word: the kit's exit gate refuses the run by name (chai1_opt.pairtrack).  A call with nothing to cut
(``S`` within one slice of ``S_eff``: a deep MSA, or the rows already sliced by big.py's ``msa_rows``) books ``fallback:no_tail`` (declared) and
the statement runs whole.  B > 1 (never on the kit's routes) takes the union of the batch's rows."""
import sys
import torch

__all__ = ["MsaPad", "new_ledger", "NAME", "STRATEGY"]
__version__ = "1"
NAME = "msa_pad"
STRATEGY = "LOCAL.chai1.msa_pad"
TAG = "[chai1-opt]"


def _round_up(n: int, block: int) -> int:
    return ((int(n) + block - 1) // block) * block


def new_ledger(expected=("no_tail",)):
    """This lever's ``opt_core.counters.Ledger``.  Its gate holds on a run whose every call had nothing to cut (a deep MSA, or ``msa_rows`` ahead of
    it: ``state=skipped reason=all_fallback:no_tail`` on the LEVER line — a declared step-aside); ``not_bitwise`` or an error refuses it."""
    from opt_core.counters import Ledger

    class MsaPadLedger(Ledger):
        def gate(self, name=None, *, require_served=False):
            return Ledger.gate(self, name, require_served=require_served)

    return MsaPadLedger(STRATEGY, impl=f"chai1_eager.msa_kernels@{__version__}", origin="kit", expected=tuple(expected))


class MsaPad:
    """``CFG['msa_impl']`` callable.  ``TR`` = the installed ``chai1_eager.trunk`` module (its slice constants are read, never restated)."""

    def __init__(self, TR, ledger, selftest: bool = True):
        self.TR, self.ledger, self.selftest = TR, ledger, bool(selftest)
        self.block_opm = int(TR.OuterProductMean.CH)
        self.block_mod = int(TR.MSAPairWeightedAveraging.CH)
        assert self.block_mod % self.block_opm == 0, (self.block_mod, self.block_opm)
        self.classes = {}                      # (S, s_mod, s_opm, N, dtype) -> "pass" | "differs"
        self._mask_ref = None                  # (mask tensor, s_eff): one host read per distinct mask object (one per fold: the recycles share it)
        self.calls = {"served": 0, "no_tail": 0, "not_bitwise": 0, "selftests": 0}

    chai1_opt_lever = NAME

    # -------------------------------------------------------------------------------------------------------------------- facts
    def describe(self) -> dict:
        return {"impl": f"chai1_eager.msa_kernels@{__version__}", "block_opm": self.block_opm, "block_mod": self.block_mod, "selftest": self.selftest,
                "classes": {f"S{k[0]}->{k[1]}/{k[2]}@N{k[3]}": v for k, v in self.classes.items()}}

    def _s_eff(self, mask) -> int:
        ref = self._mask_ref
        if ref is not None and ref[0] is mask:
            return ref[1]
        rows = mask.any(-1)
        if rows.dim() > 1:
            rows = rows.any(0)                 # [S]: the rows with any True over the batch
        idx = rows.nonzero()
        s_eff = int(idx.max().item()) + 1 if idx.numel() else 0
        self._mask_ref = (mask, s_eff)
        return s_eff

    # -------------------------------------------------------------------------------------------------------------------- the plug
    def __call__(self, mod, s, z, msa_input_feats, msa_mask, pair_mask):
        if msa_input_feats is None or msa_mask is None or msa_input_feats.dim() != 4 or msa_mask.dim() != 3 or tuple(msa_input_feats.shape[:3]) != tuple(msa_mask.shape):
            self.ledger.fallback("no_tail"); self.calls["no_tail"] += 1
            return NotImplemented
        S, N = int(msa_input_feats.shape[1]), int(pair_mask.shape[1])
        s_eff = self._s_eff(msa_mask)
        s_mod = min(S, max(self.block_mod, _round_up(s_eff, self.block_mod)))
        s_opm = min(S, max(self.block_opm, _round_up(s_eff, self.block_opm)))
        if s_mod >= S and s_opm >= S:
            self.ledger.fallback("no_tail"); self.calls["no_tail"] += 1
            return NotImplemented
        key = (S, s_mod, s_opm, N, str(msa_input_feats.dtype))
        state = self.classes.get(key)
        if state == "differs":
            self.ledger.fallback("not_bitwise"); self.calls["not_bitwise"] += 1
            return NotImplemented
        cut = lambda: mod.forward_rows(s, z, msa_input_feats[:, :s_mod], msa_mask[:, :s_mod], pair_mask, s_opm=(s_opm if s_opm < s_mod else None))
        shape = f"S{S}->{s_mod}/{s_opm}@N{N}"
        if state is None and self.selftest:
            self.calls["selftests"] += 1
            z_ref = mod.forward_rows(s, z, msa_input_feats, msa_mask, pair_mask)          # the whole statement, once per class
            ref_host = z_ref.to("cpu"); dev = z_ref.device
            del z_ref
            z_new = cut()
            ok = bool(torch.equal(z_new.to("cpu"), ref_host))
            self.classes[key] = "pass" if ok else "differs"
            sys.stderr.write(f"{TAG} MSA_PAD class S={S} s_eff={s_eff} rows_module={s_mod} rows_opm={s_opm} N={N} selftest={'pass' if ok else 'DIFFERS'} "
                             f"(the whole statement vs the cut one on this call's operands, torch.equal on the returned pair tensor)\n"); sys.stderr.flush()
            if not ok:
                self.ledger.fallback("not_bitwise"); self.calls["not_bitwise"] += 1
                del z_new
                return ref_host.to(dev)
            del ref_host
            self.ledger.serve(shape); self.calls["served"] += 1
            return z_new
        z_new = cut()
        if state is None:
            self.classes[key] = "unchecked"
        self.ledger.serve(shape); self.calls["served"] += 1
        return z_new
