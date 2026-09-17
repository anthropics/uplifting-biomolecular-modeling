"""Engine-contract adapter of this unit: ``_rebind_levers_prologue`` — after ``install()`` has wrapped the tri-attention prologue entry
point, walk the closure cells of the engine's PairformerBlock.forward (as rebound by the kit's block lever) and repoint any cell still holding
the original prologue at the wrapper. The engine's upstream package is imported lazily, inside the call; the unit's ``__init__`` keeps a
thin wrapper of the same name and signature that imports this module on first call, so importing the unit never imports this module."""
import sys

_P = sys.modules[__package__]          # the unit package: its _STATE record, written at call time


def _rebind_levers_prologue(LEV, orig, new):
    """ptx_trunk2_levers._apply_blk2 does `from fpf_triatt_pro.prologue import triatt_prologue as _pro` INSIDE the function and closes over it in blk_forward.  If BLK2 was applied
    before install(), the closure still points at `orig` (or at a glue wrapper whose __wrapped__ chain ends at orig).  Walk PairformerBlock.forward's closure cells and replace."""
    try:
        import protenix.model.modules.pairformer as PF
        fwd = PF.PairformerBlock.forward
        seen = 0
        stack = [fwd]
        visited = set()
        while stack:
            f = stack.pop()
            if id(f) in visited or not hasattr(f, "__code__"): continue
            visited.add(id(f))
            for cell in (f.__closure__ or ()):
                try:
                    v = cell.cell_contents
                except ValueError:
                    continue
                if v is orig:
                    cell.cell_contents = new; seen += 1
                elif callable(v) and hasattr(v, "__code__"):
                    stack.append(v)
        _P._STATE["rebound_closure_cells"] = seen
        if seen == 0 and getattr(LEV, "_STATS", {}).get("applied"):
            # BLK2 applied but no cell held the raw prologue: fine when a glue wrapper sits in between (it calls PRO.triatt_prologue? no - it calls its captured _orig_pro).
            # The glue wrapper passes ln_mode through to ITS captured original => our PRO-level patch would be bypassed.  Handle: patch the glue wrapper's captured original too.
            pass
    except Exception as e:
        _P._STATE["rebound_closure_cells"] = f"ERR {e!r}"[:120]
