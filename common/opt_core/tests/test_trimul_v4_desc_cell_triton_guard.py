"""kernels.trimul row v4: a TENSOR-DESCRIPTOR launch cell of the table (K1 impl tma / tma2, K3 impl tma -- written on tl.make_tensor_descriptor +
triton.set_allocator, triton >= 3.4) is never handed to a stack whose stated triton cannot build it (0.5.212.1: atlasfold on
torch 2.7.1+cu128 / triton 3.3.1 at 200 tokens -- every fast / big TriMul call raised AttributeError('make_tensor_descriptor') inside the v4
launch because the (100, 256] cells read the cc reference column's descriptor cell).  Pure checks (no GPU, no framework): the kit-derived case,
the whole table on every pre-3.4 stack word, inertness on every descriptor-era column and the planning call, the helper units, and the serving
call's line (static)."""
import inspect
import re

import pytest

from opt_core.kernels import trimul as T

ATLASFOLD_STACKS = ("H100:2.7.1+cu128/3.3.1/cueq0.10.0", "A100:2.7.1+cu128/3.3.1/cueq0.10.0")       # models/atlasfold: img_af (torch 2.7.1+cu128, triton 3.3.1, cuequivariance 0.10.0)
PRE_DESC_STACKS = ATLASFOLD_STACKS + ("H100:2.7.1+cu126/3.3.1/cueq0.10.0", "A100:2.7.1+cu126/3.3.1/cueq0.10.0",   # protenix_v1 img_ptx1 / opendde img_odde* / genie3 g3cu126
                                      "H100:2.7.1+cu128/3.3.1/nocueq", "B200:2.7.1+cu128/3.3.1/cueq0.10.0",         # an unlisted library word / an unmeasured part on the same wheel
                                      "H100:2.6.0+cu124/3.2.0/nocueq", "H100:2.5.1+cu124/3.1.0/nocueq")            # older wheels (caliby / chai1's retired image): reference-column readers
TOKEN = "v4_cell=package(triton:"
KEY = re.compile(r"^(\d+\.\d+)\|([a-z0-9_]+)\|C(\d+)\|H(\d+)\|N<=(\d+)\|(out|in)\|(fwd|fwdbwd)$")


def _triton_mm(stack):
    return tuple(int(x) for x in stack.split("/")[1].split(".")[:2])


def _args(prec):
    """cell precision word -> select()'s (dtype, residency, tf32)."""
    if prec == "f32z_bf16":
        return "bf16", "fp32", None
    if prec == "tf32":
        return "fp32", None, True
    return prec, None, None


def _listed_columns():
    cols = set()
    for c in T.table()["cells"].values():
        for k in ("fast_per_stack", "big_per_stack", "exact_per_stack"):
            cols.update((c.get(k) or {}).keys())
        for per in (c.get("ms") or {}).values():
            cols.update(per.keys())
    return sorted(cols)


def _family_sizes():
    """{(cc, prec, C, H, dir, pass): sorted bucket bounds} over the plain cell keys."""
    fam = {}
    for key in T.table()["cells"]:
        m = KEY.match(key)
        if m:
            cc, prec, C, H, N, d, pas = m.groups()
            fam.setdefault((cc, prec, int(C), int(H), d, pas), []).append(int(N))
    return {k: sorted(v) for k, v in fam.items()}


def _probe_sizes(bounds):
    out, lo = set(), 0
    for n in bounds:
        out.update((n, (lo + n) // 2 + 1)); lo = n
    out.add(bounds[-1] + 500)
    return sorted(out)


def _select(cc, prec, C, H, n, d, pas, word, stack):
    dt, res, tf = _args(prec)
    return T.select(cc, dt, C, H, n, "outgoing" if d == "out" else "incoming", word=word, residency=res, tf32=tf, backward=(pas == "fwdbwd"), stack=stack, has_cueq=True)


def test_helper_units():
    assert T.V4_DESC_TRITON_MIN == (3, 4) and set(("tma", "tma2")) <= set(T.V4_DESC_IMPLS)
    assert T.v4_desc_cell({"k1": {"impl": "tma", "BM": 64}, "k3": {"BM": 64}})
    assert T.v4_desc_cell({"k1": {"BM": 64}, "k3": {"impl": "TMA"}})
    assert T.v4_desc_cell({"k1": {"impl": "tma2"}, "k3": {}})
    assert not T.v4_desc_cell({"k1": {"BM": 128, "BN": 128}, "k3": {"BM": 64}})           # pointer cell (no impl word)
    assert not T.v4_desc_cell({"k1": {"impl": "ptr"}, "k3": {"impl": "ptr"}})
    assert not T.v4_desc_cell(None) and not T.v4_desc_cell("tma") and not T.v4_desc_cell({"k1": "tma"})
    cell = T.table()["cells"]["9.0|bf16|C128|H128|N<=256|out|fwd"]
    ref = T.cc_reference_column("9.0")
    recorded = T._cell_config(cell, "v4", ref)
    assert T.v4_desc_cell(recorded), recorded                                                   # the reference column's v4 cell IS a descriptor cell (the SMOKE7 cell)
    assert T.v4_table_cell(cell, ref, "3.7.1") == (recorded, "")                              # a descriptor-era triton reads it unchanged
    assert T.v4_table_cell(cell, ref, "3.4.0") == (recorded, "")
    assert T.v4_table_cell(cell, ref, None) == (recorded, "")                                 # a planning call (no stack stated): unchanged
    assert T.v4_table_cell(cell, ref, "none") == (recorded, "")                               # unparsable: unchanged (the serving call holds the line with the triton in hand)
    cfg, note = T.v4_table_cell(cell, ref, "3.3.1")
    assert cfg is None and note == "v4_cell=package(triton:3.3.1<3.4:tensor_descriptor_cell_of:%s)" % ref
    assert T.v4_table_cell(cell, ref, "3.2.0")[0] is None


@pytest.mark.parametrize("stack", ATLASFOLD_STACKS[:1])
def test_atlasfold_geometry_small_n_on_torch271_reads_no_descriptor_cell(stack):
    """The SMOKE7 case: cc 9.0, bf16, c_z = c_hidden = 128, N in (100, 256], both directions, the tier words the kit's modes pass (fast / big)
    and the developer row word v4 -> row v4 (the column's word, unchanged) with config None (the v4 package's own cell for (9.0, triton 3.3):
    pointer kernels) and the reason's token; above 256 tokens the listed torch-2.7.1 column answers as before (native / v4 pointer cells)."""
    for n in (101, 128, 179, 200, 256):
        for d in ("outgoing", "incoming"):
            for word in ("fast", "big", "v4"):
                s = T.select("9.0", "bf16", 128, 128, n, d, word=word, stack=stack, has_cueq=True)
                assert s.row == "v4", (n, d, word, T.describe(s))
                assert s.config is None and (TOKEN + "3.3.1<3.4:tensor_descriptor_cell_of:") in s.reason, (n, d, word, s.config, s.reason)
    for n in (257, 384, 512, 896, 1280, 2000):
        for d in ("outgoing", "incoming"):
            for word in ("fast", "big"):
                s = T.select("9.0", "bf16", 128, 128, n, d, word=word, stack=stack, has_cueq=True)
                assert s.stack == stack and TOKEN not in s.reason and not T.v4_desc_cell(s.config), (n, d, word, T.describe(s))


@pytest.mark.parametrize("stack", PRE_DESC_STACKS)
def test_no_cell_hands_a_descriptor_cell_to_a_pre_3_4_triton(stack):
    """Over EVERY plain cell family x every bucket bound / midpoint / beyond x {fast, big, exact, v4}: no Selection on a stack word whose triton
    is below 3.4 carries a v4 tensor-descriptor launch cell; wherever the guard fired the row is still v4 and the reason names it."""
    assert _triton_mm(stack) < (3, 4)
    seen = fired = 0
    for (cc, prec, C, H, d, pas), bounds in _family_sizes().items():
        for n in _probe_sizes(bounds):
            for word in ("fast", "big", "exact", "v4"):
                try:
                    s = _select(cc, prec, C, H, n, d, pas, word, stack)
                except T.Refusal:
                    continue
                seen += 1
                assert not (s.row == "v4" and T.v4_desc_cell(s.config)), (stack, cc, prec, C, H, n, d, pas, word, s.config, s.reason)
                if TOKEN in s.reason:
                    fired += 1
                    assert s.row == "v4" and s.config is None
    assert seen > 1000
    if stack.startswith(("H100:2.7.1", "A100:2.7.1+cu12", "B200:2.7.1", "H100:2.6", "H100:2.5")):
        assert fired > 0, stack


def test_descriptor_era_columns_and_the_planning_call_are_byte_inert():
    """Every LISTED column runs triton >= 3.4 or is a torch-2.7.1 column: on the >= 3.4 columns and on the planning call (stack None) the guard never
    fires and the table's descriptor cells are returned exactly where recorded (the N<=256 cc-9.0 bf16 cells on the 2.10 / 2.12 / 2.13 columns ->
    K1 impl tma) -- the same kernel is selected before / after for every kit on those images."""
    cols = _listed_columns()
    desc_cols = [c for c in cols if _triton_mm(c) >= (3, 4)]
    assert desc_cols and all(_triton_mm(c) < (3, 4) for c in cols if c not in desc_cols)
    assert all(c.split(":")[1].startswith("2.7.1+") for c in cols if c not in desc_cols)     # the pre-descriptor listed columns are the torch-2.7.1 images, nothing else
    kept = 0
    for stack in [c for c in desc_cols if c.startswith("H100:")] + [None]:                  # the H100 columns of every descriptor-era wheel + the planning call (the full column set: CHANGES 0.5.212.1 grid)
        for (cc, prec, C, H, d, pas), bounds in _family_sizes().items():
            for n in _probe_sizes(bounds):
                for word in ("fast", "big", "exact", "v4"):
                    try:
                        s = _select(cc, prec, C, H, n, d, pas, word, stack)
                    except T.Refusal:
                        continue
                    assert TOKEN not in s.reason, (stack, cc, prec, C, H, n, d, word, s.reason)
                    kept += bool(s.row == "v4" and T.v4_desc_cell(s.config))
    assert kept > 100                                                                          # descriptor cells still served on their own columns
    for col in ("H100:2.13.0+cu130/3.7.1/cueq0.11.1", "H100:2.12.0+cu130/3.7.0/cueq0.10.0", "H100:2.10.0+cu128/3.6.0/cueq0.10.0"):
        for word in ("fast", "big"):
            s = T.select("9.0", "bf16", 128, 128, 200, "outgoing", word=word, stack=col, has_cueq=True)
            assert s.row == "v4" and s.stack == col and T.v4_desc_cell(s.config) and s.config["k1"]["impl"] == "tma", (col, word, T.describe(s))
    s = T.select("9.0", "bf16", 128, 128, 200, "outgoing", word="fast", has_cueq=True)        # planning call: the reference column's recorded cell, as before
    assert s.row == "v4" and T.v4_desc_cell(s.config)


def test_the_serving_call_holds_the_same_line_by_name():
    """_serve_v4 (static: no framework here): a caller-built descriptor cell on a triton without the API keeps the package's own cell (said once),
    and an AttributeError naming triton inside the launch is a Refusal BY NAME -- never a kernel error through a word."""
    src = inspect.getsource(T._serve_v4)
    assert 'v4_desc_cell(config) and not getattr(K, "HAS_DESC", True)' in src
    assert "serving the v4 package's own cell" in src and "_say_once(" in src
    assert "except AttributeError as e:" in src and 'raise Refusal("triton:%s"' in src
    face = inspect.getsource(T._serve_v4_kernel_face)                                          # the (64, 128) face held this line already
    assert "no_tensor_descriptor" in face and "except AttributeError" in face
