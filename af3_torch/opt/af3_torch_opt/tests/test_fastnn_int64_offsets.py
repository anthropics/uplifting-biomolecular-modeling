"""XFOLD-003 corrected in the port: the three fastnn Triton kernels (`xfold/fastnn/{gated_linear_unit,layer_norm,attention}.py`) form their
row / head offsets in int64. CPU only — the kernels are read as source. Held: each promoted statement is present; no kernel scales an int32 row
or head index by a stride; against the archive's member every kernel differs in exactly those statements (the arithmetic on data is upstream's,
so outputs inside the old bound are unchanged byte for byte); the pin names the three files as changed and the patch file carries the same diff."""
import ast
import io
import json
import os
import re
import tarfile

import pytest

HOME = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
FASTNN = os.path.join(HOME, "opt", "forward", "af3t", "af3_torch", "xfold", "fastnn")
PINS = json.load(open(os.path.join(HOME, "stock", "PINS.json")))
FILES = {"xfold/fastnn/gated_linear_unit.py": "_glu_kernel", "xfold/fastnn/layer_norm.py": "_layer_norm_fwd_fused", "xfold/fastnn/attention.py": "_attention_core"}
PROMOTED = {                                                                                   # file -> the statements that make the index int64 before a stride scales it
    "xfold/fastnn/gated_linear_unit.py": ["offs_am = ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M).to(tl.int64)", "offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)"],
    "xfold/fastnn/layer_norm.py": ["row = tl.program_id(0).to(tl.int64)"],
    "xfold/fastnn/attention.py": ["off_hz = tl.program_id(1).to(tl.int64)"],
}
UPSTREAM = {                                                                                   # the archive's statements they replace
    "xfold/fastnn/gated_linear_unit.py": ["offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M", "offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)"],
    "xfold/fastnn/layer_norm.py": ["row = tl.program_id(0)"],
    "xfold/fastnn/attention.py": ["off_hz = tl.program_id(1)"],
}


def _kernel_lines(src, name):
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    seg = ast.get_source_segment(src, fn)
    return [re.sub(r"#.*$", "", ln).strip() for ln in seg.splitlines() if re.sub(r"#.*$", "", ln).strip()]


def test_every_row_or_head_index_is_int64_before_it_meets_a_stride():
    for rel, kernel in FILES.items():
        src = open(os.path.join(HOME, "opt", "forward", "af3t", "af3_torch", rel)).read()
        lines = _kernel_lines(src, kernel)
        for stmt in PROMOTED[rel]:
            assert stmt in lines, (rel, stmt)
        for stmt in UPSTREAM[rel]:
            assert stmt not in lines, (rel, stmt)
    glu = "\n".join(_kernel_lines(open(os.path.join(FASTNN, "gated_linear_unit.py")).read(), "_glu_kernel"))
    assert "x_ptrs = x_ptr + (offs_am[:, None] * stride_xm +" in glu and "offs_cm[:, None] + stride_on * offs_cn[None, :]" in glu      # the promoted vectors are the ones the strides scale
    ln = "\n".join(_kernel_lines(open(os.path.join(FASTNN, "layer_norm.py")).read(), "_layer_norm_fwd_fused"))
    assert ln.index("row = tl.program_id(0).to(tl.int64)") < ln.index("Y += row * N") < ln.index("X += row * N")
    at = "\n".join(_kernel_lines(open(os.path.join(FASTNN, "attention.py")).read(), "_attention_core"))
    for use in ("off_q = off_hz * stride_qh +", "off_hz_bias = (off_hz % H)", "off_o = off_hz * stride_oh +"):
        assert at.index("off_hz = tl.program_id(1).to(tl.int64)") < at.index(use), use
    assert not re.search(r"tl\.program_id\(1\)(?!\.to\(tl\.int64\))", at)                   # no second, unpromoted read of the head index


def test_the_arithmetic_on_data_is_upstreams_statement_for_statement():
    arch = os.path.join(HOME, "stock", PINS["upstream"]["archive"]["file"])
    if not os.path.isfile(arch):
        pytest.skip(f"{arch} is not in this tree (run.sh install fetches it)")
    with tarfile.open(arch, "r:gz") as t:
        up = {m.name.split("/", 1)[1]: t.extractfile(m).read().decode() for m in t.getmembers() if m.isfile() and "/" in m.name and m.name.split("/", 1)[1] in FILES}
    assert set(up) == set(FILES)
    for rel, kernel in FILES.items():
        ours = _kernel_lines(open(os.path.join(HOME, "opt", "forward", "af3t", "af3_torch", rel)).read(), kernel)
        theirs = _kernel_lines(up[rel], kernel)
        swap = dict(zip(PROMOTED[rel], UPSTREAM[rel]))
        assert [swap.get(l, l) for l in ours] == theirs, rel                                # same statements in the same order but for the promotions
    port = PINS["upstream"]["port"]["changed_files"]
    assert all(rel in port for rel in FILES) and port == sorted(port)
    patch = open(os.path.join(HOME, "upstream_issues", "XFOLD-003_fastnn_int64_offsets.diff")).read()
    for rel in FILES:
        assert f"--- a/{rel}" in patch and all(("+    " + s.split(" = ")[0]) in patch for s in PROMOTED[rel]), rel
    assert not os.path.exists(os.path.join(HOME, "opt", "af3_torch_opt", "tests", "test_fastnn_token_limit.py"))
    fwd = open(os.path.join(HOME, "opt", "af3_torch_opt", "forward.py")).read() + open(os.path.join(HOME, "opt", "af3_torch_opt", "forward_impl.py")).read()
    assert "FastnnTokenLimit" not in fwd and "fastnn_token_refusal" not in fwd                # no token gate: memory decides the reach of off / exact
