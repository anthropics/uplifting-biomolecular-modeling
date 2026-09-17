"""The carried kits' own spellings, read from their source literals — the lock tests' readers (the mode table, the registry and the
in-process route are held to what the kit files say, never to a transcription): the driver's default ``--levels`` and accepted lever set, the
eager stack's ``LEVERS``, the DSTEP add-on's lever names, the kit's deterministic recipe statements, the W names of the driver's lever block."""
from __future__ import annotations

import ast
import re
from typing import Optional, Tuple

from chai1_opt import stack


def driver_default_levels(driver_path: str) -> str:
    """The driver's own default ``--levels`` value, read from its argparse literal (the kit's spelling of its default line)."""
    src = open(driver_path, "r", encoding="utf-8").read()
    m = re.search(r'add_argument\("--levels",\s*default="([A-Z0-9,]+)"', src)
    if not m:
        raise ValueError(f"no --levels default found in {driver_path}")
    return m.group(1)


def driver_accepted_levers(driver_path: str) -> Tuple[str, ...]:
    """The lever names the driver accepts on ``--levels`` (its ``if levels - {...}: ap.error(...)`` refusal), read from the source."""
    src = open(driver_path, "r", encoding="utf-8").read()
    m = re.search(r"if levels - \{([^}]*)\}: ap\.error\(", src)
    if not m:
        raise ValueError(f"no `if levels - {{...}}: ap.error(` refusal found in {driver_path}")
    return tuple(sorted(ast.literal_eval("[" + m.group(1) + "]")))


def driver_lever_names(driver_path: str) -> Tuple[str, ...]:
    """Every ``"Wn" in levels`` / ``"Wn" not in levels`` name the driver tests for."""
    src = open(driver_path, "r", encoding="utf-8").read()
    return tuple(sorted(set(re.findall(r'"(W\d)" (?:not )?in levels', src))))


def dstep_all_levers(stackx_path: str) -> Tuple[str, ...]:
    """The DSTEP add-on's own lever names, read from its source literals (``chai1_fastln/stackx.py``: ``HOIST_LEVERS``, ``COMPILE_LEVERS`` then ``ATTN_LEVERS`` keys —
    the add-on's ``ALL_LEVERS = tuple(HOIST_LEVERS) + tuple(COMPILE_LEVERS) + tuple(ATTN_LEVERS)``)."""
    src = open(stackx_path, "r", encoding="utf-8").read()
    out = []
    for lit in ("HOIST_LEVERS", "COMPILE_LEVERS", "ATTN_LEVERS"):
        m = re.search(rf"^{lit} = (\{{[^}}]*\}})", src, re.M)
        if not m:
            raise ValueError(f"no `{lit} = {{...}}` literal found in {stackx_path}")
        out += list(ast.literal_eval(m.group(1)).keys())
    return tuple(out)


def eager_stack_levers(stack_path: str) -> Tuple[str, ...]:
    """The eager stack's own ``LEVERS`` tuple, read from its source literal (``chai1_eager/stack.py``)."""
    src = open(stack_path, "r", encoding="utf-8").read()
    m = re.search(r"^LEVERS = \(([^)]*)\)", src, re.M)
    if not m:
        raise ValueError(f"no `LEVERS = (...)` literal found in {stack_path}")
    return tuple(ast.literal_eval("(" + m.group(1) + ")"))


def kit_recipe(proto_path: str) -> dict:
    """The recipe as the kit writes it (the det.py pair lock): the statements of ``apply_deterministic_mode`` in source order."""
    src = open(proto_path, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_deterministic_mode")
    body = [ast.unparse(n) for n in fn.body]                    # ast.unparse spells strings with single quotes
    joined = "\n".join(body)
    m_env = re.search(r"os\.environ\.get\('([A-Z_]+)', '0'\)", joined)
    m_jit = re.search(r"os\.environ\.get\('([A-Z_]+)', '1'\) != '0'", joined)
    m_cub = re.search(r"os\.environ\.setdefault\('([A-Z_]+)', '([^']+)'\)", joined)
    return {"env_switch": m_env.group(1) if m_env else None,
            "jit_profiling": "torch._C._jit_set_profiling_mode(False)" in joined,
            "jit_optout": m_jit.group(1) if m_jit else None,
            "cublas": m_cub.groups() if m_cub else None,
            "cudnn_deterministic": "torch.backends.cudnn.deterministic = True" in joined,
            "cudnn_benchmark_false": "torch.backends.cudnn.benchmark = False" in joined,
            "deterministic_algorithms": "torch.use_deterministic_algorithms(True, warn_only=mode == 'warn')" in joined,
            "statements": body}


def kit_lever_names_in_block(path: Optional[str] = None) -> Tuple[str, ...]:
    """The W names the driver's lever block tests for (``"Wn" in levels``): what the in-process route can install."""
    nodes, _ = stack.kit_lever_statements(path)
    names = set()
    for n in nodes:
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Constant):
            names.add(n.test.left.value)
    return tuple(sorted(names))
