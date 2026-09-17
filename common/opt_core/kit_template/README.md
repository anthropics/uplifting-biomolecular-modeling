# kit_template — the two files the kit carries as byte-identical copies

| file | copied to | what it is |
|---|---|---|
| `_build_backend.py` | `<engine>/opt/_build_backend.py` (beside the kit's `pyproject.toml`) | the PEP 517 build backend that places the kit's generated autoload `.pth` at the wheel root; `python _build_backend.py <package> <ENV> <tag>` prints the `.pth` text |
| `_core_gate.py` | `<engine>/opt/<package>/_core_gate.py` (inside the kit's package) | the pre-import core pin gate: `gate(__file__)` compares the kit's `[tool.opt_core]` pin with the `opt_core` the interpreter would import (located, not imported; its `MANIFEST.json` read from disk) and refuses by name with exit 3 on absent / mismatched / pre-manifest core or unreadable pin |

Entry pattern (statement one of every documented entry — `python -m <package>`, console script, `run.sh` target, `<package>._autoload`,
in-process `enable()` — before any `opt_core` import):

```python
from ._core_gate import gate
gate(__file__)          # [<tag>] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …  -> exit 3
import opt_core         # only now
```

`gate()` at entry is THE pin/version gate (it works on an absent or stale core); a kit makes no separate
`opt_core.gates.core_pin_check` call in its activation — one producer of the 'pinned X, installed Y' fact (`core_pin_check` stays the
core's live-tree comparison for tests and tools).

Both files are standard-library only, import nothing of the core and touch no `sys.path`; `common/opt_core/tests/test_instances_backend_hygiene.py`
holds every adopting kit's copies to these bytes, and `tests/test_core_gate.py` holds the gate's behaviour.
