"""Every mode line's FULL hook chain executes once through, in the line's order, without recursion (CPU; a clean interpreter per line).

The hazard: the tensor-parallel line `("big", "tp")` runs hooks `("cells", "tp_rowpair", "fast_inference")` — the cells hook is the entry and
executes the tp hook through OPENFOLD3_OB0_OPT_PAIR_CHAIN; a tp hook that chain-loaded "the next sitecustomize.py" by scanning sys.path from its
HEAD would find the cells hook (first on the path) and re-execute it, which re-executes the tp hook ... RecursionError at activation, every rank
NOT ACTIVE, rc 3 with nothing folded — invisible to any single-hook test. The tp hook scans only the directories AFTER its own. This test resolves EVERY line of modes.LINES the way the package does (modes.resolve:
hooks, hook_dirs, entry_hook, the chain exports), lays the hook directories at the head of sys.path as activation does, exports the chain variables,
runs the entry hook with runpy (hooks.run) in a fresh interpreter, and counts every sitecustomize.py execution through an audit hook: each hook
directory of the line executes exactly once, nothing named sitecustomize.py executes twice, and the interpreter exits 0."""
import json
import os
import subprocess
import sys
import textwrap

import pytest

from openfold3_ob0_opt import modes
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()
TP_ENV = {"OF3TP_RANK": "0", "OF3TP_WORLD": "2", "OPENFOLD3_OB0_OPT_N_GPU": "2"}      # a rank of the row-sharded line (the tp hook arms; nothing is imported, so nothing installs)

CHILD = textwrap.dedent('''
    import collections, json, os, runpy, sys
    hook_dirs = json.loads(os.environ["_CHAIN_TEST_HOOK_DIRS"])
    entry = os.environ["_CHAIN_TEST_ENTRY"]
    counts = collections.Counter()
    def _audit(ev, args):
        if ev == "exec" and args and getattr(args[0], "co_filename", "").endswith("sitecustomize.py"):
            counts[os.path.realpath(args[0].co_filename)] += 1
    sys.addaudithook(_audit)
    sys.setrecursionlimit(200)                       # a re-entrant chain fails fast and by name
    sys.path[:0] = hook_dirs                         # activation: the line's hook directories first, in the line's order
    runpy.run_path(entry, run_name="openfold3_ob0_opt_hook_chain_test")     # hooks.run: the entry hook as the interpreter would at start-up
    print("COUNTS " + json.dumps(counts))
''')


def _cases():
    out = []
    for (mode, el), ln in modes.LINES.items():
        if not ln.hooks:
            continue
        if el in modes.TP_LINES:
            out.append(pytest.param(mode, el, dict(TP_ENV), None, 2, id=f"{mode}/{el}"))
        elif mode == "big":
            out.append(pytest.param(mode, el, {}, 6000, None, id=f"{mode}/{el}@6000"))          # the offload port engaged: the longest chain (confhead > of3o > cells > of3t > of3_levers)
            out.append(pytest.param(mode, el, {}, 300, None, id=f"{mode}/{el}@300"))            # below the port's item gate: the port's hooks step aside (of3o_aside)
        else:
            out.append(pytest.param(mode, el, {}, None, None, id=f"{mode}/{el or '-'}"))
    return out


def test_the_tp_line_chains_cells_then_tp_rowpair_then_fast_inference():
    assert tuple(modes.LINES[("big", "tp")].hooks) == ("cells", "tp_rowpair", "fast_inference")


def test_off_has_no_hooks():
    assert tuple(modes.LINES[("off", None)].hooks) == ()


@pytest.mark.parametrize("mode,el,environ,n_tokens,n_gpu", _cases())
def test_every_lines_hook_chain_executes_once_through_without_recursion(mode, el, environ, n_tokens, n_gpu):
    res = modes.resolve(mode, HOME, environ=environ, n_tokens=n_tokens, n_gpu=n_gpu)
    assert res.hooks, (mode, el)
    hook_dirs = [os.path.realpath(d) for d in res.hook_dirs]
    assert len(hook_dirs) == len(res.hooks) and res.entry_hook and os.path.isfile(res.entry_hook)
    for d in hook_dirs:
        assert os.path.isfile(os.path.join(d, "sitecustomize.py")), d
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENFOLD3_OB0_OPT", "OF3T_", "OF3O_", "OF3TP_", "OF3_", "MODEL_OPT_LEVERS_OFF"))}
    env.update({k: str(v) for k, v in res.exports.items()})                 # the line's switches and its chain variables (CHAIN_ENV: the directory each chaining hook executes next)
    env.update(environ)
    env.update({"_CHAIN_TEST_HOOK_DIRS": json.dumps(res.hook_dirs), "_CHAIN_TEST_ENTRY": res.entry_hook,
                "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)})   # the package and opt_core importable in the child as they are here
    r = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"{mode}/{el}: hooks {res.hooks} rc={r.returncode}\n{r.stderr[-3000:]}"
    assert "RecursionError" not in r.stderr, r.stderr[-3000:]
    line = [x for x in r.stdout.splitlines() if x.startswith("COUNTS ")]
    assert line, r.stdout[-2000:]
    counts = json.loads(line[-1][len("COUNTS "):])
    twice = {f: n for f, n in counts.items() if n > 1}
    assert not twice, f"{mode}/{el}: a hook executed more than once (re-entrant chain): {twice}"
    for d in hook_dirs:                                                      # every hook of the line ran — the chain reached its end
        assert counts.get(os.path.join(d, "sitecustomize.py")) == 1, f"{mode}/{el}: hook {d} did not execute exactly once: {counts}"
