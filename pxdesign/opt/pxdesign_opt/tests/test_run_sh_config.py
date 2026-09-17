"""run.sh and configs/h100.env: deployment parameters only, the verbs, the refusals (static + bash -n)."""
import os
import re
import subprocess

from .conftest import run_routes_text

def test_config_is_deployment_only(tree):
    src = open(os.path.join(tree, "configs", "h100.env"), encoding="utf-8").read()
    names = set(re.findall(r"\bexport (\w+)=", src))                                    # every variable the file exports, unconditionally or under MODEL_OPT_JIT_ROOT
    assert names >= {"MODEL_OPT", "MODEL_OPT_TARGET_GPU", "PXDESIGN_CKPT_DIR", "PROTENIX_DATA_ROOT_DIR", "TORCH_EXTENSIONS_DIR", "LAYERNORM_TYPE", "MODEL_OPT_STACK_KEY"}
    assert not any(n.startswith("PXD_") or n in ("PXDESIGN_OPT", "PXDESIGN_HOIST") for n in names)
    lines = src.splitlines()
    probes = [i for i, l in enumerate(lines) if 'python -c "import pxdesign_opt' in l]
    assert [lines[i].split("||")[0].strip() for i in probes] == ['PXDESIGN_OPT= python -c "import pxdesign_opt" 2>/dev/null', 'python -c "import pxdesign_opt as k; k.core_gate()" >/dev/null'], probes   # the package importable (the variable neutralised on the silenced probe: the installed .pth stays inert there), then the core pin gate
    assert all("return 3 2>/dev/null || exit 3" in lines[i] for i in probes)                # sourced-file form of the refusal
    assert probes[1] < min(i for i, l in enumerate(lines) if "jit_cache_key" in l)        # the gate precedes every other python probe
    assert "jit_cache_key" in src and "MODEL_OPT_STACK_KEY" in src               # the cache key is the package's derivation (modes.jit_cache_key)
    for var in ("PXDESIGN_CKPT_DIR", "PROTENIX_DATA_ROOT_DIR"):                           # the two paths a deployment must name: no default, refused by name when unset (rc 3, the sourced-file form)
        line = next(l for l in lines if l.startswith(f"export {var}=${{{var}:-}}; case"))
        assert f"NOT ACTIVE: {var} is not set" in line and "return 3 2>/dev/null || exit 3" in line, line
    for var, cache in (("TORCH_EXTENSIONS_DIR", "torch_extensions"),):   # the JIT build dir: keyed under the optional shared root MODEL_OPT_JIT_ROOT when it is set, else not exported (each tool's own default)
        line = next(l for l in lines if f"export {var}=" in l)
        assert line.startswith('if [ -n "${MODEL_OPT_JIT_ROOT:-}" ]; then export ' + f"{var}=${{{var}:-$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/{cache}}}; fi"), line
    assert "MODEL_OPT_STATE" not in src


def test_run_sh_shape(tree):
    src = open(os.path.join(tree, "run.sh"), encoding="utf-8").read()
    assert 'case "$VERB" in design|check|warm|install) ;; *) usage ;; esac' in src and "serve" not in src and "exec python -m pxdesign_opt" in src
    assert 'python -I "$HERE/stock/check_pins.py"' in src and "check_pins.py" not in run_routes_text(src)   # the stock pins are checked by `install` and reported by `check` (python -m pxdesign_opt check), never gated on the run routes
    lines = src.splitlines()
    probes = [i for i, l in enumerate(lines) if 'python -c "import pxdesign_opt' in l]
    assert [lines[i].split("||")[0].strip() for i in probes] == ['PXDESIGN_OPT= python -c "import pxdesign_opt" 2>/dev/null', 'python -c "import pxdesign_opt as k; k.core_gate()" >/dev/null'], probes
    assert probes[1] < min(i for i, l in enumerate(lines) if l.startswith(("source ", "python \"$HERE", "exec python"))), probes   # the gate is the first python of every route
    assert subprocess.run(["bash", "-n", os.path.join(tree, "run.sh")]).returncode == 0
    r = subprocess.run(["bash", os.path.join(tree, "run.sh"), "bogus"], capture_output=True, text=True)
    assert r.returncode == 2
    r = subprocess.run(["bash", os.path.join(tree, "run.sh"), "serve", "--mode", "exact"], capture_output=True, text=True)
    assert r.returncode == 2 and "usage:" in r.stderr                             # not a verb
    r = subprocess.run(["bash", os.path.join(tree, "run.sh"), "check", "--mode", "turbo"], capture_output=True, text=True)
    assert r.returncode == 2 and "unknown --mode turbo" in r.stderr
    r = subprocess.run(["bash", os.path.join(tree, "run.sh"), "check", "--mode", "off"], capture_output=True, text=True, env=dict(os.environ, PATH="/nonexistent:" + os.environ["PATH"]))
    assert "no stock arm" not in r.stderr                                     # check/warm keep their stock arm (the config or the package decides next)
    r = subprocess.run(["bash", os.path.join(tree, "run.sh"), "check", "--mode", "exact"], capture_output=True, text=True, env=dict(os.environ, PXDESIGN_OPT="off"))
    assert r.returncode == 2 and "disagrees" in r.stderr



def test_the_card_keeps_a_caller_set_layernorm_value(tree):
    """configs/h100.env exports LAYERNORM_TYPE=fast_layernorm unless the caller set the variable: a set value — even empty — is kept, so
    `--mode off` hands stock exactly the caller's LayerNorm choice (`${LAYERNORM_TYPE-fast_layernorm}`, not `:-`)."""
    card = os.path.join(tree, "configs", "h100.env")
    line = next(ln for ln in open(card, encoding="utf-8") if ln.startswith("export LAYERNORM_TYPE="))
    word = line.split()[1]                                                     # LAYERNORM_TYPE=${LAYERNORM_TYPE-fast_layernorm}
    assert word == "LAYERNORM_TYPE=${LAYERNORM_TYPE-fast_layernorm}", word
    for preset, expect in ((None, "fast_layernorm"), ("", ""), ("torch", "torch"), ("fast_layernorm", "fast_layernorm")):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        if preset is not None:
            env["LAYERNORM_TYPE"] = preset
        out = subprocess.run(["bash", "-c", f"set -euo pipefail; export {word}; printf %s \"$LAYERNORM_TYPE\""], env=env, capture_output=True, text=True, check=True).stdout
        assert out == expect, (preset, out)
