"""Route preload: the small-N gate package's TriMul callees are imported at activation, from the outside.

Under ARM T (fast / big) the FPF_OPS TriMul provider is the sealed package ``third_party/fpf_smalln`` (byte-frozen at its published
bytes); it resolves its two TriMul callees — ``FPF_SMALLN_TRIMUL_EXACT_FN`` (below its size gate) and ``FPF_SMALLN_TRIMUL_FAST_FN`` (at /
above) — lazily, at its first TriMul call.  The kit environment (env.sh) names both as ``ptx_trimul_routes:<fn>`` (the shared core's TriMul
provider by tier word, src/ptx_trimul_routes.py), and the activation proof of levers ``trimul_core`` / ``trimul_core_exact``
(``stack.MODULE_PROOF``: that module in ``sys.modules``, its LEVER words applied) is read BEFORE the first forward.  So the kit imports the
modules those two specs name here, right after its sitecustomize applied the levers and before the proof is evaluated — the package's
bytes stay untouched.  A spec whose module cannot be imported is NAMED in the returned record (``errors``); the proof then refuses the lever
BY NAME (never a silent stock TriMul).  An unset spec is named too (``unset``): env.sh exports both on every row that enables the
package, so its own defaults are never consulted under the kit.  Nothing is printed: the route module prints its own APPLIED line at import,
exactly as when the package imported it.
"""
import importlib
import os
from typing import Optional

ENABLE_ENV = "FPF_SMALLN"                                            # env.sh (ARM T): the package is the FPF_OPS TriMul provider and its gate is on
SPEC_ENVS = ("FPF_SMALLN_TRIMUL_EXACT_FN", "FPF_SMALLN_TRIMUL_FAST_FN")   # module:attr callee specs the package reads at import (env.sh: ptx_trimul_routes:trimul_exact_c256 / :trimul_c256)
ROUTE_MODULE = "ptx_trimul_routes"                                   # what env.sh names for both (stack.MODULE_PROOF's module for trimul_core / trimul_core_exact)


def specs(environ=None) -> dict:
    """{spec env name: 'module:attr' | ''} as the environment names them."""
    environ = os.environ if environ is None else environ
    return {k: (environ.get(k) or "").strip() for k in SPEC_ENVS}


def enabled(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    return (environ.get(ENABLE_ENV, "0") or "0").strip() not in ("", "0")


def preload(environ=None) -> dict:
    """Import the modules the two callee specs name when the gate package is enabled.  Returns
    ``{"enabled": bool, "modules": {spec env: module}, "errors": {spec env: why}}`` (plain data for the activation detail)."""
    environ = os.environ if environ is None else environ
    out = {"enabled": enabled(environ), "modules": {}, "errors": {}}
    if not out["enabled"]:
        return out
    for key, spec in specs(environ).items():
        if not spec:
            out["errors"][key] = "unset(the package default would be consulted)"
            continue
        mod = spec.split(":", 1)[0].strip()
        try:
            importlib.import_module(mod)
            out["modules"][key] = mod
        except Exception as e:  # noqa: BLE001 — named; the activation proof refuses the lever by name
            out["errors"][key] = f"{mod}:{type(e).__name__}:{e}"[:300]
    return out


def resolved_callees() -> Optional[dict]:
    """What the gate package actually resolved, read back from it when it is loaded: ``{"exact_fn", "fast_fn"}`` (its COUNTS words), else None."""
    import sys
    m = sys.modules.get("fpf_smalln")
    if m is None:
        return None
    c = getattr(m, "COUNTS", {}) or {}
    return {"exact_fn": c.get("exact_fn"), "fast_fn": c.get("fast_fn")}
