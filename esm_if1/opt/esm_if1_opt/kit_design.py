"""The kit child of ``design --mode fast``: ``python -m esm_if1_opt.kit_design <driver arguments>`` runs the batched driver (``batched.main``,
route word ``kit``) in its own process — the driver's PREPASS / STARTUP / CALL / ITEM / OUTPUTS_WRITTEN / PEAK / KERNELS lines and its exit
status are this process's; the parent prints ACTIVE / LEVER before it and INVOCATION / EXIT after it. No environment is stripped and nothing
is proven here: this is the kit's own tier. A leading ``--upstream-fix <paths>`` pair (present only when the run named ``--upstream-fix <ID>``)
is applied before the driver starts (``upstream_fix.apply_files``).
"""
import sys


def main(argv=None):
    from . import batched, upstream_fix
    fixes, rest = upstream_fix.split_argv(sys.argv[1:] if argv is None else argv)
    if fixes:
        upstream_fix.apply_files(fixes)
    return batched.main(rest, route="kit")


if __name__ == "__main__":
    sys.exit(main())
