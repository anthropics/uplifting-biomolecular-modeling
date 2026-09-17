"""No lever binds a core kernel under an add-on's own import name (stack.CORE_KERNEL_ROUTES is empty: the triangle attention is the
core's provider's through the pair cells, imported under the core's own names — CORE_CELL_LEVERS); the route machinery stays for the
core-cell gates."""
import os

from openfold3_ob0_opt import modes, stack

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_no_line_routes_a_core_kernel_under_an_addon_name():
    assert stack.CORE_KERNEL_ROUTES == {}
    for key, line in modes.LINES.items():
        assert stack.core_routes(line.levers) == {}, key


def test_the_pair_cells_gate_the_cores_triangle_attention_rows():
    cells = stack.core_cell_levers(modes.LINES[("fast", None)].levers)
    assert "flash_triattn" in cells["triatt_block"] and "fpf_triatt_k2b" in cells["triatt_provider"]
    assert not os.path.exists(os.path.join(HOME, modes.KITS["trunk_kernels"], "of3t_hook", "of3t_flash_triattn.py"))


def test_route_gate_resolves_to_the_core_copy():
    from opt_core import kernels
    g = stack.core_route_gate("flash_triattn")
    assert g.ok, g.reason
    core_copy = kernels.carried_path("flash_triattn")
    assert g.details["resolved"] == core_copy and g.details["routed"] is True
    kernels.unroute("flash_triattn")
