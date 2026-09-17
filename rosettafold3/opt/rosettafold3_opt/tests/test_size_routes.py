"""report.size_gated_levers: a lever whose every call took a NAMED size route (the library's own small-size path, a cell's floor, a card row) is
size-gated — reached and routed by name — not silent; anything served, refused or erroring keeps the lever out of the size-gated set."""
from .. import report


def test_provider_row_components_below_the_librarys_threshold_are_size_gated_not_silent():
    desc, desc2 = {"served": 0, "fallback": {}, "errors": 0}, {
        "xatt": {"on": True, "served": 0, "passthrough:S<=100": 96},
        "xmul": {"on": True, "served": 0, "passthrough:N<=100": 208},
    }
    out = report.size_gated_levers(["fpf_xatt", "fpf_xmul"], desc, desc2)
    assert out == {"fpf_xatt": {"passthrough:S<=100": 96}, "fpf_xmul": {"passthrough:N<=100": 208}}


def test_a_refusal_or_a_served_call_is_not_a_size_gate():
    desc = {"served": 0, "fallback": {}, "errors": 0}
    assert report.size_gated_levers(["fpf_xatt"], desc, {"xatt": {"served": 0, "passthrough:S<=100": 4, "refused:rowx:cc": 2}}) == {}
    assert report.size_gated_levers(["fpf_xmul"], desc, {"xmul": {"served": 3, "passthrough:N<=100": 4, "tmk3_exact|bf16|C64|D64": 3}}) == {}
    assert report.size_gated_levers(["fpf_xatt"], desc, {"xatt": {"served": 0}}) == {}          # never reached: silent stays silent


def test_the_exact_pwa_cell_below_its_floor_is_size_gated():
    desc, desc2 = {"served": 0, "fallback": {}, "errors": 0}, {"smsa": {"on": True, "served": 0, "stock": 0, "fallback": {"floor:S<16_or_rows<65536": 8}}}
    assert report.size_gated_levers(["fpf_smsa"], desc, desc2) == {"fpf_smsa": {"floor:S<16_or_rows<65536": 8}}
    desc2b = {"smsa": {"on": True, "served": 0, "stock": 1, "fallback": {"bitcmp_failed:400x512": 7}}}
    assert report.size_gated_levers(["fpf_smsa"], desc, desc2b) == {}                             # a failed bit-compare is a refusal, not a size gate


def test_the_fast_msa_cells_below_their_token_floor_or_left_at_stock_by_a_card_row_are_size_gated():
    desc = {"served": 0, "fallback": {}, "errors": 0}
    desc2 = {"msa": {"on": True, "served": 0, "counts": {"opm": {"served": 0, "fallback": {"cc8.0:opm_cell_row_a2=x0.53-0.55_of_cuBLAS": 8}},
                                                          "pwa": {"served": 0, "fallback": {"I<32": 8}}}}}
    assert report.size_gated_levers(["fpf_msa"], desc, desc2) == {"fpf_msa": {"opm": {"cc8.0:opm_cell_row_a2=x0.53-0.55_of_cuBLAS": 8}, "pwa": {"I<32": 8}}}
    desc2b = {"msa": {"on": True, "served": 0, "counts": {"opm": {"served": 0, "fallback": {}}, "pwa": {"served": 0, "fallback": {"I<32": 8}}}}}
    assert report.size_gated_levers(["fpf_msa"], desc, desc2b) == {}                              # a unit that was never reached keeps the lever silent


def test_provider_row_hooks_never_reached_below_the_librarys_threshold_are_size_gated_by_the_trunk_size():
    """At or below cuEquivariance's own fallback threshold the stock op takes its reference path before the kit's hook: no census at all. With every
    trunk of the process at such a size (the trunk-graph budget's census names them) the lever is size-gated by name; a larger trunk keeps it silent."""
    desc = {"served": 0, "fallback": {}, "errors": 0}
    desc2 = {"xatt": {"on": True, "served": 0}, "xmul": {"on": True, "served": 0}}
    assert report.size_gated_levers(["fpf_xatt", "fpf_xmul"], desc, desc2, trunk_sizes=[20]) == {
        "fpf_xatt": {"unreached:I<=100": 20}, "fpf_xmul": {"unreached:I<=100": 20}}
    assert report.size_gated_levers(["fpf_xatt", "fpf_xmul"], desc, desc2, trunk_sizes=[20, 199]) == {}     # a trunk above the threshold: silent is silent
    assert report.size_gated_levers(["fpf_xatt", "fpf_xmul"], desc, desc2) == {}                            # no trunk record: nothing inferred


def test_trunk_sizes_read_the_budget_rules_per_size_census_and_the_graphs_capture_records():
    """tgbudget.census() nests the per-size words under by_key; the adapter's trunk_graph section lists captures with their I."""
    assert report.trunk_sizes({"graphed": 10, "skipped": 0, "by_key": {"graphed:20": 10}}, {}) == [20]
    assert report.trunk_sizes({"graphed": 0, "skipped": 10, "by_key": {"skipped:20": 10}}, {}) == [20]
    assert report.trunk_sizes({}, {"trunk_graph": {"captures": [{"I": 20, "capture_s": 0.7}, {"I": 199}]}}) == [20, 199]
    assert report.trunk_sizes({"by_key": {"graphed:20": 10, "skipped:400": 2}}, {"trunk_graph": {"captures": [{"I": 20}]}}) == [20, 400]
    assert report.trunk_sizes({}, {}) == [] and report.trunk_sizes(None, {}) == []
