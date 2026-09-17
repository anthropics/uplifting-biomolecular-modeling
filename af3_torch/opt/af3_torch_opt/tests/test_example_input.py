"""The kit ships one ready fold input, inputs/1BRS.json (README §Run): a valid AlphaFold 3 JSON the wrapper's own readers accept —
two protein chains, single-sequence MSAs, no templates declared, one seed."""
import json
import os

from af3_torch_opt import cli, stack


def test_the_shipped_example_is_a_fold_input_the_wrapper_reads():
    p = os.path.join(stack.home(), "inputs", "1BRS.json")
    j = json.load(open(p, encoding="utf-8"))
    assert (j["dialect"], j["version"], j["name"], j["modelSeeds"]) == ("alphafold3", 2, "1BRS", [1])
    chains = [(e["protein"]["id"], len(e["protein"]["sequence"])) for e in j["sequences"]]
    assert chains == [("A", 110), ("B", 89)] and all(e["protein"]["templates"] == [] and e["protein"]["pairedMsa"] == "" and e["protein"]["unpairedMsa"] == ">query\n" + e["protein"]["sequence"] + "\n" for e in j["sequences"])
    assert cli.templates_declared(p) == 0 and cli.item_name(p) == "1BRS"
