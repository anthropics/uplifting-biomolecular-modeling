"""outputs.py: items with the stock defaults (the stock flags' dests), the argv order, the one writer's file."""
import os

import pytest

from progen2_opt import outputs


def test_load_items_defaults_and_argv(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text('{"item_id": "a"}\n{"item_id": "b", "max_length": 512, "rng_seed": 7, "context": "1MK", "prefix_id": "x"}\n')
    items = outputs.load_items(str(p), "sample")
    assert items[0] == {"item_id": "a", "context": "1", "max_length": 256, "num_samples": 1, "t": 0.2, "p": 0.95, "rng_seed": 42}
    assert items[1]["max_length"] == 512 and items[1]["rng_seed"] == 7 and items[1]["context"] == "1MK" and "prefix_id" not in items[1]   # a caller's bookkeeping key is ignored
    assert outputs.sample_argv(items[0]) == ["--context", "1", "--max-length", "256", "--num-samples", "1", "--t", "0.2", "--p", "0.95", "--rng-seed", "42"]
    assert not hasattr(outputs, "unit_of_item") and not hasattr(outputs, "block_of_client_stdout")    # no request grammar, no client: the in-process unit takes the item's own typed fields
    assert list(outputs.SAMPLE_ARG_OF) == list(outputs.SAMPLE_DEFAULTS) == ["context", "max_length", "num_samples", "t", "p", "rng_seed"]
    lst = tmp_path / "items.json"
    lst.write_text('[{"context": "1", "max_length": 32}]')                                            # a JSON list; item_id numbered
    assert outputs.load_items(str(lst), "sample")[0]["item_id"] == "item0000"


def test_retired_seed_key_is_refused_by_name(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text('{"item_id": "old", "seed": 7}\n')
    with pytest.raises(ValueError, match="'seed' is retired .* use rng_seed"):
        outputs.load_items(str(p), "sample")


def test_score_items_take_the_stock_default_context(tmp_path, tree):
    p = tmp_path / "items.jsonl"
    p.write_text('{"item_id": "d"}\n')
    items = outputs.load_items(str(p), "score", os.path.join(tree, "stock", "src", "progen2"))
    assert items[0]["context"].startswith("1") and len(items[0]["context"]) == 422


def test_writer_writes_the_block_only(tmp_path):
    w = outputs.Writer(str(tmp_path / "out"), "score")
    path = w.write_item({"item_id": "x", "context": "1A"}, "ll_sum=-1.5\nll_mean=-0.5\n")
    assert path.endswith(os.path.join("items", "x", "block.txt")) and open(path).read() == "ll_sum=-1.5\nll_mean=-0.5\n" and w.n_written == 1
    assert sorted(os.listdir(tmp_path / "out")) == ["items"] and os.listdir(tmp_path / "out" / "items" / "x") == ["block.txt"]   # nothing else: no rows.jsonl, no stdout/stderr copies, no manifest
    for gone in ("read_rows", "ROWS_NAME"):
        assert not hasattr(outputs, gone), gone


def test_stock_sample_block_equals_the_serving_kits_grammar():
    """The block cut from a stock sample.py stdout == the serving kit's block for the same completions, byte for byte."""
    stdout = "loading parameters took 1.00s\nsampling\n1\n\n0\n1MSEQ2\n\n1\n1MKV2\nsampling took 0.50s\ndone.\n"
    block = outputs.block_of_stdout("sample", stdout)
    context, truncations = "1", ["1MSEQ2", "1MKV2"]
    kit_block = f"{context}\n" + "".join(f"\n{i}\n{tr}\n" for i, tr in enumerate(truncations)) + "done.\n"    # the kit's line, verbatim
    assert block == kit_block
    ll = "loading parameters\nloading parameters took 1.00s\nloading tokenizer\nloading tokenizer took 0.01s\nlog-likelihood (left-to-right, right-to-left)\nll_sum=-1.5\nll_mean=-0.5\nlog-likelihood (left-to-right, right-to-left) took 0.44s\ndone.\n"
    assert outputs.block_of_stdout("score", ll) == "ll_sum=-1.5\nll_mean=-0.5\n"                     # likelihood.py's stdout: its result lines = it minus print_time's desc / took lines and the closing done.
    sanity = ("loading parameters\nloading parameters took 1.00s\nloading tokenizer\nloading tokenizer took 0.01s\nsanity cross-entropy\n2.4 2.399 0.0004\nsanity cross-entropy took 0.10s\n"
              "sanity log-likelihood\nll_0=-1.1\nll_1=-1.1\nll_2=-1.1\nsanity log-likelihood took 0.30s\nsanity model\n2ABC1\n2ABD1\n2XYZ1\nll_x_data=-1.0\nll_x_random=-3.0\nll_x_perturb=-1.2\n"
              "sanity model took 0.70s\n") + ll.split("loading tokenizer took 0.01s\n")[1]
    assert outputs.block_of_stdout("score", sanity) == "2.4 2.399 0.0004\nll_0=-1.1\nll_1=-1.1\nll_2=-1.1\n2ABC1\n2ABD1\n2XYZ1\nll_x_data=-1.0\nll_x_random=-3.0\nll_x_perturb=-1.2\nll_sum=-1.5\nll_mean=-0.5\n"   # --sanity true: the sanity section's prints precede the pair
    assert outputs.block_of_stdout("score", "loading parameters\nloading parameters took 1.00s\n") == ""    # a process that never reached section 7: no block
