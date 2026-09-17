"""The shipped values against the carried config files; the values read back from the tokens (never judged); the argv the stock route
composes — the caller's tokens last."""
import os

from complexa_opt import settings, stock_design

TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(TREE, "stock", "src")


def test_shipped_values_are_lines_of_the_carried_configs():
    for rec in settings.SHIPPED:
        with open(os.path.join(SRC, rec["file"]), encoding="utf-8") as fh:
            text = fh.read()
        assert rec["line"] in text, rec
    binder = open(os.path.join(SRC, "configs/pipeline/binder/binder_generate.yaml")).read()
    assert settings.SHIPPED_BATCH == 16 and "batch_size: 16" in binder
    assert settings.SHIPPED_NSAMPLES == 4 and "nsamples: 4" in binder and settings.SHIPPED_NREPEAT == 1 and "nrepeat_per_sample: 1" in binder
    assert settings.SHIPPED_SEED == 5 and "seed: 5" in open(os.path.join(SRC, "configs/search_binder_local_pipeline.yaml")).read()
    assert settings.SHIPPED_RUN_NAME == "search_binder_local"
    assert settings.SHIPPED_SEARCH == "best-of-n" and "algorithm: best-of-n" in binder and "reward_model:" in binder   # the shipped search scores with upstream's reward model; the generation stage alone is the caller's two tokens


def test_values_of_reads_the_tokens_else_the_shipped_values_and_never_raises():
    assert settings.values_of([]) == {"designs": 4, "nsamples": 4, "nrepeat": 1, "counted": True, "batch": 16, "batch_given": False, "seed": 5, "seed_given": False}
    v = settings.values_of(["++generation.dataloader.dataset.nres.nsamples=8", "++seed=11", "++generation.dataloader.batch_size=8", "++generation.dataloader.batch_size=32",
                            "++generation.dataloader.dataset.nrepeat_per_sample=2", "++generation.args.nsteps=200"])
    assert v == {"designs": 16, "nsamples": 8, "nrepeat": 2, "counted": True, "batch": 32, "batch_given": True, "seed": 11, "seed_given": True}   # the last token of a key wins
    v = settings.values_of(["++generation.dataloader.dataset.nres.nsamples=abc", "++seed=x", "~generation.dataloader.batch_size"])
    assert v["counted"] is False and v["designs"] is None and v["nsamples"] == "abc" and v["seed"] == "x" and v["batch"] == 16   # read, recorded, never refused: upstream answers for them
    assert settings.last_value(["++run_name=a", "run_name=b", "~run_name"], "run_name") is None
    assert settings.last_value(["++run_name=a", "+run_name=b"], "run_name") == "b"
    assert not hasattr(settings, "MANAGED_KEYS") and not hasattr(settings, "check_overrides")   # no key of upstream's is reserved by the package


ENTRY = {"source": "custom", "target_filename": "1jz7A_N200", "target_path": None, "target_input": "A544-663", "hotspot_residues": ["A590", "A602"], "binder_length": [80, 80], "pdb_id": None}


def _entry(tmp_path):
    pdb = tmp_path / "target.pdb"
    pdb.write_text("ATOM\n")
    return dict(ENTRY, target_path=str(pdb))


def test_compose(tmp_path):
    e = _entry(tmp_path)
    common = ["/usr/local/bin/complexa", "generate", "/opt/pc/configs/search_binder_local_pipeline.yaml", "--verbose",
              "++generation.task_name=1jz7A_N200",
              "++generation.target_dict_cfg={1jz7A_N200:{source:'custom',target_filename:'1jz7A_N200'," f"target_path:'{e['target_path']}',"
              "target_input:'A544-663',hotspot_residues:['A590','A602'],binder_length:[80,80],pdb_id:null}}",
              "++ckpt_path=/weights", "++autoencoder_ckpt_path=/weights/complexa_ae.ckpt"]
    kw = dict(console_script="/usr/local/bin/complexa", config_path="/opt/pc/configs/search_binder_local_pipeline.yaml", item="1jz7A_N200", entry=e, weights_dir="/weights")
    assert stock_design.compose(overrides=[], **kw) == common                      # no token: upstream's shipped configuration exactly (best-of-n search scored by its reward model, 4 designs, seed 5, batch 16)
    tokens = ["++run_name=p1", "++generation.search.algorithm=single-pass", "++generation.reward_model=null", "++generation.dataloader.dataset.nres.nsamples=6", "++seed=7",
              "++generation.dataloader.batch_size=32", "--job-id", "2"]
    assert stock_design.compose(overrides=tokens, **kw) == common + tokens         # the caller's tokens verbatim, in order, LAST (a later Hydra token wins: nothing reserved, nothing added)
    assert not hasattr(stock_design, "SCOPE_TOKENS")                                # the package scopes nothing: the generation-stage tokens are the caller's to give
    # --input is optional: without it the command names no target of its own
    bare = stock_design.compose(console_script="/usr/local/bin/complexa", config_path="/opt/pc/configs/search_binder_local_pipeline.yaml", weights_dir="/w", overrides=["++generation.task_name=02_PDL1"])
    assert bare == ["/usr/local/bin/complexa", "generate", "/opt/pc/configs/search_binder_local_pipeline.yaml", "--verbose", "++ckpt_path=/w", "++autoencoder_ckpt_path=/w/complexa_ae.ckpt",
                    "++generation.task_name=02_PDL1"]


def test_effective_names_follow_hydras_last_token_rule():
    assert stock_design.effective_names("t1", ["++run_name=p1"]) == ("t1", "p1")
    assert stock_design.effective_names("t1", []) == ("t1", "search_binder_local")                # no run name given: the shipped run_name suffixes the root
    assert stock_design.effective_names("t1", ["++run_name=p1", "++run_name=x", "++generation.task_name=02_PDL1"]) == ("02_PDL1", "x")   # the last token wins
    assert stock_design.effective_names(None, ["++generation.task_name=02_PDL1"]) == ("02_PDL1", "search_binder_local")
    assert stock_design.effective_names(None, []) == (None, "search_binder_local")

