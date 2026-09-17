"""The clean environment and its proof: package variables, must-be-absent prefixes, PYTHONSAFEPATH and kit PYTHONPATH entries are removed and
nothing is added but PYTHONDONTWRITEBYTECODE; the probe child proves the result with the core's clean-process proof; a forbidden name makes the
proof not ok. The census counts this run's designs apart from earlier ones and names an incomplete run."""
import os

from complexa_opt import outputs, stack, stock_design


def test_clean_env_removes_the_named_classes_and_keeps_data_paths(tmp_path):
    base = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "COMPLEXA_OPT": "off", "COMPLEXA_KIT_X": "1", "NVIDIA_TF32_OVERRIDE": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "PYTHONSAFEPATH": "1", "COMPLEXA_INIT": "1", "LOCAL_CODE_PATH": "/opt/pc",
            "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": os.pathsep.join([stack.package_dir(), "/elsewhere", os.path.join(stack.tree_home(), "opt")])}
    env, removed = stock_design.clean_env(base)
    assert removed["variables"] == sorted(["COMPLEXA_OPT", "COMPLEXA_KIT_X", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "PYTHONSAFEPATH"])
    assert removed["pythonpath"] == [stack.package_dir(), os.path.join(stack.tree_home(), "opt")]
    assert env["PYTHONPATH"] == "/elsewhere" and env["COMPLEXA_INIT"] == "1" and env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert set(env) == (set(base) - set(removed["variables"])) | {"PYTHONDONTWRITEBYTECODE"}     # nothing of the package's is added to the child
    for k in removed["variables"]:
        assert k not in env
    env2, _ = stock_design.clean_env({"PATH": "/bin", "PYTHONPATH": stack.package_dir()})
    assert "PYTHONPATH" not in env2                                                                # a PYTHONPATH holding only kit entries is dropped whole


def test_probe_child_proves_the_clean_environment(tmp_path):
    env, removed = stock_design.clean_env({"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "COMPLEXA_OPT": "off", "LOCAL_CODE_PATH": "/opt/pc"})
    proof = stock_design.env_proof(env, cwd=str(tmp_path))
    assert proof["ok"] is True, proof
    assert proof["forbidden_present"] == [] and proof["kit_modules_loaded"] == [] and proof["kit_dirs_on_path"] == [] and proof["torch_loaded_before_proof"] is False
    assert "LOCAL_CODE_PATH" in proof["present"] and "PYTHONPATH" not in proof["present"]
    line = stock_design.env_clean_line(proof, removed)
    assert line.startswith("[complexa-opt] ENV-CLEAN ok (1 variables removed: COMPLEXA_OPT; proof: ") and "torch_preloaded=False" in line and "LOCAL_CODE_PATH" in line
    dirty = dict(env, COMPLEXA_KIT_LEVER="1", PYTHONPATH=stack.package_dir())
    bad = stock_design.env_proof(dirty, cwd=str(tmp_path))
    assert bad["ok"] is False and bad["forbidden_present"] == ["COMPLEXA_KIT_LEVER"] and bad["kit_dirs_on_path"]
    assert stock_design.env_clean_line(bad, removed).startswith("[complexa-opt] NOT STOCK: ")


def test_census_counts_this_runs_designs_and_names_a_short_run(tmp_path):
    out = tmp_path / "o"
    root = out / "inference" / "search_binder_local_pipeline_t1_p1"
    assert outputs.run_root(str(out), "t1", "p1") == str(root)
    assert outputs.run_root(str(out), "t1", None) == str(out / "inference" / "search_binder_local_pipeline_t1")
    assert outputs.run_root(str(out), None, "p1") == str(out / "inference" / "search_binder_local_pipeline")      # upstream: no task name, no run suffix
    assert outputs.results_csv(str(out)) == str(out / "inference" / "results_search_binder_local_pipeline_0.csv")
    d0 = root / "job_0_n_200_id_0_single_orig0"; d0.mkdir(parents=True); (d0 / "job_0_n_200_id_0_single_orig0.pdb").write_text("ATOM\n")
    c = outputs.census(str(out), str(root), "t1", 4)
    assert c["designs_written"] == 1 and c["designs_expected"] == 4 and c["items_complete"] is False and c["incomplete"] == "designs 1/4" and c["designs_preexisting"] == 0
    before = set(outputs.design_files(str(root)))                                                 # a second run into the same root: earlier designs counted apart, never refused
    d1 = root / "job_0_n_200_id_1_single_orig1"; d1.mkdir(); (d1 / "job_0_n_200_id_1_single_orig1.pdb").write_text("ATOM\n")
    c = outputs.census(str(out), str(root), "t1", 1, preexisting=before)
    assert c["designs_written"] == 1 and c["designs_preexisting"] == 1 and c["items_complete"] is True and c["incomplete"] is None
    assert c["files"] == ["inference/search_binder_local_pipeline_t1_p1/job_0_n_200_id_1_single_orig1/job_0_n_200_id_1_single_orig1.pdb"]
    c = outputs.census(str(out), str(root), "t1", 1)                                               # more PDBs than designs asked (upstream's best-of-n keeps one per replica): complete, never short
    assert c["designs_written"] == 2 and c["designs_expected"] == 1 and c["items_complete"] is True and c["incomplete"] is None
    c = outputs.census(str(out), str(root), "t1", None)                                            # a design count the package could not read: complete = wrote something
    assert c["designs_expected"] is None and c["items_complete"] is True
    c = outputs.census(str(out), str(out / "inference" / "nope"), "t2", 4)
    assert c["incomplete"] == "run root missing" and c["item_failed"] == {"t2": "run root missing"}
    assert outputs.newest_root(str(out), 0.0) == str(root) and outputs.newest_root(str(out), 4e9) is None
