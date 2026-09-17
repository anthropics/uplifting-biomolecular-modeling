# ---------------------------------------------------------------------------
# Preset configurations for the unified pipeline entry
# ---------------------------------------------------------------------------

PRESETS = {
    "preview": {
        "N_max_runs": 1,
        "target_template_rmsd_thres": 2.0,
        "N_sample": 100,
        "N_step": 400,
        "return_topk": 5,
        "extended_w_af2": 0.5,
        "extended_w_ptx": 0.5,
        "early_stop": True,
        "min_early_stop_rounds": 0,
        "min_early_stop_successes": 1,
    },
    "extended": {
        "N_max_runs": 1,
        "target_template_rmsd_thres": 2.0,
        "N_sample": 500,
        "N_step": 400,
        "return_topk": 5,
        "extended_w_af2": 0.5,
        "extended_w_ptx": 0.5,
        "early_stop": True,
        "min_early_stop_rounds": 0,
        "min_early_stop_successes": 1,
    },
}
