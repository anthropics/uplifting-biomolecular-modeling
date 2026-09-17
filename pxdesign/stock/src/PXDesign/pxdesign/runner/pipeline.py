# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unified diffusion-only design pipeline.

Presets:
- preview  : AF2-only pipeline with lighter sampling and ranking.
- extended : AF2 + Protenix combined ranking.

There are two distinct ranking / evaluation logics:
  - "preview"  : AF2-only ranking rules
  - "extended" : joint AF2 + Protenix ranking rules
The ranking logic used is completely decoupled from the CLI:
it is automatically inferred from the actual outputs instead of any user flag.

Decision rule:
  - If PTX-related columns exist in the final summary table
      (ptx_* or ptx_mini_*)
      → use the "extended" ranking pipeline
  - Otherwise
      → fall back to the "preview" (AF2-only) ranking pipeline

Likewise, whether target-template logic (use_target_template_or_not) is executed
is automatically determined by whether the Protenix filter is enabled in the
evaluation configuration, instead of being controlled by CLI arguments.

Assumes a SINGLE task (i.e., no per-task grouping loops).
"""

import argparse
import copy
import json
import logging
import os
import time

import torch
from protenix.config import save_config
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.seed import seed_everything
from pxdbench.run import run_task
from pxdbench.utils import convert_cifs_to_pdbs, str2bool

from pxdesign.runner.dumper import DataDumper
from pxdesign.runner.helpers import save_top_designs, use_target_template_or_not
from pxdesign.runner.inference import InferenceRunner
from pxdesign.runner.presets import PRESETS
from pxdesign.utils.infer import (
    convert_to_bioassembly_dict,
    derive_seed,
    download_inference_cache,
    get_configs,
)
from pxdesign.utils.inputs import process_input_file
from pxdesign.utils.pipeline import check_tool_weights

logger = logging.getLogger(__name__)


def _get_overridden_keys(argv) -> set:
    """
    Inspect raw argv and infer which long-form options were explicitly set
    by the user, so that presets do NOT overwrite those values.
    """
    if argv is None:
        return set()

    overridden = set()
    it = iter(argv)
    for token in it:
        if not token.startswith("-"):
            continue
        if token.startswith("--"):
            # --foo or --foo=bar
            name = token[2:]
            if "=" in name:
                name = name.split("=", 1)[0]
            overridden.add(name.replace("-", "_"))
    return overridden


class DesignPipeline(InferenceRunner):
    def __init__(self, *args, use_ptx_filter: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_run = 0
        self.use_ptx_filter = use_ptx_filter

    def infer_and_eval(
        self,
        seed: int,
        run_id: int,
        pipeline_args,
        progress_per_infer: float = 10.0,
        progress_per_eval: float = 30.0,
    ):
        self.dump_dir = os.path.join(
            self.configs.dump_dir, f"global_run_{self.global_run}"
        )
        os.makedirs(self.dump_dir, exist_ok=True)
        self.dumper = DataDumper(base_dir=self.dump_dir)

        seed_everything(seed=seed, deterministic=True)
        orig_seqs = self._inference(seed)

        if DIST_WRAPPER.world_size > 1:
            torch.distributed.barrier()

        if self.use_ptx_filter:
            use_target_template = None
            if DIST_WRAPPER.rank == 0:
                task_name = list(orig_seqs.keys())[0]
                gt_cif_path = os.path.join(
                    self.dump_dir,
                    task_name,
                    f"seed_{seed}",
                    "predictions",
                    f"{task_name}_sample_0.cif",
                )
                use_target_template = use_target_template_or_not(
                    self.configs.eval,
                    pipeline_args,
                    gt_cif_path,
                    orig_seqs[task_name],
                    task_name,
                    os.path.join(self.configs.dump_dir, "target_pred"),
                    device="cuda:0",
                    seed=seed,
                )
            use_templ_list = DIST_WRAPPER.all_gather_object(use_target_template)
            print("use_templ_list: ", use_templ_list)
            use_target_template = [x for x in use_templ_list if x is not None][0]
            print("use_target_template: ", use_target_template)

            if use_target_template:
                self.configs.eval.binder.tools.ptx.use_template = True
                self.configs.eval.binder.tools.ptx.use_msa = False
                self.configs.eval.binder.tools.ptx.model_name = (
                    "protenix_mini_tmpl_v0.5.0"
                )
                print("[INFO] Use target template in the Protenix filter!")
        else:
            use_target_template = False

        cur_progress = (
            run_id * (progress_per_infer + progress_per_eval) + progress_per_infer
        )
        if DIST_WRAPPER.rank == 0:
            print(f"----------Current progress: {cur_progress:.2f}%----------")

        eval_inputs = []
        for task_name in orig_seqs:
            input_dir = os.path.join(
                self.dump_dir, task_name, f"seed_{seed}", "predictions"
            )
            if not os.path.exists(input_dir):
                logger.warning(f"Cannot find inference results under {input_dir}")
                continue
            pdb_dir, pdb_names, cond_chains, binder_chains = convert_cifs_to_pdbs(
                input_dir
            )
            eval_inputs.append(
                {
                    "task": "binder",
                    "name": task_name,
                    "pdb_dir": pdb_dir,
                    "pdb_names": pdb_names,
                    "cond_chains": cond_chains,
                    "binder_chains": binder_chains,
                    "out_dir": input_dir,
                    "orig_seqs": orig_seqs[task_name],
                }
            )
        results = [
            run_task(
                eval_inputs[i],
                self.configs.eval,
                device_id=DIST_WRAPPER.local_rank,
                seed=seed,
            )
            for i in range(len(eval_inputs))
        ]
        all_eval = DIST_WRAPPER.all_gather_object(results)
        all_eval = [x for sub in all_eval for x in sub]
        cur_progress = (run_id + 1) * (progress_per_infer + progress_per_eval)
        if DIST_WRAPPER.rank == 0:
            print("all eval results: ", all_eval)
            print(f"----------Current progress: {cur_progress:.2f}%----------")
        return all_eval, orig_seqs, use_target_template


# ---------- CLI & orchestration ----------


def parse_pipeline_args(argv=None):
    """
    Parse pipeline-level CLI arguments (high-level presets + core knobs).
    Remaining args are passed to get_configs (model/eval config).
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preset",
        type=str,
        choices=["preview", "extended", "custom"],
        default="preview",
        help=(
            "High-level pipeline preset. "
            "'preview' / 'extended' set a bundle of defaults "
            "for sampling and ranking. 'none' disables presets."
        ),
    )

    parser.add_argument(
        "--N_max_runs",
        type=int,
        default=1,
        help="Max number of global pipeline rounds.",
    )
    parser.add_argument(
        "--target_template_rmsd_thres",
        type=float,
        default=2.0,
        help="Max RMSD between GT target and prediction to treat as 'template-like'.",
    )

    # Output and ranking caps
    parser.add_argument(
        "--return_topk",
        type=int,
        default=5,
        help="How many designs to keep per task after ranking.",
    )
    parser.add_argument(
        "--min_total_return",
        type=int,
        default=10,
        help="If total successes < this, pad with failed designs up to this total.",
    )
    parser.add_argument(
        "--max_success_return",
        type=int,
        default=20,
        help="Max number of success rows to return.",
    )
    parser.add_argument(
        "--extended_w_af2",
        type=float,
        default=0.5,
        help="Weight for AF2 rank in extended (AF2+Protenix) ranking.",
    )
    parser.add_argument(
        "--extended_w_ptx",
        type=float,
        default=0.5,
        help="Weight for PTX rank in extended (AF2+Protenix) ranking.",
    )

    # Early-stop knobs
    parser.add_argument(
        "--early_stop",
        type=str2bool,
        default=True,
        help="Whether to early-stop the global pipeline if enough successes are accumulated.",
    )
    parser.add_argument(
        "--min_early_stop_rounds",
        type=int,
        default=0,
        help="Min number of rounds before early-stop is allowed.",
    )
    parser.add_argument(
        "--min_early_stop_successes",
        type=int,
        default=1,
        help="Min number of total successes required to trigger early-stop.",
    )

    overridden_keys = _get_overridden_keys(argv)
    pipeline_args, remaining = parser.parse_known_args(argv)

    preset_name = pipeline_args.preset
    if preset_name and preset_name != "custom":
        preset_cfg = PRESETS.get(preset_name, {})
        for key, value in preset_cfg.items():
            # Do not overwrite CLI-explicit arguments
            if key in overridden_keys:
                continue
            setattr(pipeline_args, key, value)

    return pipeline_args, remaining


def parse_args(argv=None):
    """
    Top-level argument parser:
      - parse pipeline-level knobs and presets
      - parse model/eval configs via get_configs
      - inject dtype and deepspeed flags into eval configs
    """
    pipeline_args, remaining_args = parse_pipeline_args(argv)
    configs = get_configs(remaining_args)
    for tool_name in ["ptx_mini", "ptx"]:
        configs["eval"]["binder"]["tools"][tool_name].update(
            {
                "dtype": configs.dtype,
                "use_deepspeed_evo_attention": configs.use_deepspeed_evo_attention,
            }
        )
    return configs, vars(pipeline_args)


def detect_use_ptx_filter(configs) -> bool:
    """
    Detect whether Protenix filter is enabled in eval configs.
    """
    binder_cfg = configs.eval.binder
    use_ptx = False
    for attr in ["eval_protenix", "eval_protenix_mini"]:
        if hasattr(binder_cfg, attr) and getattr(binder_cfg, attr):
            use_ptx = True
    return use_ptx


def main(argv=None):
    configs, p = parse_args(argv)
    os.makedirs(configs.dump_dir, exist_ok=True)
    configs.input_json_path = process_input_file(
        configs.input_json_path, out_dir=configs.dump_dir
    )
    download_inference_cache(configs)
    check_tool_weights()

    # convert cif / pdb to bioassembly dict
    if DIST_WRAPPER.rank == 0:
        save_config(configs, os.path.join(configs.dump_dir, "config.yaml"))
        with open(configs.input_json_path, "r") as f:
            orig_inputs = json.load(f)
        for x in orig_inputs:
            convert_to_bioassembly_dict(x, configs.dump_dir)
        configs.input_json_path = os.path.join(configs.dump_dir, "pipeline_input.json")
        with open(configs.input_json_path, "w") as f:
            json.dump(orig_inputs, f, indent=4)

    if DIST_WRAPPER.world_size > 1:
        if DIST_WRAPPER.rank == 0:
            new_inputs = []
            with open(configs.input_json_path, "r") as f:
                ori_input = json.load(f)[0]
            for i in range(DIST_WRAPPER.world_size):
                new_input = copy.deepcopy(ori_input)
                new_input["name"] = ori_input["name"] + f"_chunk{i}"
                new_inputs.append(new_input)
            with open(
                os.path.join(configs.dump_dir, "chunk_pipeline_input.json"), "w"
            ) as f:
                json.dump(new_inputs, f, indent=4)
        # split N_sample over workers
        if hasattr(configs, "sample_diffusion") and hasattr(
            configs.sample_diffusion, "N_sample"
        ):
            configs.sample_diffusion.N_sample = (
                configs.sample_diffusion.N_sample // DIST_WRAPPER.world_size
            )
        configs.input_json_path = os.path.join(
            configs.dump_dir, "chunk_pipeline_input.json"
        )

    use_ptx_filter = detect_use_ptx_filter(configs)
    runner = DesignPipeline(configs, use_ptx_filter=use_ptx_filter)

    N = p["N_max_runs"]
    seeds = configs.seeds
    if not seeds:
        base = time.time_ns()
        seeds = [(base + i) % (2**31 - 1) for i in range(N)]
    else:
        assert len(seeds) == N, "The number of seeds must equal N_max_runs"

    progress_per_infer = round(30.0 / p["N_max_runs"])
    progress_per_eval = round(60.0 / p["N_max_runs"])

    cumulative_success = {}  # name -> int

    for i in range(p["N_max_runs"]):
        with open(configs.input_json_path, "r") as f:
            cur_inputs = json.load(f)

        local_seed = derive_seed(seeds[i], DIST_WRAPPER.rank)
        runner.local_print(f"----------Pipeline with seed {local_seed}----------")
        runner.print(f"Current {len(cur_inputs)} design tasks for loop {i}:")
        runner.print(f"Current tasks: {cur_inputs}")

        all_eval_results, orig_seqs, use_target_template = runner.infer_and_eval(
            seed=local_seed,
            run_id=i,
            pipeline_args=p,
            progress_per_infer=progress_per_infer,
            progress_per_eval=progress_per_eval,
        )
        assert len(cur_inputs) == len(all_eval_results)

        # save meta info
        meta_info = {"mode": "Extended" if use_ptx_filter else "Preview"}
        if use_ptx_filter:
            if use_target_template:
                runner.configs.eval.binder.tools.ptx.use_template = True
                runner.configs.eval.binder.tools.ptx.use_msa = False
                runner.configs.eval.binder.tools.ptx.model_name = (
                    "protenix_mini_tmpl_v0.5.0"
                )
                meta_info["protenix"] = "Protenix-Mini-Templ"
            else:
                meta_info["protenix"] = "Protenix"
        if DIST_WRAPPER.rank == 0:
            task_name = cur_inputs[0]["name"]
            if DIST_WRAPPER.world_size > 1:
                task_name = task_name[: -len("_chunk*")]
            output_dir = os.path.join(configs.dump_dir, "design_outputs", task_name)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "task_info.json"), "w") as f:
                json.dump(meta_info, f, indent=4)

        # accumulate successes by SINGLE key (af2_easy_success)
        for item in all_eval_results:
            name = item["name"]
            with open(item["summary_save_path"], "r") as f:
                summary = json.load(f)
            cnt = int(summary.get("af2_easy_success.count", 0))
            cumulative_success[name] = cumulative_success.get(name, 0) + cnt

        print(
            f"[Loop {i}] Designed: {[it['name'] for it in all_eval_results]}, "
            f"cumulative success: {cumulative_success}"
        )

        success_names = []
        for data in cur_inputs:
            task_name = data["name"]
            success_count = cumulative_success.get(task_name, 0)
            if (
                ((i + 1) >= p["min_early_stop_rounds"])
                and (success_count >= p["min_early_stop_successes"])
                and p["early_stop"]
            ) or i == p["N_max_runs"] - 1:
                success_names.append(task_name)

        runner.global_run += 1
        next_inputs = [x for x in cur_inputs if x["name"] not in success_names]
        if not next_inputs or i == p["N_max_runs"] - 1:
            print("Finish all designs!")
            if DIST_WRAPPER.rank == 0:
                save_top_designs(
                    p,
                    configs,
                    orig_seqs,
                    use_template=use_target_template,
                )
            break

    if DIST_WRAPPER.rank == 0:
        print("----------Current progress: 100.00%----------")


if __name__ == "__main__":
    main()
