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
PXDesign command line interface (CLI).

This CLI unifies the entry points from the existing scripts:

- inference.py       →   raw diffusion inference (no evaluation)
- pipeline.py        →   multi-run infer + eval + collect design pipeline

We expose a single `pxdesign` command with subcommands:

    pxdesign infer    ...   # raw diffusion sampling
    pxdesign pipeline ...   # full design pipeline (with presets)

For the `pipeline` subcommand, common usage patterns are captured via presets:

    --preset preview   : multi-run AF2-only pipeline (PTX filters disabled)
    --preset extended  : multi-run AF2 + Protenix pipeline (default)

Internally, `pipeline.py` still has two ranking modes:

  - "preview"  : AF2-only ranking
  - "extended" : AF2 + Protenix ranking

But which mode is actually used is decided *inside* the pipeline, based on
whether PTX-related columns exist in the final summary and whether the
Protenix filter is enabled in the eval config.
"""

import logging
import os
from functools import wraps
from pathlib import Path
from typing import Dict, List

import click

# ---------------------------------------------------------------------------
# Helpers for shared options
# ---------------------------------------------------------------------------


def build_argv(common: Dict[str, object], extra: List[str]) -> List[str]:
    """
    Convert the shared parameter dict plus extra args into a flat argv list.
    """
    argv: List[str] = []
    for key, value in common.items():
        flag = f"--{key}"
        argv.extend([flag, str(value)])
    return [*argv, *extra]


def common_run_options(func):
    """
    Attach shared CLI options and pass them as a single `common` dict argument.

    Shared options:
        --dump_dir / -o
        --input / -i
        --dtype
        --N_sample
        --N_step
        --eta_type
        --eta_min
        --eta_max

    The wrapped function will receive:

        def cmd(ctx, common, ...):
            # `common` is a dict with the above keys.
    """

    # Define options from "bottom" to "top" (decorators are applied in reverse order)
    @click.option(
        "--eta_max",
        type=float,
        default=2.5,
        show_default=True,
        help="Maximum value for the eta schedule.",
    )
    @click.option(
        "--eta_min",
        type=float,
        default=2.5,
        show_default=True,
        help="Minimum value for the eta schedule.",
    )
    @click.option(
        "--eta_type",
        type=str,
        default="const",
        show_default=True,
        help="Eta schedule type.",
    )
    @click.option(
        "--N_step",
        "n_step",
        type=int,
        default=400,
        show_default=True,
        help="Number of diffusion steps.",
    )
    @click.option(
        "--N_sample",
        "n_sample",
        type=int,
        default=5,
        show_default=True,
        help="Number of diffusion samples.",
    )
    @click.option(
        "--dtype",
        type=click.Choice(["fp32", "bf16"], case_sensitive=False),
        default="bf16",
        show_default=True,
        help="Inference dtype.",
    )
    @click.option(
        "--input",
        "-i",
        "input_path",
        type=click.Path(exists=True, dir_okay=False),
        required=True,
        help="Path to the input file (YAML/JSON).",
    )
    @click.option(
        "--dump_dir",
        "-o",
        type=str,
        required=True,
        help="Directory where outputs will be written.",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):

        # Extract common options from kwargs and pack them into a single dict.
        common = dict(
            dump_dir=kwargs.pop("dump_dir"),
            input_json_path=kwargs.pop("input_path"),
            dtype=kwargs.pop("dtype"),
            N_sample=kwargs.pop("n_sample"),
            N_step=kwargs.pop("n_step"),
            eta_type=kwargs.pop("eta_type"),
            eta_min=kwargs.pop("eta_min"),
            eta_max=kwargs.pop("eta_max"),
        )
        # Pass `common` as a single keyword argument to the wrapped function.
        return func(*args, common=common, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Root CLI command group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """
    PXDesign — protein binder design pipeline.

    \b
    This top-level click group provides subcommands:
        - infer    : raw diffusion inference (no eval)
        - pipeline : full design pipeline (multi-run, AF2-only or AF2+PTX)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )


# ---------------------------------------------------------------------------
# `infer` subcommand — direct diffusion inference (inference.py)
# ---------------------------------------------------------------------------


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,  # allow argparse to handle unknown args
        allow_extra_args=True,  # pass remaining args to argparse
    )
)
@common_run_options
@click.pass_context
def infer(ctx: click.Context, common: Dict[str, object]) -> None:
    """
    Run raw diffusion inference without evaluation (inference.py).

    This command forwards the shared run options plus any remaining CLI
    arguments directly to the argparse parser inside inference.py.

    \b
    Usage:
    pxdesign infer
          -o out
          -i input.json
          --dtype bf16
          --N_sample 100
          --N_step 400
          --eta_type const --eta_min 2.5 --eta_max 2.5
          [other inference-specific args...]
    """
    from . import inference as _inference

    extra_args: List[str] = list(ctx.args)
    argv = build_argv(common, extra_args)
    _inference.main(argv)


# ---------------------------------------------------------------------------
# `pipeline` subcommand — full design pipeline (pipeline.py)
# ---------------------------------------------------------------------------


@cli.command(
    name="pipeline",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@common_run_options
@click.option(
    "--preset",
    type=click.Choice(["preview", "extended", "custom"], case_sensitive=False),
    default="custom",
    show_default=True,
    help=(
        "Pipeline preset:\n"
        "  preview  : multi-run AF2-only (PTX filters disabled via Hydra overrides)\n"
        "  extended : multi-run AF2 + Protenix (default)\n"
    ),
)
@click.option(
    "--N_max_runs",
    "n_max_runs",
    type=int,
    default=1,
    show_default=True,
    help="Maximum number of design runs (default: 1).",
)
@click.pass_context
def pipeline(
    ctx: click.Context,
    common: Dict[str, object],
    preset: str,
    n_max_runs: int | None,
) -> None:
    """
    Run the full multi-run design pipeline (pipeline.py).

    \b
    Internally, the pipeline will:
      - perform diffusion-based inference,
      - run AF2 and optionally Protenix filters,
      - aggregate sample-level outputs across runs,
      - automatically choose AF2-only vs AF2+PTX ranking rules
        based on the presence of PTX columns and eval configs.

    The `--preset` flag only controls how some frequently used arguments are
    filled in by default; the actual ranking mode is decided *inside*
    `pipeline.py` and not by the CLI.
    """
    from . import pipeline as _pipeline

    preset = preset.lower()
    extra_args: List[str] = list(ctx.args)
    preset_args: List[str] = []
    preset_args.extend(["--N_max_runs", str(n_max_runs)])

    # ---- Apply preset-specific overrides ----
    n_sample = int(common["N_sample"])

    if preset == "preview":
        # Preview preset:
        #   - multi-run AF2-only pipeline
        #   - PTX filters disabled via Hydra-style overrides
        preset_args.extend(
            [
                "--eval.binder.eval_complex",
                "true",
                "--eval.binder.eval_binder_monomer",
                "true",
                "--eval.binder.eval_protenix",
                "false",
                "--eval.binder.eval_protenix_mini",
                "false",
            ]
        )

    elif preset == "extended":
        # Extended preset:
        #   - multi-run AF2 + Protenix
        #   - do NOT override eval config; rely on config defaults,
        #     which should enable Protenix filters in this mode.
        # Users can still tweak Protenix-related knobs via extra args.
        preset_args.extend(
            [
                "--eval.binder.eval_complex",
                "true",
                "--eval.binder.eval_binder_monomer",
                "true",
                "--eval.binder.eval_protenix",
                "true",
                "--eval.binder.eval_protenix_mini",
                "false",
            ]
        )

    elif preset == "custom":
        # Custom preset:
        #   - no automatic overrides except optional N_max_runs (if provided)
        #   - user must provide all pipeline-related knobs explicitly.
        pass

    else:
        raise click.BadParameter(f"Unknown preset: {preset!r}")

    preset_args.extend(
        [
            "--min_total_return",
            str(n_sample),
            "--max_success_return",
            str(n_sample),
        ]
    )

    argv = build_argv(common, [*preset_args, *extra_args])
    _pipeline.main(argv)


# ---------------------------------------------------------------------------
# `check-input` subcommand — validate YAML config
# ---------------------------------------------------------------------------


@cli.command(name="check-input")
@click.option(
    "--yaml",
    "yaml_file",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the YAML configuration file to validate.",
)
def check_input(yaml_file: str) -> None:
    """
    Validate the format and content of a design YAML file.

    Usage:
        pxdesign check-input --yaml <file.yaml>
    """
    from ..utils import inputs

    inputs.check_yaml_file(yaml_file)


# ---------------------------------------------------------------------------
# `parse-target` subcommand — visual verification tool
# ---------------------------------------------------------------------------


@cli.command(name="parse-target")
@click.option(
    "--yaml",
    "yaml_file",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the input YAML configuration file.",
)
@click.option(
    "--output_dir",
    "-o",
    "out_dir",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Directory where the parsed CIF file will be saved.",
)
def parse_target(yaml_file: str, out_dir: str) -> None:
    """
    Parse and crop the target structure based on the YAML configuration.

    This command generates a `*_target_parsed.cif` file that represents
    exactly what the model "sees" internally, with chains re-labeled and
    regions cropped according to the input configuration.
    The output is intended for visual inspection in PyMOL or Mol* to
    verify that crop and hotspot settings are correctly specified.

    Usage:
        pxdesign parse-target --yaml <file.yaml> -o <debug_dir>
    """

    # Lazy import to avoid slowing down CLI startup
    from ..utils import inputs

    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    print(f"Parsing target from {yaml_file} ...")
    try:
        # Call the backend utility function to parse the target
        # Note: `file_path` corresponds to the input YAML file
        inputs.dump_target_cif_from_input_file(
            file_path=yaml_file,
            out_dir=out_dir,
        )
        print(f"✅ Parsed target saved to: {out_dir}")
    except Exception as e:
        # Catch assertion errors or parsing failures and report a user-friendly message
        print(f"❌ Error parsing target: {e}")
        # Exit with a non-zero status code to indicate failure in CLI usage
        exit(1)


# ---------------------------------------------------------------------------
# `prepare-msa` subcommand
# ---------------------------------------------------------------------------


@cli.command(name="prepare-msa")
@click.option(
    "--yaml",
    "yaml_file",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the input YAML configuration file.",
)
@click.option(
    "--output_yaml",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Optional path to write the updated YAML file. "
    "If not provided, the input YAML will be modified in-place.",
)
def prepare_msa(yaml_file: str, output_yaml: str | None) -> None:
    """
    Populate target-chain MSA paths in a PXDesign YAML configuration.

    This command injects precomputed MSA directories into the input YAML
    file under `target.chains[*].msa`. Existing MSA entries are preserved.

    Usage:
        pxdesign prepare-msa --yaml <file.yaml>
        pxdesign prepare-msa --yaml <file.yaml> --output_yaml <new.yaml>
    """
    import tempfile

    import yaml
    from protenix.data.json_maker import cif_to_input_json
    from pxdbench.tools.ptx.ptx_utils import populate_msa_with_cache

    from pxdesign.data.utils import pdb_to_cif
    from pxdesign.utils.infer import build_chain_mapping

    # -----------------------------
    # helpers
    # -----------------------------
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise click.ClickException(
                f"Invalid YAML: top-level must be a mapping: {path}"
            )
        if "target" not in cfg or "chains" not in cfg["target"]:
            raise click.ClickException("Invalid YAML: missing `target.chains`.")
        if "file" not in cfg["target"]:
            raise click.ClickException("Invalid YAML: missing `target.file`.")
        return cfg

    def _maybe_convert_pdb_to_cif_and_mapping(
        cfg: dict,
    ) -> tuple[str, dict[str, str] | None]:
        """
        Returns:
            cif_file: path to CIF (original CIF or converted from PDB)
            chain_mapping: mapping from CIF label_asym_id -> YAML chain id (auth_asym_id),
                           only needed when input is PDB.
        """
        target_file = cfg["target"]["file"]
        keep_chains = list(cfg["target"]["chains"].keys())

        if target_file.endswith(".pdb"):
            fd, cif_file = tempfile.mkstemp(
                prefix="pxdesign_pdb2cif_",
                suffix=".cif",
            )
            os.close(fd)
            atom_array = pdb_to_cif(target_file, cif_file)
            m = build_chain_mapping(
                atom_array.auth_asym_id,
                atom_array.chain_id,
                keep_chains=keep_chains,
            )
            chain_mapping = {v: k for k, v in m.items()}
            return cif_file, chain_mapping

        return target_file, None

    def _filter_input_json_by_chains(
        input_json: dict,
        avail_chains: set[str],
        chain_mapping: dict[str, str] | None = None,
    ) -> dict:
        """
        Keep only sequence entries whose label_asym_id intersects with avail_chains.

        If chain_mapping is provided (PDB case), first map CIF label_asym_id -> YAML chain id.
        Also trims label_asym_id to the kept subset.
        """
        new_sequences: list[dict] = []

        for seq in input_json.get("sequences", []):
            if not isinstance(seq, dict) or len(seq) != 1:
                raise click.ClickException(
                    f"Invalid input JSON `sequences` entry: {seq}"
                )

            entity_key, entity = next(iter(seq.items()))
            label_ids = entity.get("label_asym_id")

            # Entities without label_asym_id: keep them (rare, but safe)
            if label_ids is None:
                new_sequences.append(seq)
                continue

            keep_ids: list[str] = []
            for c in label_ids:
                if chain_mapping is not None:
                    # c is label_asym_id; map to YAML chain id if possible
                    if c not in chain_mapping:
                        continue
                    c = chain_mapping[c]
                if c in avail_chains:
                    keep_ids.append(c)

            if not keep_ids:
                continue

            new_entity = entity.copy()
            new_entity["label_asym_id"] = keep_ids
            new_sequences.append({entity_key: new_entity})

        new_json = input_json.copy()
        new_json["sequences"] = new_sequences
        return new_json

    def _extract_msa_dirs(input_json: dict) -> dict[str, str]:
        """
        Extract mapping: chain_id -> precomputed_msa_dir
        Assumes populate_msa_with_cache has added entity['msa']['precomputed_msa_dir'].
        """
        msa_dirs: dict[str, str] = {}
        for seq_entry in input_json.get("sequences", []):
            entity = next(iter(seq_entry.values()))
            msa = entity.get("msa", {})
            msa_dir = msa.get("precomputed_msa_dir")
            if not msa_dir:
                continue
            for asym_id in entity.get("label_asym_id", []):
                msa_dirs[asym_id] = msa_dir
        return msa_dirs

    # -----------------------------
    # main logic
    # -----------------------------
    cfg = _load_yaml(yaml_file)

    cif_file, chain_mapping = _maybe_convert_pdb_to_cif_and_mapping(cfg)
    avail_chains = set(cfg["target"]["chains"].keys())

    input_json = cif_to_input_json(cif_file, save_entity_and_asym_id=True)
    input_json = _filter_input_json_by_chains(
        input_json,
        avail_chains=avail_chains,
        chain_mapping=chain_mapping,
    )

    if cfg["target"]["file"].endswith(".pdb"):
        try:
            os.remove(cif_file)
        except OSError:
            pass

    # populate precomputed MSA using cache
    input_json = populate_msa_with_cache([input_json])[0]

    msa_dirs = _extract_msa_dirs(input_json)

    # sanity: ensure every requested chain has an MSA dir available
    missing = [c for c in avail_chains if c not in msa_dirs]
    if missing:
        raise click.ClickException(
            f"Missing MSA for chain(s): {missing}. Check MSA cache or chain IDs."
        )

    chains = cfg["target"]["chains"]
    updated = False

    for chain_id, chain_cfg in chains.items():
        # Skip shorthand definitions like: C: "all"
        if not isinstance(chain_cfg, dict):
            continue

        # Preserve existing msa
        if "msa" in chain_cfg:
            continue

        chain_msa_dir = msa_dirs[str(chain_id)]
        if not os.path.isdir(chain_msa_dir):
            raise click.ClickException(
                f"MSA directory not found for chain '{chain_id}': {chain_msa_dir}"
            )

        chain_cfg["msa"] = chain_msa_dir
        updated = True
        click.echo(f"✔ Added MSA for chain {chain_id}: {chain_msa_dir}")

    if not updated:
        click.echo("No changes made (all chains already have MSA specified).")

    out_path = output_yaml or yaml_file
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    click.echo(f"✅ Updated YAML saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point for running `cli.py` directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
