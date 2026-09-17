"""Python API for Caliby sequence design.

Example usage::

    from caliby import load_model

    model = load_model("caliby")
    results = model.sample(["my_protein.pdb"], num_seqs_per_pdb=4)
    print(results["seq"])
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from opt_core.oom import is_oom  # an out-of-memory is never rerouted to the serial fallback

# Set AtomWorks env vars to safe defaults before any atomworks imports.
os.environ.setdefault("PDB_MIRROR_PATH", "")
os.environ.setdefault("CCD_MIRROR_PATH", "")
os.environ.setdefault("MODEL_PARAMS_DIR", "model_params")

_PACKAGE_ROOT = Path(__file__).resolve().parent
_DEFAULT_SAMPLING_CFG_PATH = _PACKAGE_ROOT / "configs" / "seq_des" / "inference.yaml"


def _merge_sampling_cfg(
    base_cfg,
    *,
    num_seqs_per_pdb: int | None = None,
    batch_size: int | None = None,
    omit_aas: list[str] | None = None,
    num_workers: int | None = None,
    temperature: float | None = None,
    verbose: bool | None = None,
    sampling_overrides: dict[str, Any] | None = None,
):
    """Merge user-provided overrides into a copy of the base sampling config."""
    from omegaconf import OmegaConf

    overrides: dict[str, Any] = {}
    if num_seqs_per_pdb is not None:
        overrides["num_seqs_per_pdb"] = num_seqs_per_pdb
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if omit_aas is not None:
        overrides["omit_aas"] = omit_aas
    if num_workers is not None:
        overrides["num_workers"] = num_workers
    if temperature is not None:
        overrides.setdefault("potts_sampling_cfg", {})["potts_temperature"] = temperature
    if verbose is not None:
        overrides["verbose"] = verbose

    if sampling_overrides:
        overrides = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(overrides), OmegaConf.create(sampling_overrides)),
            resolve=True,
        )

    if overrides:
        return OmegaConf.merge(base_cfg, overrides)
    return base_cfg


class CalibyModel:
    """A loaded Caliby model ready for sequence design, scoring, or packing.

    Use :func:`load_model` to create an instance.
    """

    def __init__(self, model, data_cfg, sampling_cfg, device: str):
        self.model = model
        self.data_cfg = data_cfg
        self.sampling_cfg = sampling_cfg
        self.device = device

    def sample(
        self,
        pdb_paths: list[str],
        *,
        out_dir: str | None = None,
        num_seqs_per_pdb: int | None = None,
        batch_size: int | None = None,
        omit_aas: list[str] | None = None,
        num_workers: int | None = None,
        temperature: float | None = None,
        verbose: bool | None = None,
        pos_constraint_df: pd.DataFrame | None = None,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> dict[str, list]:
        """Design sequences for the given PDB/CIF structures.

        Args:
            pdb_paths: Paths to PDB or CIF files.
            out_dir: Directory for output CIF files. If None, uses a temp dir.
            num_seqs_per_pdb: Sequences to sample per structure (default: 1).
            batch_size: Batch size for processing (default: 4).
            omit_aas: Amino acid one-letter codes to exclude, e.g. ``["C"]``.
            num_workers: Data loading workers (default: 2).
            temperature: Potts sampling temperature (default: 0.01).
            verbose: Print constraint info (default: True).
            pos_constraint_df: Positional constraints. See :func:`make_constraints`.
            sampling_overrides: Advanced overrides merged into the sampling config,
                e.g. ``{"potts_sampling_cfg": {"potts_sweeps": 1000}}``.

        Returns:
            Dict with keys ``"example_id"``, ``"out_pdb"``, ``"seq"``, ``"U"``,
            ``"input_seq"``. Each value is a list.
        """
        from caliby.eval.eval_utils.seq_des_utils import run_seq_des

        merged_cfg = _merge_sampling_cfg(
            self.sampling_cfg,
            num_seqs_per_pdb=num_seqs_per_pdb,
            batch_size=batch_size,
            omit_aas=omit_aas,
            num_workers=num_workers,
            temperature=temperature,
            verbose=verbose,
            sampling_overrides=sampling_overrides,
        )
        if out_dir is None:
            out_dir = tempfile.mkdtemp(prefix="caliby_")

        return run_seq_des(
            model=self.model,
            data_cfg=self.data_cfg,
            sampling_cfg=merged_cfg,
            pdb_paths=pdb_paths,
            device=self.device,
            out_dir=out_dir,
            pos_constraint_df=pos_constraint_df,
        )

    def ensemble_sample(
        self,
        pdb_to_conformers: dict[str, list[str]],
        *,
        out_dir: str | None = None,
        num_seqs_per_pdb: int | None = None,
        batch_size: int | None = None,
        omit_aas: list[str] | None = None,
        num_workers: int | None = None,
        temperature: float | None = None,
        verbose: bool | None = None,
        pos_constraint_df: pd.DataFrame | None = None,
        use_primary_res_type: bool = True,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> dict[str, list]:
        """Design sequences using an ensemble of conformers per structure.

        Args:
            pdb_to_conformers: Maps PDB name to list of conformer file paths.
                The first conformer is the primary structure.
            out_dir: Directory for output CIF files. If None, uses a temp dir.
            use_primary_res_type: Use residue types from the primary conformer.
            (Other args same as :meth:`sample`.)

        Returns:
            Dict with keys ``"example_id"``, ``"out_pdb"``, ``"seq"``, ``"U"``,
            ``"input_seq"``.
        """
        from caliby.eval.eval_utils.seq_des_utils import run_seq_des_ensemble

        merged_cfg = _merge_sampling_cfg(
            self.sampling_cfg,
            num_seqs_per_pdb=num_seqs_per_pdb,
            batch_size=batch_size,
            omit_aas=omit_aas,
            num_workers=num_workers,
            temperature=temperature,
            verbose=verbose,
            sampling_overrides=sampling_overrides,
        )
        if out_dir is None:
            out_dir = tempfile.mkdtemp(prefix="caliby_")

        return run_seq_des_ensemble(
            model=self.model,
            data_cfg=self.data_cfg,
            sampling_cfg=merged_cfg,
            pdb_to_conformers=pdb_to_conformers,
            device=self.device,
            out_dir=out_dir,
            pos_constraint_df=pos_constraint_df,
            use_primary_res_type=use_primary_res_type,
        )

    def score(
        self,
        pdb_paths: list[str],
        *,
        batch_size: int | None = None,
        num_workers: int | None = None,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> dict[str, list]:
        """Score the native sequences of the given structures.

        Args:
            pdb_paths: Paths to PDB or CIF files.
            batch_size: Batch size for scoring (default: 4).
            num_workers: Data loading workers (default: 2).
            sampling_overrides: Advanced overrides for sampling config.

        Returns:
            Dict with keys ``"example_id"``, ``"seq"``, ``"U"``, ``"U_i"``.
        """
        from caliby.eval.eval_utils.seq_des_utils import score_samples

        merged_cfg = _merge_sampling_cfg(
            self.sampling_cfg,
            batch_size=batch_size,
            num_workers=num_workers,
            sampling_overrides=sampling_overrides,
        )
        return score_samples(
            model=self.model,
            data_cfg=self.data_cfg,
            sampling_cfg=merged_cfg,
            pdb_paths=pdb_paths,
            device=self.device,
        )

    def score_ensemble(
        self,
        pdb_to_conformers: dict[str, list[str]],
        *,
        num_workers: int | None = None,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> dict[str, list]:
        """Score native sequences against an ensemble of conformers.

        Args:
            pdb_to_conformers: Maps PDB name to list of conformer file paths.
            num_workers: Data loading workers (default: 2).
            sampling_overrides: Advanced overrides for sampling config.

        Returns:
            Dict with keys ``"example_id"``, ``"seq"``, ``"U"``, ``"U_i"``.
        """
        from caliby.eval.eval_utils.seq_des_utils import score_samples_ensemble

        merged_cfg = _merge_sampling_cfg(
            self.sampling_cfg,
            num_workers=num_workers,
            sampling_overrides=sampling_overrides,
        )
        return score_samples_ensemble(
            model=self.model,
            data_cfg=self.data_cfg,
            sampling_cfg=merged_cfg,
            pdb_to_conformers=pdb_to_conformers,
            device=self.device,
        )

    def self_consistency_eval(
        self,
        designed_pdbs: list[str],
        *,
        out_dir: str | None = None,
        num_models: int = 5,
        sample_models: bool = True,
        num_recycles: int = 3,
        use_multimer: bool = False,
    ) -> dict[str, dict[str, float]]:
        """Run AF2 self-consistency evaluation on designed structures.

        Folds each designed sequence with AlphaFold2 and compares the
        predicted structure to the designed backbone.

        Requires the ``af2`` extra: ``pip install 'caliby[af2]'``.

        Args:
            designed_pdbs: Paths to designed PDB/CIF files (e.g. from
                :meth:`sample` results ``"out_pdb"``).
            out_dir: Directory for AF2 predictions and metrics. If None,
                uses a temp dir.
            num_models: Number of AF2 models to sample (best by pLDDT is kept).
            sample_models: Randomly sample from the 5 AF2 models.
            num_recycles: Number of AF2 recycling iterations.
            use_multimer: Use AF2-Multimer.

        Returns:
            Dict mapping ``example_id`` to a dict with keys
            ``"sc_ca_rmsd"``, ``"avg_ca_plddt"``, ``"tmalign_score"``.
        """
        from caliby.eval.eval_utils import eval_metrics
        from caliby.eval.eval_utils.folding_utils import get_struct_pred_model

        if out_dir is None:
            out_dir = tempfile.mkdtemp(prefix="caliby_sc_")

        # Build a config dict that get_struct_pred_model expects.
        from omegaconf import OmegaConf

        struct_pred_cfg = OmegaConf.create({
            "model_name": "af2",
            "base_cfg": str(_PACKAGE_ROOT / "configs" / "struct_pred" / "struct_pred_base.yaml"),
            "af2": {
                "data_dir": None,
                "num_models": num_models,
                "sample_models": sample_models,
                "num_recycles": num_recycles,
                "save_best": True,
                "use_multimer": use_multimer,
            },
        })
        struct_pred_model = get_struct_pred_model(struct_pred_cfg, device=self.device)

        return eval_metrics.run_self_consistency_eval(designed_pdbs, struct_pred_model, out_dir=out_dir)

    def sidechain_pack(
        self,
        pdb_paths: list[str],
        *,
        out_dir: str | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> dict[str, list]:
        """Pack sidechains onto the given backbone structures.

        Args:
            pdb_paths: Paths to PDB or CIF files.
            out_dir: Directory for output CIF files. If None, uses a temp dir.
            batch_size: Batch size for packing (default: 4).
            num_workers: Data loading workers (default: 2).
            sampling_overrides: Advanced overrides for sampling config.

        Returns:
            Dict with keys ``"example_id"``, ``"out_pdb"``.
        """
        from caliby.eval.eval_utils.seq_des_utils import run_sidechain_packing

        merged_cfg = _merge_sampling_cfg(
            self.sampling_cfg,
            batch_size=batch_size,
            num_workers=num_workers,
            sampling_overrides=sampling_overrides,
        )
        if out_dir is None:
            out_dir = tempfile.mkdtemp(prefix="caliby_")

        return run_sidechain_packing(
            model=self.model,
            data_cfg=self.data_cfg,
            sampling_cfg=merged_cfg,
            pdb_paths=pdb_paths,
            device=self.device,
            out_dir=out_dir,
        )


def load_model(
    model_name: str = "caliby",
    device: str | None = None,
    sampling_cfg_path: str | None = None,
) -> CalibyModel:
    """Load a Caliby model for reuse across multiple calls.

    Args:
        model_name: Registered model name (``"caliby"``, ``"soluble_caliby"``,
            etc.) or path to a ``.ckpt`` file.
        device: Torch device string. Defaults to ``"cuda"`` if available.
        sampling_cfg_path: Path to a custom sampling YAML config. If None,
            uses the built-in defaults.

    Returns:
        A :class:`CalibyModel` ready for sampling, scoring, or packing.
    """
    import hydra
    import torch
    from omegaconf import OmegaConf

    from caliby.checkpoint_utils import get_cfg_from_ckpt, load_from_checkpoint
    from caliby.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
    from caliby.weights import resolve_ckpt_path

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = resolve_ckpt_path(model_name)
    lit_sd_model = load_from_checkpoint(LitSeqDenoiser, ckpt_path).eval()
    model_cfg, _ = get_cfg_from_ckpt(ckpt_path)
    data_cfg = hydra.utils.instantiate(model_cfg.data)

    cfg_path = sampling_cfg_path or str(_DEFAULT_SAMPLING_CFG_PATH)
    sampling_cfg = OmegaConf.load(cfg_path)

    sd_model = lit_sd_model.model
    sd_model.to(device)
    torch.set_grad_enabled(False)

    return CalibyModel(model=sd_model, data_cfg=data_cfg, sampling_cfg=sampling_cfg, device=device)


class _XCleanDataset:  # XATTEMPT L-B "loader": map-style dataset whose items are the cleaned-file paths written by the stock clean_pdb()
    def __init__(self, pdb_paths: list[str], out_dir: str):
        self._pdb_paths, self._out_dir = pdb_paths, out_dir

    def __len__(self) -> int:
        return len(self._pdb_paths)

    def __getitem__(self, idx: int) -> str:
        from caliby.data.preprocessing.atomworks.clean_pdbs import clean_pdb

        return clean_pdb(self._pdb_paths[idx], self._out_dir)


def _x_identity(batch):
    return batch


def clean_pdbs(
    pdb_paths: list[str],
    *,
    out_dir: str | None = None,
    num_workers: int = 1,
) -> list[str]:
    """Clean PDB/CIF files and write sanitized mmCIF copies.

    This is most useful before sequence design or ensemble generation when
    working with structures that may contain blank chain IDs, unresolved
    atoms, or residue names unsupported by downstream tools.

    Args:
        pdb_paths: Paths to input PDB or CIF files.
        out_dir: Directory for cleaned mmCIF files. If None, uses a temp dir.
        num_workers: Number of parallel workers to use.

    Returns:
        List of cleaned mmCIF file paths in the same order as ``pdb_paths``.
    """
    from joblib import Parallel, delayed

    from caliby.data.preprocessing.atomworks.clean_pdbs import clean_pdb

    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="caliby_clean_")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if num_workers == 1:
        return [clean_pdb(pdb_path, out_dir) for pdb_path in pdb_paths]

    # XATTEMPT add-on lever L-B (env CALIBY_X_CLEAN=loader): run the stock clean_pdb() per file inside `num_workers` torch DataLoader
    # worker processes -- the same mechanism (fork of this already-imported process, index-ordered results, no shuffling) that
    # caliby's own InferenceDataLoader uses for featurisation in this very process; each worker pays the lazy CCD load once, in
    # parallel.  Outputs are the files clean_pdb writes, byte-identical to the serial path except biotite's `_entry.date/time`;
    # any failure -> falls back to the stock serial loop (identical outputs) with a note on stderr.
    if os.environ.get("CALIBY_X_CLEAN", "") == "loader":
        import sys as _sys

        try:
            from torch.utils.data import DataLoader as _DL

            _ds = _XCleanDataset(list(map(str, pdb_paths)), str(out_dir))
            _paths = [b[0] for b in _DL(_ds, batch_size=1, shuffle=False, num_workers=min(num_workers, len(pdb_paths)), collate_fn=_x_identity)]
            if len(_paths) == len(pdb_paths) and all(isinstance(p, str) and os.path.exists(p) for p in _paths):
                return _paths
            print("[CALIBY_X_CLEAN=loader] incomplete result -> serial fallback", file=_sys.stderr)
        except Exception as _exc:  # noqa: BLE001
            if is_oom(_exc): raise
            print(f"[CALIBY_X_CLEAN=loader] failed ({_exc!r}) -> serial fallback", file=_sys.stderr)
        return [clean_pdb(pdb_path, out_dir) for pdb_path in pdb_paths]

    parallel = Parallel(n_jobs=num_workers)
    return list(parallel(delayed(clean_pdb)(pdb_path, out_dir) for pdb_path in pdb_paths))


def make_constraints(constraints: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Build a positional constraint DataFrame from a dict.

    Args:
        constraints: Maps ``pdb_key`` to a dict of constraint columns.
            Valid columns: ``"fixed_pos_seq"``, ``"fixed_pos_scn"``,
            ``"fixed_pos_override_seq"``, ``"pos_restrict_aatype"``,
            ``"symmetry_pos"``.

    Example::

        make_constraints({
            "2fyzA": {"fixed_pos_seq": "A1-50", "pos_restrict_aatype": "A60:AVG"},
        })

    Returns:
        A DataFrame suitable for the ``pos_constraint_df`` argument.
    """
    rows = [{"pdb_key": pdb_key, **cols} for pdb_key, cols in constraints.items()]
    return pd.DataFrame(rows)


def make_ensemble_constraints(
    constraints: dict[str, dict[str, str]],
    pdb_to_conformers: dict[str, list[str]],
) -> pd.DataFrame:
    """Build a constraint DataFrame expanded across ensemble conformers.

    Convenience wrapper around :func:`make_constraints` that replicates
    each PDB's constraints for every conformer in its ensemble.

    Args:
        constraints: Maps ``pdb_key`` to a dict of constraint columns
            (same format as :func:`make_constraints`).
        pdb_to_conformers: Maps PDB name to list of conformer file paths
            (same mapping passed to :meth:`CalibyModel.ensemble_sample`).

    Example::

        make_ensemble_constraints(
            {"8sot": {"fixed_pos_seq": "A1-10"}},
            {"8sot": ["8sot.cif", "conf_0.pdb", "conf_1.pdb"]},
        )

    Returns:
        A DataFrame with one row per conformer, suitable for the
        ``pos_constraint_df`` argument of :meth:`CalibyModel.ensemble_sample`.
    """
    base_df = make_constraints(constraints)

    from caliby.eval.eval_utils.eval_setup_utils import get_ensemble_constraint_df

    return get_ensemble_constraint_df(base_df, pdb_to_conformers)


# ---------------------------------------------------------------------------
# Protpardelle ensemble generation
# ---------------------------------------------------------------------------


def generate_ensembles(
    pdb_paths: list[str],
    *,
    out_dir: str,
    num_samples_per_pdb: int = 32,
    batch_size: int = 8,
    model_params_path: str | None = None,
    sampling_yaml_path: str | None = None,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Generate structural ensembles using Protpardelle-1c partial diffusion.

    Args:
        pdb_paths: Paths to input PDB/CIF files.
        out_dir: Output directory for generated conformers.
        num_samples_per_pdb: Number of conformers to generate per structure.
        batch_size: Batch size for Protpardelle sampling.
        model_params_path: Directory for model weights. Defaults to
            ``$MODEL_PARAMS_DIR`` or ``"model_params"``.
        sampling_yaml_path: Path to Protpardelle sampling YAML config.
            If None, uses the built-in partial diffusion config.
        seed: Random seed.

    Returns:
        Dict mapping PDB stem to list of generated conformer file paths.
    """
    import importlib
    import shutil

    import lightning as L
    import torch
    from hydra.core.global_hydra import GlobalHydra
    from tqdm import tqdm

    from caliby.weights import ensure_dir

    if model_params_path is None:
        model_params_path = os.environ.get("MODEL_PARAMS_DIR", "model_params")
    if sampling_yaml_path is None:
        sampling_yaml_path = str(
            _PACKAGE_ROOT / "configs" / "protpardelle-1c" / "multichain_backbone_partial_diffusion.yaml"
        )

    # Set up protpardelle env vars and import.
    ensure_dir(f"{model_params_path}/proteinmpnn")
    ensure_dir(f"{model_params_path}/protpardelle-1c")
    os.environ["PROTPARDELLE_OUTPUT_DIR"] = f"{out_dir}/protpardelle_outputs_temp"
    os.environ["FOLDSEEK_BIN"] = "."
    os.environ["ESMFOLD_PATH"] = "."
    os.environ["PROTEINMPNN_WEIGHTS"] = f"{model_params_path}/proteinmpnn"
    os.environ["PROTPARDELLE_MODEL_PARAMS"] = f"{model_params_path}/protpardelle-1c"
    protpardelle_sample = importlib.import_module("protpardelle.sample")
    GlobalHydra.instance().clear()
    _x_pp_model_cache_install(protpardelle_sample)  # XATTEMPT L-G (no-op unless CALIBY_X_PP_CACHE=1)

    L.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pdb_to_conformers: dict[str, list[str]] = {}

    for pdb_file in tqdm(pdb_paths, desc="Generating ensembles"):
        all_save_dirs = protpardelle_sample.sample(
            sampling_yaml_path=Path(sampling_yaml_path),
            motif_pdb=Path(pdb_file),
            batch_size=batch_size,
            num_samples=num_samples_per_pdb,
            num_mpnn_seqs=0,
        )

        # Move results to our output directory.
        pdb_stem = Path(pdb_file).stem
        conformer_paths = []
        for save_dir in all_save_dirs:
            dest_parent = Path(out_dir) / save_dir.parent.name
            dest_parent.mkdir(parents=True, exist_ok=True)
            dest_path = dest_parent / save_dir.name
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.move(str(save_dir), str(dest_parent))

            # Collect generated PDB files.
            for f in sorted(dest_path.glob("sample_*.pdb")):
                conformer_paths.append(str(f))

        pdb_to_conformers[pdb_stem] = conformer_paths

    return pdb_to_conformers


# XATTEMPT add-on lever L-G (env CALIBY_X_PP_CACHE=1): protpardelle.sample.sample() re-loads the Protpardelle-1c checkpoint from disk
# (torch.load + Protpardelle(config) + load_state_dict) on EVERY call, and caliby.generate_ensembles calls it once per input PDB.
# With the switch on, load_model(config_path, checkpoint_path[, device]) as seen by protpardelle.sample is memoised per
# (config_path, checkpoint_path, device): the same weights on the same device give the identical module object in eval mode
# (Protpardelle.sample never mutates parameters or buffers), so outputs are unchanged; ~1.4 s per backbone saved.
_X_PP_MODELS: dict = {}


def _x_pp_model_cache_install(protpardelle_sample) -> None:
    if os.environ.get("CALIBY_X_PP_CACHE", "0") in ("", "0"):
        return
    if getattr(protpardelle_sample.load_model, "_x_cached", False):
        return
    _orig = protpardelle_sample.load_model

    def load_model_cached(config_path, checkpoint_path, device=None):
        key = (str(config_path), str(checkpoint_path), str(device))
        m = _X_PP_MODELS.get(key)
        if m is None:
            m = _orig(config_path, checkpoint_path, device) if device is not None else _orig(config_path, checkpoint_path)
            _X_PP_MODELS[key] = m
        m.eval()
        return m

    load_model_cached._x_cached = True
    protpardelle_sample.load_model = load_model_cached


# ---------------------------------------------------------------------------
# Module-level convenience functions (load model + call + return)
# ---------------------------------------------------------------------------


def caliby_sample(
    pdb_paths: list[str],
    *,
    model_name: str = "caliby",
    device: str | None = None,
    out_dir: str | None = None,
    num_seqs_per_pdb: int | None = None,
    batch_size: int | None = None,
    omit_aas: list[str] | None = None,
    num_workers: int | None = None,
    temperature: float | None = None,
    verbose: bool | None = None,
    pos_constraint_df: pd.DataFrame | None = None,
    sampling_overrides: dict[str, Any] | None = None,
) -> dict[str, list]:
    """One-shot sequence design: load model, sample, return results.

    For repeated calls, prefer :func:`load_model` to avoid reloading weights.
    See :meth:`CalibyModel.sample` for argument details.
    """
    model = load_model(model_name=model_name, device=device)
    return model.sample(
        pdb_paths,
        out_dir=out_dir,
        num_seqs_per_pdb=num_seqs_per_pdb,
        batch_size=batch_size,
        omit_aas=omit_aas,
        num_workers=num_workers,
        temperature=temperature,
        verbose=verbose,
        pos_constraint_df=pos_constraint_df,
        sampling_overrides=sampling_overrides,
    )


def caliby_ensemble_sample(
    pdb_to_conformers: dict[str, list[str]],
    *,
    model_name: str = "caliby",
    device: str | None = None,
    out_dir: str | None = None,
    num_seqs_per_pdb: int | None = None,
    batch_size: int | None = None,
    omit_aas: list[str] | None = None,
    num_workers: int | None = None,
    temperature: float | None = None,
    verbose: bool | None = None,
    pos_constraint_df: pd.DataFrame | None = None,
    use_primary_res_type: bool = True,
    sampling_overrides: dict[str, Any] | None = None,
) -> dict[str, list]:
    """One-shot ensemble sequence design.

    See :meth:`CalibyModel.ensemble_sample` for argument details.
    """
    model = load_model(model_name=model_name, device=device)
    return model.ensemble_sample(
        pdb_to_conformers,
        out_dir=out_dir,
        num_seqs_per_pdb=num_seqs_per_pdb,
        batch_size=batch_size,
        omit_aas=omit_aas,
        num_workers=num_workers,
        temperature=temperature,
        verbose=verbose,
        pos_constraint_df=pos_constraint_df,
        use_primary_res_type=use_primary_res_type,
        sampling_overrides=sampling_overrides,
    )


def caliby_score(
    pdb_paths: list[str],
    *,
    model_name: str = "caliby",
    device: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    sampling_overrides: dict[str, Any] | None = None,
) -> dict[str, list]:
    """One-shot scoring.

    See :meth:`CalibyModel.score` for argument details.
    """
    model = load_model(model_name=model_name, device=device)
    return model.score(
        pdb_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        sampling_overrides=sampling_overrides,
    )


def caliby_score_ensemble(
    pdb_to_conformers: dict[str, list[str]],
    *,
    model_name: str = "caliby",
    device: str | None = None,
    num_workers: int | None = None,
    sampling_overrides: dict[str, Any] | None = None,
) -> dict[str, list]:
    """One-shot ensemble scoring.

    See :meth:`CalibyModel.score_ensemble` for argument details.
    """
    model = load_model(model_name=model_name, device=device)
    return model.score_ensemble(
        pdb_to_conformers,
        num_workers=num_workers,
        sampling_overrides=sampling_overrides,
    )


def caliby_sidechain_pack(
    pdb_paths: list[str],
    *,
    model_name: str = "caliby",
    device: str | None = None,
    out_dir: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    sampling_overrides: dict[str, Any] | None = None,
) -> dict[str, list]:
    """One-shot sidechain packing.

    See :meth:`CalibyModel.sidechain_pack` for argument details.
    """
    model = load_model(model_name=model_name, device=device)
    return model.sidechain_pack(
        pdb_paths,
        out_dir=out_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        sampling_overrides=sampling_overrides,
    )
