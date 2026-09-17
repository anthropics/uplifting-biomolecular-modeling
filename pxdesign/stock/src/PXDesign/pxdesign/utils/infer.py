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

import hashlib
import logging
import os
import sys
import urllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from ml_collections.config_dict import ConfigDict
from protenix.config import parse_configs
from protenix.data.parser import DistillationMMCIFParser
from protenix.utils.file_io import dump_gzip_pickle
from pxdbench.pxd_configs.eval import eval_configs

from pxdesign.configs.configs_base import configs as configs_base
from pxdesign.configs.configs_data import data_configs
from pxdesign.configs.configs_infer import inference_configs
from pxdesign.data.utils import pdb_to_cif

URL = {
    "pxdesign_v0.1.0": "https://pxdesign.tos-cn-beijing.volces.com/release_model/pxdesign_v0.1.0.pt",
    "protenix_base_default_v0.5.0": "https://pxdesign.tos-cn-beijing.volces.com/release_model/protenix_base_default_v0.5.0.pt",
    "protenix_mini_default_v0.5.0": "https://pxdesign.tos-cn-beijing.volces.com/release_model/protenix_mini_default_v0.5.0.pt",
    "protenix_mini_tmpl_v0.5.0": "https://pxdesign.tos-cn-beijing.volces.com/release_model/protenix_mini_tmpl_v0.5.0.pt",
    "ccd_components_file": "https://pxdesign.tos-cn-beijing.volces.com/release_data/components.v20240608.cif",
    "ccd_components_rdkit_mol_file": "https://pxdesign.tos-cn-beijing.volces.com/release_data/components.v20240608.cif.rdkit_mol.pkl",
    "pdb_cluster_file": "https://pxdesign.tos-cn-beijing.volces.com/release_data/clusters-by-entity-40.txt",
}

ALIASES = {
    "N_sample": "sample_diffusion.N_sample",
    "N_step": "sample_diffusion.N_step",
    "eta_type": "sample_diffusion.eta_schedule.type",
    "eta_min": "sample_diffusion.eta_schedule.min",
    "eta_max": "sample_diffusion.eta_schedule.max",
    "gamma0": "sample_diffusion.gamma0",
    "gamma_min": "sample_diffusion.gamma_min",
    "sample_diffusion_chunk_size": "infer_setting.sample_diffusion_chunk_size",
}

logger = logging.getLogger(__name__)


def download_inference_cache(configs) -> None:
    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        bar_length = 30
        filled_length = int(bar_length * percent // 100)
        bar = "=" * filled_length + "-" * (bar_length - filled_length)

        status = f"\r[{bar}] {percent:.1f}%"
        print(status, end="", flush=True)

        if downloaded >= total_size:
            print()

    def download_from_url(tos_url, checkpoint_path, check_weight=True):
        urllib.request.urlretrieve(
            tos_url, checkpoint_path, reporthook=progress_callback
        )
        if check_weight:
            try:
                ckpt = torch.load(checkpoint_path)
                del ckpt
            except:
                os.remove(checkpoint_path)
                raise RuntimeError(
                    "Download model checkpoint failed, please download by yourself with "
                    f"wget {tos_url} -O {checkpoint_path}"
                )

    for cache_name in (
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
        "pdb_cluster_file",
    ):
        cur_cache_fpath = configs["data"][cache_name]
        if not os.path.exists(cur_cache_fpath):
            os.makedirs(os.path.dirname(cur_cache_fpath), exist_ok=True)
            tos_url = URL[cache_name]
            assert os.path.basename(tos_url) == os.path.basename(cur_cache_fpath), (
                f"{cache_name} file name is incorrect, `{tos_url}` and "
                f"`{cur_cache_fpath}`. Please check and try again."
            )
            logger.info(
                f"Downloading data cache from\n {tos_url}... to {cur_cache_fpath}"
            )
            download_from_url(tos_url, cur_cache_fpath, check_weight=False)

    checkpoint_path = os.path.join(
        configs.load_checkpoint_dir, f"{configs.model_name}.pt"
    )
    if not os.path.exists(checkpoint_path):
        os.makedirs(configs.load_checkpoint_dir, exist_ok=True)
        tos_url = URL[configs.model_name]
        logger.info(
            f"Downloading model checkpoint from\n {tos_url}... to {checkpoint_path}"
        )
        download_from_url(tos_url, checkpoint_path)

    # download protenix checkpoints
    for model_name in [
        "protenix_base_default_v0.5.0",
        "protenix_mini_default_v0.5.0",
        "protenix_mini_tmpl_v0.5.0",
    ]:
        checkpoint_path = os.path.join(configs.load_checkpoint_dir, f"{model_name}.pt")
        if not os.path.exists(checkpoint_path):
            tos_url = URL[model_name]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}... to {checkpoint_path}"
            )
            download_from_url(tos_url, checkpoint_path)

    # set checkpoint dir for ptx tools in PXDesignBench
    if hasattr(configs, "eval"):
        configs.eval.binder.tools.ptx.load_checkpoint_dir = configs.load_checkpoint_dir
        configs.eval.binder.tools.ptx_mini.load_checkpoint_dir = (
            configs.load_checkpoint_dir
        )


def remap_arg_key(key: str) -> str:
    if key.startswith("--"):
        name = key[2:]
        mapped = ALIASES.get(name, name)
        return "--" + mapped
    return key


def parse_sys_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    remapped = []

    i = 0
    while i < len(argv):
        k = argv[i]
        # if k starts with "--", check whether it matches alias
        if k.startswith("--") and i + 1 < len(argv):
            remapped.append(remap_arg_key(k))
            remapped.append(argv[i + 1])
            i += 2
        else:
            remapped.append(k)
            i += 1

    return " ".join(remapped)


def get_configs(argv=None) -> ConfigDict:
    configs = {
        **configs_base,
        **{"data": data_configs},
        **inference_configs,
        **{"eval": eval_configs},
    }
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(argv),
        fill_required_with_null=True,
    )
    return configs


class DisableLogging:
    def __enter__(self):
        logging.disable(logging.WARNING)

    def __exit__(self, exc_type, exc, tb):
        logging.disable(logging.NOTSET)


# -------------------------
# Handling PDB input
# -------------------------


def parse_ranges(range_str: str) -> list[tuple[int, int]]:
    """
    Parse "1-30,40-50,66" -> [(1,30),(40,50),(66,66)]
    """
    ranges: list[tuple[int, int]] = []
    for part in range_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            ranges.append((int(a), int(b)))
        else:
            x = int(part)
            ranges.append((x, x))
    return ranges


def format_ranges(ints: Iterable[int]) -> str:
    """
    Compress sorted integers into "a-b,c,d-e".
    """
    xs = sorted(set(int(x) for x in ints))
    if not xs:
        return ""
    out: list[str] = []
    s = e = xs[0]
    for x in xs[1:]:
        if x == e + 1:
            e = x
        else:
            out.append(f"{s}-{e}" if s != e else f"{s}")
            s = e = x
    out.append(f"{s}-{e}" if s != e else f"{s}")
    return ",".join(out)


# -------------------------
# Chain mapping
# -------------------------


def build_chain_mapping(
    old_ids: Iterable[str],
    new_ids: Iterable[str],
    *,
    keep_chains: Iterable[str] | None = None,
    err_hint: str = "Please consider using a CIF structure file in your JSON file.",
) -> dict[str, str]:
    """
    Build mapping old_id -> new_id with consistency check.

    keep_chains:
      - None: keep all chains
      - otherwise: only build mapping for chains in keep_chains
    """
    old_ids = list(old_ids)
    new_ids = list(new_ids)
    if len(old_ids) != len(new_ids):
        raise ValueError("old_ids and new_ids must have the same length.")

    keep = set(keep_chains) if keep_chains is not None else None

    mapping: dict[str, str] = {}
    for old, new in zip(old_ids, new_ids):
        if keep is not None and old not in keep:
            continue
        if old not in mapping:
            mapping[old] = new
        elif mapping[old] != new:
            raise ValueError(
                f"Inconsistent mapping: chain '{old}' maps to both "
                f"'{mapping[old]}' and '{new}'. It will raise ambiguity. {err_hint}"
            )
    return mapping


# -------------------------
# Residue mapping (res_id <-> auth_res_id)
# -------------------------


@dataclass(frozen=True)
class ResidueMaps:
    # (chain_id, res_id) -> (auth_asym_id, auth_res_id)
    resid2auth: dict[tuple[str, int], tuple[str, int]]
    # (auth_asym_id, auth_res_id) -> (chain_id, res_id)
    auth2resid: dict[tuple[str, int], tuple[str, int]]


def build_residue_maps(
    atom_array,
    *,
    strict_bijective: bool = True,
    err_hint: str = "Please consider using a CIF structure file in your JSON file.",
) -> ResidueMaps:
    """
    Build residue-level mapping with uniqueness checks.

    Ensures:
      - each (chain_id, res_id) maps to a single (auth_asym_id, auth_res_id)
      - (optional) bijection: each (auth_asym_id, auth_res_id) maps back to a single (chain_id, res_id)
    """
    chain_id = atom_array.chain_id
    res_id = atom_array.res_id
    auth_asym_id = atom_array.auth_asym_id
    auth_res_id = atom_array.auth_res_id

    resid2auth: dict[tuple[str, int], tuple[str, int]] = {}
    auth2resid: dict[tuple[str, int], tuple[str, int]] = {}

    for c, r, ac, ar in zip(chain_id, res_id, auth_asym_id, auth_res_id):
        key = (str(c), int(r))
        val = (str(ac), int(ar))

        if key in resid2auth and resid2auth[key] != val:
            raise ValueError(
                "Non-unique mapping detected: same (chain_id, res_id) maps to multiple "
                f"(auth_asym_id, auth_res_id).\n  key={key}\n  first={resid2auth[key]}\n  new={val}\n"
                f"{err_hint}"
            )
        resid2auth.setdefault(key, val)

        if strict_bijective:
            if val in auth2resid and auth2resid[val] != key:
                raise ValueError(
                    "Non-unique mapping detected: same (auth_asym_id, auth_res_id) maps to multiple "
                    f"(chain_id, res_id).\n  val={val}\n  first={auth2resid[val]}\n  new={key}\n"
                    f"{err_hint}"
                )
            auth2resid.setdefault(val, key)

    return ResidueMaps(resid2auth=resid2auth, auth2resid=auth2resid)


# -------------------------
# Converters: crop / hotspot
# -------------------------


def convert_crop_auth_to_new(
    crop_dict: dict[str, str],
    residue_maps: ResidueMaps,
    *,
    strict_mapping: bool = True,
) -> dict[str, str]:
    result: dict[str, list[int]] = defaultdict(list)

    for auth_chain, range_str in crop_dict.items():
        for start, end in parse_ranges(range_str):
            for auth_r in range(start, end + 1):
                key = (auth_chain, int(auth_r))
                if key not in residue_maps.auth2resid:
                    if strict_mapping:
                        raise KeyError(
                            f"Requested auth residue not found in atom_array: {key}"
                        )
                    else:
                        continue
                new_c, new_r = residue_maps.auth2resid[key]
                result[new_c].append(new_r)

    return {new_c: format_ranges(rs) for new_c, rs in result.items() if rs}


def convert_hotspot_auth_to_new(
    hotspot_dict: dict[str, list[int]],
    residue_maps: ResidueMaps,
    *,
    strict_mapping: bool = True,
) -> dict[str, list[int]]:
    """
    {chain_id: [11,22]} -> {auth_asym_id: [auth_res_id,...]} (sorted unique)
    """
    result: dict[str, list[int]] = defaultdict(list)

    for chain, res_list in hotspot_dict.items():
        for r in res_list:
            key = (chain, int(r))
            if key not in residue_maps.auth2resid:
                if strict_mapping:
                    raise KeyError(f"Hotspot residue not found in atom_array: {key}")
                else:
                    continue
            ac, ar = residue_maps.auth2resid[key]
            result[ac].append(ar)

    return {ac: sorted(set(ars)) for ac, ars in result.items() if ars}


# -------------------------
# Apply filter rewrite (chain_id / crop / msa / hotspot)
# -------------------------


def rewrite_input_dict_inplace(
    input_dict: dict,
    *,
    chain_mapping: dict[str, str],  # old chain_id -> new chain_id (e.g. PDB->CIF)
    residue_maps: ResidueMaps | None,
) -> None:
    """
    Rewrite cond_dict['filter'] in-place using chain_mapping and residue_maps.
    """
    cond_dict = input_dict["condition"]
    filt = cond_dict.get("filter", {})
    if filt:
        # chain_id list
        if "chain_id" in filt and filt["chain_id"]:
            filt["chain_id"] = [chain_mapping[c] for c in filt["chain_id"]]

        # crop dict: {chain_id: "ranges"}
        if "crop" in filt and filt["crop"]:
            if residue_maps is None:
                raise ValueError(
                    "filter.crop requires residue_maps (atom_array) to convert res_id -> auth_res_id."
                )
            filt["crop"] = convert_crop_auth_to_new(filt["crop"], residue_maps)
        cond_dict["filter"] = filt

    # msa dict: {chain_id: ...}
    if "msa" in cond_dict and cond_dict["msa"]:
        cond_dict["msa"] = {chain_mapping[k]: v for k, v in cond_dict["msa"].items()}

    input_dict["condition"] = cond_dict
    # hotspot dict: {chain_id: [res_ids]}
    if "hotspot" in input_dict and input_dict["hotspot"]:
        if residue_maps is None:
            raise ValueError(
                "filter.hotspot requires residue_maps (atom_array) to convert res_id -> auth_res_id."
            )
        input_dict["hotspot"] = convert_hotspot_auth_to_new(
            input_dict["hotspot"], residue_maps
        )


# -------------------------
# Main entry: convert_to_bioassembly_dict
# -------------------------


def convert_to_bioassembly_dict(input_dict: dict, out_dir: str | None = None):
    """
    Returns:
      - if input is already .pkl.gz: (str_file)  (kept as your original behavior)
      - else: (out_path, chain_mapping)
    """
    assert "condition" in input_dict, "input_dict must have 'condition' key"
    cond_dict = input_dict["condition"]
    str_file = cond_dict["structure_file"]
    if out_dir is None:
        out_dir = os.path.dirname(str_file)

    if str_file.endswith(".pkl.gz"):
        return str_file

    chain_mapping: dict[str, str] = {}
    residue_maps: ResidueMaps | None = None
    atom_array = None

    if str_file.endswith(".cif"):
        parser = DistillationMMCIFParser(str_file)
        d = parser.get_structure_dict()

    elif str_file.endswith(".pdb"):
        cif_file = os.path.join(out_dir, os.path.basename(str_file)[:-4] + ".cif")
        atom_array = pdb_to_cif(str_file, cif_file)

        filter_chains = cond_dict.get("filter", {}).get("chain_id", [])
        chain_mapping = build_chain_mapping(
            atom_array.auth_asym_id,
            atom_array.chain_id,
            keep_chains=filter_chains if filter_chains else None,
        )
        residue_maps = build_residue_maps(atom_array)

        rewrite_input_dict_inplace(
            input_dict,
            chain_mapping=chain_mapping,
            residue_maps=residue_maps,
        )
        parser = DistillationMMCIFParser(cif_file)
        d = parser.get_structure_dict()

    else:
        raise ValueError(f"Unsupported structure file! {str_file}")

    out_path = Path(out_dir) / f"{Path(str_file).stem}.pkl.gz"
    assert str(out_path).endswith(".pkl.gz"), "Bioassembly dict should end with .pkl.gz"
    dump_gzip_pickle(d, out_path)
    input_dict["condition"]["structure_file"] = str(out_path)

    return d


def configure_runtime_env(
    use_fast_ln: bool = False, use_deepspeed_evo: bool = False
) -> None:
    """
    Independent runtime knobs:
      - use_fast_ln -> LAYERNORM_TYPE
      - use_deepspeed_evo  -> DEEPSPEED_EVO (+ CUTLASS dependency)
    """

    # LayerNorm
    if use_fast_ln:
        os.environ["LAYERNORM_TYPE"] = "fast_layernorm"

    # DeepSpeed Evo: fully independent
    os.environ["DEEPSPEED_EVO"] = "true" if use_deepspeed_evo else "false"

    if not use_deepspeed_evo:
        return

    if "CUTLASS_PATH" in os.environ and os.environ["CUTLASS_PATH"]:
        cutlass_path = Path(os.environ["CUTLASS_PATH"]).expanduser()
    else:
        cutlass_path = Path.home() / "cutlass"
        os.environ["CUTLASS_PATH"] = str(cutlass_path)

    if not cutlass_path.is_dir():
        print("")
        print(f"[WARNING] CUTLASS not found at: {cutlass_path}")
        print(
            "  PXDesign uses DeepSpeed Evo kernels which require NVIDIA CUTLASS v3.5.1."
        )
        print("  To install:")
        print(
            '    git clone -b v3.5.1 https://github.com/NVIDIA/cutlass.git "$HOME/cutlass"'
        )
        print('    export CUTLASS_PATH="$HOME/cutlass"')
        print("")


def derive_seed(base_seed: int, rank: int = 0, digits: int = 6) -> int:
    mod = 10**digits
    msg = f"pxdesign|{base_seed}|{rank}".encode()
    h = hashlib.blake2b(msg, digest_size=8).digest()
    return int.from_bytes(h, "little") % mod
