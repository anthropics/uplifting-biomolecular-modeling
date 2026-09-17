import argparse
from pathlib import Path

import numpy as np
import torch
from eval import (
    add_benchmark_args,
    attention_pair_features,
    attention_pair_scores,
    run_benchmark,
)

from atlaslm.model import AtlasLM

EVOLUTIONARY_SCALE_WEIGHTS = {
    "esmc_300m": (
        "EvolutionaryScale/esmc-300m-2024-12",
        "data/weights/esmc_300m_2024_12_v0.pth",
    ),
    "esmc_600m": (
        "EvolutionaryScale/esmc-600m-2024-12",
        "data/weights/esmc_600m_2024_12_v0.pth",
    ),
    "esmc_6b": (
        "biohub/esmc-6b-2024-12",
        "data/weights/esmc_6b_2024_12_v0.pth",
    ),
}
MODEL_CONFIGS = {
    "esmc_300m": dict(d_model=960, n_heads=15, n_layers=30),
    "esmc_600m": dict(d_model=1152, n_heads=18, n_layers=36),
    "esmc_3b": dict(d_model=2304, n_heads=36, n_layers=48),
    "esmc_6b": dict(d_model=2560, n_heads=40, n_layers=80),
}


def get_model(model_name: str) -> AtlasLM:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported models: {', '.join(MODEL_CONFIGS)}"
        )
    return AtlasLM(**MODEL_CONFIGS[model_name]).eval()


def resolve_pretrained_path(model_name: str) -> Path:
    if model_name not in EVOLUTIONARY_SCALE_WEIGHTS:
        raise ValueError(
            f"No Hugging Face ESM-C weight is registered for {model_name}. "
            "Pass --model-path explicitly."
        )
    return (
        evolutionary_scale_data_root(model_name)
        / EVOLUTIONARY_SCALE_WEIGHTS[model_name][1]
    )


def evolutionary_scale_data_root(model_name: str) -> Path:
    from huggingface_hub import snapshot_download

    repo_id, _ = EVOLUTIONARY_SCALE_WEIGHTS[model_name]
    return Path(snapshot_download(repo_id=repo_id))


def load_model(
    path: str | Path,
    model_name: str,
    device: str | torch.device,
    dtype: torch.dtype,
) -> AtlasLM:
    device = torch.device(device)
    with torch.device("meta"):
        model = get_model(model_name).to(dtype)
    model = model.to_empty(device=device)
    state_dict = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if all(key.startswith("model.") for key in state_dict):
        state_dict = {key[len("model.") :]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    for module in model.modules():
        if hasattr(module, "init_buffers"):
            module.init_buffers(device=device)
    return model


def load_evolutionary_scale_model(
    model_name: str,
    device: str | torch.device,
    dtype: torch.dtype,
) -> AtlasLM:
    return load_model(
        resolve_pretrained_path(model_name),
        model_name=model_name,
        device=device,
        dtype=dtype,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ESM-C on the Rao2021 unsupervised contact benchmark."
    )
    parser.add_argument("--model-name", default="esmc_600m")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="Weight dtype after loading the model.",
    )
    add_benchmark_args(parser)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    if args.model_path is None:
        model = load_evolutionary_scale_model(
            args.model_name,
            device=args.device,
            dtype=dtype,
        )
    else:
        model = load_model(
            args.model_path,
            model_name=args.model_name,
            device=args.device,
            dtype=dtype,
        )

    def get_attentions(seq: str) -> torch.Tensor:
        out = model.embed_sequences([seq], return_attentions=True)
        if out.attentions is None:
            raise RuntimeError("Model did not return attentions")
        return torch.stack(out.attentions, dim=1)[0]

    def extract_attention_features(
        seq: str, i_idx: np.ndarray, j_idx: np.ndarray
    ) -> np.ndarray:
        return attention_pair_features(
            get_attentions(seq),
            len(seq),
            i_idx,
            j_idx,
            n_layers=model.n_layers,
            n_heads=model.n_heads,
        )

    def extract_attention_scores(
        seq: str,
        i_idx: np.ndarray,
        j_idx: np.ndarray,
        score_cache: dict,
    ) -> np.ndarray:
        return attention_pair_scores(
            get_attentions(seq),
            len(seq),
            i_idx,
            j_idx,
            n_layers=model.n_layers,
            n_heads=model.n_heads,
            score_cache=score_cache,
        )

    run_benchmark(
        extract_attention_features,
        root_path=args.root_path,
        n_bootstraps=args.n_bootstraps,
        n_train=args.n_train,
        extract_attention_scores=extract_attention_scores,
    )


if __name__ == "__main__":
    main()
