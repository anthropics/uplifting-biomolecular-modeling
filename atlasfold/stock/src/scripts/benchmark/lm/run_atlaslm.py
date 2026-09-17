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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AtlasLM on the Rao2021 unsupervised contact benchmark."
    )
    parser.add_argument("--model-name", default="atlaslm-3b")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    add_benchmark_args(parser)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    from atlaslm.model import AtlasLM
    from atlaslm.pretrained import MODEL_CONFIG_MAP, MODEL_CONFIGS, get_model_name

    source = args.model_path if args.model_path is not None else args.model_name
    config = None
    if args.model_path is not None:
        model_name = get_model_name(args.model_name)
        config = MODEL_CONFIGS[MODEL_CONFIG_MAP[model_name]]

    model = AtlasLM.from_pretrained(
        source,
        config=config,
        device=args.device,
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
