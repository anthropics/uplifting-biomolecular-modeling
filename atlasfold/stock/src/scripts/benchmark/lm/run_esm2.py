import argparse

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
        description="Evaluate ESM2 on the Rao2021 unsupervised contact benchmark."
    )
    parser.add_argument("--model-name", default="esm2_t30_150M_UR50D")
    parser.add_argument("--device", default="cuda")
    add_benchmark_args(parser)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    import esm

    model, alphabet = esm.pretrained.load_model_and_alphabet(args.model_name)
    model = model.eval().to(args.device)
    batch_converter = alphabet.get_batch_converter()

    def get_attentions(seq: str) -> torch.Tensor:
        _, _, batch_tokens = batch_converter([(0, seq)])
        esm_out = model.forward(batch_tokens.to(args.device), need_head_weights=True)
        return esm_out["attentions"][0]

    def extract_attention_features(
        seq: str, i_idx: np.ndarray, j_idx: np.ndarray
    ) -> np.ndarray:
        return attention_pair_features(
            get_attentions(seq),
            len(seq),
            i_idx,
            j_idx,
            n_layers=model.num_layers,
            n_heads=model.attention_heads,
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
            n_layers=model.num_layers,
            n_heads=model.attention_heads,
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
