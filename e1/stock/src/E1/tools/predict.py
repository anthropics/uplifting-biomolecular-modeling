import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import torch
from safetensors.torch import save_file

from .. import dist
from ..io import read_fasta_sequences
from ..modeling import E1ForMaskedLM
from ..predictor import E1Prediction, E1Predictor

logger = logging.getLogger(__name__)


def save_prediction(prediction: E1Prediction, output_dir: str) -> None:
    _id = prediction["id"]
    context_id = prediction["context_id"]

    tensors = {}
    if "logits" in prediction:
        tensors["logits"] = prediction["logits"]
    if "token_embeddings" in prediction:
        tensors["token_embeddings"] = prediction["token_embeddings"]
    if "mean_token_embeddings" in prediction:
        tensors["mean_token_embeddings"] = prediction["mean_token_embeddings"]

    filename = f"{context_id}+{_id}.safetensors" if context_id else f"{_id}.safetensors"
    save_file(tensors, os.path.join(output_dir, filename), metadata={"id": _id, "context_id": context_id})


@click.command()
@click.option("--model-name", type=str, required=True, help="Name of the model to use")
@click.option("--input-path", type=str, required=True, help="Path to the input fasta file")
@click.option("--output-dir", type=str, default="predictions", help="Directory to save the predictions")
@click.option("--context-path", type=str, help="Path to the context fasta file")
@click.option(
    "--max-batch-tokens", type=int, default=65536, help="Maximum number of tokens to batch in a single forward pass"
)
@click.option(
    "--save-masked-positions-only",
    type=bool,
    default=False,
    help="Whether to save logits/embeddings for only the masked positions in each sequence",
)
@click.option(
    "--fields-to-save",
    type=list[str],
    default=["logits", "token_embeddings", "mean_token_embeddings"],
    help="Fields to save",
)
def predict(
    model_name: str,
    input_path: str,
    max_batch_tokens: int,
    save_masked_positions_only: bool,
    fields_to_save: list[str],
    context_path: str | None,
    output_dir: str,
) -> None:
    # Model is loaded in fp32, the actual predictions is made under torch.autocast(..., torch.bfloat16)
    # context manager
    model = E1ForMaskedLM.from_pretrained(model_name, device_map=dist.get_device(), dtype=torch.float)
    model.eval()

    context_seqs: dict[str, str] | None = None
    if context_path is not None:
        context_seqs = read_fasta_sequences(context_path)

    input_sequences = read_fasta_sequences(input_path)
    sequence_ids, sequences = zip(*input_sequences.items())

    predictor = E1Predictor(
        model,
        max_batch_tokens=max_batch_tokens,
        save_masked_positions_only=save_masked_positions_only,
        fields_to_save=fields_to_save,
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for prediction in predictor.predict(
            sequences=list(sequences), sequence_ids=list(sequence_ids), context_seqs=context_seqs
        ):
            futures.append(executor.submit(save_prediction, prediction, output_dir))

        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    dist.setup_dist()
    try:
        predict()
    finally:
        dist.destroy_process_group()
