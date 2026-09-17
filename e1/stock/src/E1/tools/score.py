import logging

import click
import polars as pl
import torch

from .. import dist
from ..io import read_fasta_sequences
from ..modeling import E1ForMaskedLM
from ..scorer import E1Scorer, EncoderScoreMethod

logger = logging.getLogger(__name__)


@click.command()
@click.option("--model-name", type=str, required=True, help="Name of the model to use")
@click.option("--mutants-path", type=str, required=True, help="Path to the input fasta file")
@click.option("--parent-path", type=str, required=True, help="Path to the parent fasta file")
@click.option("--output-path", type=str, required=True, help="Path to the output file")
@click.option("--context-path", type=str, help="Path to the context fasta file")
@click.option(
    "--max-batch-tokens", type=int, default=65536, help="Maximum number of tokens to batch in a single forward pass"
)
@click.option(
    "--scoring-method",
    type=click.Choice(EncoderScoreMethod, case_sensitive=False),
    default=EncoderScoreMethod.MASKED_MARGINAL,
    help="Method to use for scoring",
)
@click.option("--context-reduction", type=str, default="mean", help="How to reduce the context scores")
def score(
    model_name: str,
    mutants_path: str,
    parent_path: str,
    output_path: str,
    context_path: str | None,
    max_batch_tokens: int,
    scoring_method: EncoderScoreMethod,
    context_reduction: str,
) -> None:
    # Model is loaded in fp32, the actual predictions is made under torch.autocast(..., torch.bfloat16)
    # context manager
    model = E1ForMaskedLM.from_pretrained(model_name, dtype=torch.float).to(dist.get_device())
    model.eval()

    context_seqs: dict[str, str] | None = None
    if context_path is not None:
        context_seqs = read_fasta_sequences(context_path)

    parent_seqs = list(read_fasta_sequences(parent_path).items())
    assert len(parent_seqs) == 1, "Only one parent sequence is supported for E1 models"
    parent_sequence = parent_seqs[0][1]

    mutated_seqs = list(read_fasta_sequences(mutants_path).items())
    assert len(mutated_seqs) > 0, "No mutated sequences found in the input file"
    mutated_sequences = [seq for _, seq in mutated_seqs]
    mutated_sequence_ids = [seq_id for seq_id, _ in mutated_seqs]

    scorer = E1Scorer(model, method=scoring_method, max_batch_tokens=max_batch_tokens)
    scores = scorer.score(
        parent_sequence=parent_sequence,
        sequences=mutated_sequences,
        sequence_ids=mutated_sequence_ids,
        context_seqs=context_seqs,
        context_reduction=context_reduction,
    )

    if dist.get_rank() == 0:
        scores_df = pl.from_dicts(scores)
        scores_df.write_csv(output_path)

    dist.barrier()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    dist.setup_dist()
    try:
        score()
    finally:
        dist.destroy_process_group()
