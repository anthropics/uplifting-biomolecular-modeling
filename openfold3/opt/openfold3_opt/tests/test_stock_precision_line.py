"""The stock caller's proof line names the EFFECTIVE Lightning precision of the run beside the effective eval kernel flags
(stock_pred.effective_precision): the runner yaml's pl_trainer_args.precision as given, else Lightning's own default 32-true — what
upstream's shipped predict configuration (opt/openfold3_opt/shipped_predict.yml: model_update presets [predict], nothing else) runs at."""
import os

from openfold3_opt import modes, stock_pred

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_effective_precision_words():
    assert stock_pred.effective_precision(os.path.join(HOME, modes.SHIPPED_YAML)).startswith("32-true (Lightning default")
    assert stock_pred.effective_precision(os.path.join(HOME, modes.KERNELS_OFF_YAML)).startswith("32-true (Lightning default")
    assert stock_pred.effective_precision(os.path.join(HOME, modes.STOCK_YAML)) == "bf16-mixed"        # the tree's stock configuration names it
    assert stock_pred.effective_precision(os.path.join(HOME, modes.FAST_BF16_YAML)) == "bf16-mixed"
    assert stock_pred.LIGHTNING_DEFAULT_PRECISION == "32-true"
