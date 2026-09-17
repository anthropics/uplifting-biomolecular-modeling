"""accel -- the acceleration levers for GPN-Star (songlab-cal/gpn @ 6f28c81b): exact (bitwise) and fast (TF32) modes."""

from __future__ import annotations


GPN_COMMIT = "6f28c81bcbfe7d65cb6d8ece9ce88f87ca583791"

# Released GPN-Star checkpoints (Hugging Face, immutable revisions) and the paper's native windows.
MODELS = {
    "v100-200m": dict(repo="songlab/gpn-star-hg38-v100-200m", revision="0c949f132d35619a3eb188b402848c998a3313ae",
                      n_species=100, native_window=128, params_m=203, desc="hg38, 100-way vertebrate, 200M (README default)"),
    "m447-200m": dict(repo="songlab/gpn-star-hg38-m447-200m", revision="0d8f5164fba8e8108200c160370b86652bfb6244",
                      n_species=447, native_window=256, params_m=203, desc="hg38, 447-way mammal, 200M (heaviest released workload)"),
    "p243-200m": dict(repo="songlab/gpn-star-hg38-p243-200m", revision="16773016130e826cd3d72e91ed312a3cb27ba2b5",
                      n_species=243, native_window=256, params_m=203, desc="hg38, 243-way primate, 200M (paper's top model)"),
    "mm39-85m": dict(repo="songlab/gpn-star-mm39-v35-85m", revision="dac8b6b1debe89e20dd96c94c25fb6273fb6fa38",
                     n_species=35, native_window=128, params_m=86, desc="mouse, 35-way, 85M"),
    "ce11-25m": dict(repo="songlab/gpn-star-ce11-n135-25m", revision="078ad60c23bbaf40fbeae89fb509a21320a71bbb",
                     n_species=135, native_window=128, params_m=26, desc="worm, 135-way nematode, 25M (small)"),
    "tair10-25m": dict(repo="songlab/gpn-star-tair10-b18-25m", revision="317841ed55164ff4cd3d14e636bcc2d5de23cf49",
                       n_species=18, native_window=128, params_m=26, desc="arabidopsis, 18-way, 25M (smallest)"),
}


def load_model(key: str, device: str = "cuda", dtype=None):
    """Load a stock GPN-Star MLM exactly as gpn.star.inference.MLMforVEPModel does (fp32, eval)."""
    import torch
    from gpn import register_auto_classes
    from transformers import AutoModelForMaskedLM

    spec = MODELS[key]
    register_auto_classes("star")
    kwargs = {"revision": spec["revision"]}
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForMaskedLM.from_pretrained(spec["repo"], **kwargs)
    model.eval()
    if device is not None:
        model.to(device)
    torch.cuda.synchronize() if device == "cuda" else None
    return model
