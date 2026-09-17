"""fpf_clisampler — the diffusion-sampler CUDA-graph + step-invariant hoist levers (`infopt_graphs.protenix.GraphedDenoiseLoop`, `dit_hoist`) wired into the stock `protenix pred` CLI path via a runner hook. See clisampler.py."""
from .clisampler import install_hook, report  # noqa: F401
