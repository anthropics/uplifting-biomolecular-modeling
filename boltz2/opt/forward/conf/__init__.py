"""opt/forward/conf — lever implementations for Boltz-2 2.2.1's modules after the trunk / outside the sampler / at the
front: the template module (template_levers.py) and the diffusion conditioning (cond_levers.py). Applied in the worker process by the engine adapter ``boltz2_opt.conf`` (``worker_launch --attach conf``,
switch ``BOLTZ_CONF=<word>[,<word>…]`` — one word per lever); nothing here reads the environment or patches anything at import.
The patched classes are boltz 2.2.1's (MIT), read-only under stock/.
"""
LEVER_MODULES = ("template_levers", "cond_levers")
