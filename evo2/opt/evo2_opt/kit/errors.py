"""The kit's one refusal type: a launch condition of the levers does not hold on the constructed model (named; nothing is patched)."""


class KitRefused(RuntimeError):
    """The exact levers cannot be installed on this model / stack as constructed — the message names the condition."""
