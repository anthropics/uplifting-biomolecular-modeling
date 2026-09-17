"""The kernels' typed refusal."""


class Unsupported(NotImplementedError):
    """Raised, by name, for inputs these kernels do not handle (nothing falls back silently)."""
