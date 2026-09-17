"""stand-in jax.experimental.serialize_executable (tests only): importable so lever `lowercache` installs on the stand-in stack; the
stand-in programs are plain callables (no lowering), so the lever takes its traced path and nothing is ever (de)serialized here."""


def serialize(compiled):
    raise RuntimeError("stand-in stack: no executable to serialize")


def deserialize_and_load(serialized, in_tree, out_tree, backend=None, execution_devices=None):
    raise RuntimeError("stand-in stack: no executable to load")
