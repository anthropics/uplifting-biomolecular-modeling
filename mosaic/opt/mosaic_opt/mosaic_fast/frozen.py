"""P3 — freeze featurized inputs. Boltz-2 featurization re-draws `ref_pos` (RDKit conformer sampling) in every process, so the same seed does not
reproduce a trajectory unless the features are frozen. save_features(features, path) / load_features(path) round-trip the dict of arrays as npz
(+ sha256). The structure writer is not serialisable; for the final PDB/mmCIF call model.binder_features(...) once in the writing process and use
model.predict with the frozen arrays (coordinates do not depend on the writer)."""
import hashlib, numpy as np
def save_features(features: dict, path: str) -> str:
    np.savez(path, **{k: np.asarray(v) for k, v in features.items()})
    return hashlib.sha256(open(path if path.endswith(".npz") else path + ".npz", "rb").read()).hexdigest()
def load_features(path: str, expected_sha256: str | None = None):
    import jax.numpy as jnp
    if expected_sha256 is not None:
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert got == expected_sha256, f"frozen feature sha256 mismatch: {got} != {expected_sha256}"
    z = np.load(path)
    return {k: jnp.array(z[k]) for k in z.files}
