import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

TOKENS = "ARNDCQEGHILKMFPSTWYV"


def tokenize(sequence: str) -> np.ndarray:
    return np.array([TOKENS.index(s) for s in sequence], dtype=np.int32)


class LossTerm(eqx.Module):
    def __call__(self, *args, key, **kwds) -> tuple[float, dict]:
        raise NotImplementedError

    def __rmul__(self, scalar: float):
        if not (isinstance(scalar, float) or isinstance(scalar, int)):
            return NotImplemented

        return LinearCombination(l=[self], weights=jnp.array([scalar]))

    def __add__(self, other):
        return 1.0 * self + 1.0 * other

    def __neg__(self):
        return (-1.0) * self

    def __sub__(self, other):
        return self + (-1.0) * other


class LinearCombination(eqx.Module):
    """Weighted linear combination of loss terms."""

    # losses: list[tuple[float, any]]
    l: list[LossTerm]
    weights: jax.Array

    def __call__(self, *args, key, **kwargs) -> tuple[float, list]:
        r = 0.0
        aux_values = []
        for w, loss in zip(self.weights, self.l):
            v, a = loss(*args, key=key, **kwargs)
            key = jax.random.fold_in(key, 1)
            r += w * v
            aux_values.append(a)
        return r, aux_values

    def __rmul__(self, scalar: float):
        if not (isinstance(scalar, float) or isinstance(scalar, int)):
            return NotImplemented

        return LinearCombination(
            l=self.l,
            weights=self.weights * scalar,
        )

    def __add__(self, other):
        if isinstance(other, LossTerm):
            other = 1.0 * other  # lift to LinearCombination

        if not isinstance(other, LinearCombination):
            return NotImplemented

        return LinearCombination(
            l=self.l + other.l,
            weights=jnp.concatenate([self.weights, other.weights]),
        )

    def __sub__(self, other):
        return self + (-1.0) * other

    def __neg__(self):
        return (-1.0) * self
