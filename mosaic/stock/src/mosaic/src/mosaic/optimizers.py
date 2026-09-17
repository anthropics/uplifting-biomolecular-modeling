import time
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PyTree
from scipy.special import log_softmax, softmax

from mosaic.common import LinearCombination, LossTerm

AbstractLoss = LossTerm | LinearCombination


def _ranking_leaf(aux, ranking_aux_name: str):
    """Locate the ranking-metric leaf in a loss's ``aux`` tree, or ``None``.

    ``ranking_aux_name`` is matched as a dict key ANYWHERE in the path, so the
    lookup is robust to nesting and to ``reduce_samples`` wrapping per-sample
    scalar metrics into loss-sorted lists (which appends a ``SequenceKey`` and
    would defeat a terminal-key match). When the metric is such a per-sample
    list, the best sample -- list index 0, since aux is sorted ascending by
    loss -- is returned.
    """
    matches = [
        (path, leaf)
        for path, leaf in jax.tree_util.tree_leaves_with_path(aux)
        if any(getattr(k, "key", None) == ranking_aux_name for k in path)
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][1]

    def _trailing_seq_idx(path):
        for k in reversed(path):
            if isinstance(k, jax.tree_util.SequenceKey):
                return k.idx
        return 0

    return min(matches, key=lambda pl: _trailing_seq_idx(pl[0]))[1]


def _print_iter(iter, aux, v):
    # first filter out anything that isn't a float or has number of dimensions > 0
    aux = eqx.filter(
        aux,
        lambda v: isinstance(v, float | str) or v.shape == (),
    )
    print(
        iter,
        f"loss: {v:0.2f}",
        " ".join(
            f"{jax.tree_util.keystr(k, simple=True, separator='.')}:{v: 0.2f}"
            for (k, v) in jax.tree_util.tree_leaves_with_path(aux)
            if hasattr(v, "item") or isinstance(v, float)
        ),
    )


# Split this up so changing optim parameters doesn't trigger re-compilation of loss function
def _eval_loss_and_grad(
    loss_function: AbstractLoss,
    x,
    key,
):
    """
    Evaluates the loss function and its gradient.

    Args:
    - loss_function: ...
    - x: soft sequence (N x 20 array with each row in the simplex)
    - key: jax random key

    Returns:
    - ((value, aux), g): value of the loss function and auxiliary information, and gradient of the loss with respect to x

    """
    # standardize input to avoid recompilation
    x = np.array(x, dtype=np.float32)
    (v, aux), g = _____eval_loss_and_grad(loss_function, x=x, key=key)
    return (jnp.nan_to_num(v, nan=1000000.0), aux), jnp.nan_to_num(
        g - g.mean(axis=-1, keepdims=True)
    )


# more underscores == more private
@eqx.filter_jit
def _____eval_loss_and_grad(loss, x, key):
    return eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)


@eqx.filter_jit
def batched_eval(
    loss: AbstractLoss,
    xs: Float[Array, "B N K"],
    keys: jax.Array,
) -> tuple[Float[Array, " B"], PyTree, Float[Array, "B N K"]]:
    """Evaluate loss+grad for B sequences with B keys."""
    assert xs.ndim == 3, f"xs must be 3D [B, N, K], got {xs.ndim}D"

    def single(x: Float[Array, "N K"], key: jax.Array):
        (v, aux), g = eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)
        v = jnp.nan_to_num(v, nan=1e6)
        g = jnp.nan_to_num(g - g.mean(axis=-1, keepdims=True))
        return v, aux, g

    return jax.vmap(single)(xs, keys)


# def _proposal(sequence, g, temp, alphabet_size: int = 20):
#     input = jax.nn.one_hot(sequence, alphabet_size)
#     g_i_x_i = (g * input).sum(-1, keepdims=True)
#     logits = -((input * g).sum() - g_i_x_i + g) / temp
#     return jax.nn.softmax(logits), jax.nn.log_softmax(logits)


# rewrite in numpy to use float64
# note: this _does not match the taylor expansion estimate (above)_ (there's an extra normalization). seems better though.
def _proposal(sequence, g, temp, alphabet_size: int = 20):
    input = np.eye(alphabet_size)[sequence]
    g_i_x_i = (g * input).sum(-1, keepdims=True)
    logits = -((input * g).sum(-1, keepdims=True) - g_i_x_i + g) / temp
    return softmax(logits, axis=-1), log_softmax(logits, axis=-1)


def gradient_MCMC(
    loss,
    sequence: Int[Array, " N"],
    temp=0.001,
    proposal_temp=0.01,
    max_path_length=2,
    steps=50,
    alphabet_size: int = 20,
    key: None = None,
    detailed_balance: bool = False,
    fix_loss_key: bool = True,
):
    """
    Implements the gradient-assisted MCMC sampler from "Plug & Play Directed Evolution of Proteins with
    Gradient-based Discrete MCMC." Uses first-order taylor approximation of the loss to propose mutations.

        WARNING: Fixes random seed used for loss evaluation.

    Args:
    - loss: log-probability/function to minimize
    - sequence: initial sequence
    - proposal_temp: temperature of the proposal distribution
    - temp: temperature for the loss function
    - max_path_length: maximum number of mutations per step
    - steps: number of optimization steps
    - key: jax random key
    - detailed_balance: whether to maintain detailed balance

    """

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    key_model = key
    (v_0, aux_0), g_0 = _eval_loss_and_grad(
        loss, jax.nn.one_hot(sequence, alphabet_size), key=key_model
    )
    for iter in range(steps):
        start_time = time.time()
        ### generate a proposal

        for i in range(50):
            proposal = sequence.copy()
            mutations = []
            log_q_forward = 0.0
            path_length = jax.random.randint(
                key=jax.random.key(np.random.randint(10000)),
                minval=1,
                maxval=max_path_length + 1,
                shape=(),
            )
            key = jax.random.fold_in(key, 0)
            for _ in range(path_length):
                p, log_p = _proposal(
                    proposal, g_0, proposal_temp, alphabet_size=alphabet_size
                )
                mut_idx = jax.random.choice(
                    key=key,
                    a=len(np.ravel(p)),
                    p=np.ravel(p),
                    shape=(),
                )
                key = jax.random.fold_in(key, 0)
                position, AA = np.unravel_index(mut_idx, p.shape)
                log_q_forward += log_p[position, AA]
                mutations += [(position, AA)]
                proposal = proposal.at[position].set(AA)
            # check if proposal is same as current sequence
            if np.all(proposal == sequence):
                print(f"\t {i}: proposal is the same as current sequence, skipping.")
                # _print_iter(iter, {"": aux_0, "time": time.time() - start_time}, v_0)
                # continue
            else:
                break
        muts = ", ".join([f"{pos}:{aa}" for (pos, aa) in mutations])
        print(f"Proposed mutations: {muts}")

        ### evaluate the proposal
        (v_1, aux_1), g_1 = _eval_loss_and_grad(
            loss,
            jax.nn.one_hot(proposal, alphabet_size),
            key=key_model if fix_loss_key else key,
        )

        # next bit is to calculate the backward probability, which is only used
        # if detailed_balance is True
        prop_backward = proposal.copy()
        log_q_backward = 0.0
        for position, AA in reversed(mutations):
            p, log_p = _proposal(
                prop_backward, g_1, proposal_temp, alphabet_size=alphabet_size
            )
            log_q_backward += log_p[position, AA]
            prop_backward = prop_backward.at[position].set(AA)

        log_acceptance_probability = (v_0 - v_1) / temp + (
            (log_q_backward - log_q_forward) if detailed_balance else 0.0
        )

        log_acceptance_probability = min(0.0, log_acceptance_probability)

        print(
            f"iter: {iter}, accept {np.exp(log_acceptance_probability): 0.3f} {v_0: 0.3f} {v_1: 0.3f} {log_q_forward: 0.3f} {log_q_backward: 0.3f}"
        )

        print()
        if -jax.random.exponential(key=key) < log_acceptance_probability:
            sequence = proposal
            (v_0, aux_0), g_0 = (v_1, aux_1), g_1

        _print_iter(iter, {"": aux_0, "time": time.time() - start_time}, v_0)

        key = jax.random.fold_in(key, 0)

    return sequence


def projection_simplex(V, z=1):
    """
    From https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    """
    V = np.array(V, dtype=np.float64)
    n_features = V.shape[1]
    U = np.sort(V, axis=1)[:, ::-1]
    z = np.ones(len(V)) * z
    cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
    ind = np.arange(n_features) + 1
    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=1)
    theta = cssv[np.arange(len(V)), rho - 1] / rho
    return np.maximum(V - theta[:, np.newaxis], 0)


def simplex_APGM(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    n_steps: int,
    stepsize: float,
    momentum: float = 0.0,
    key=None,
    max_gradient_norm: float | None = None,
    scale=1.0,
    trajectory_fn: Callable[tuple[PyTree, Float[Array, "N 20"]], any] | None = None,
    logspace: bool = False,
):
    """
    Accelerated projected gradient descent on the simplex.

    Args:
    - loss_function: function to minimize
    - x: initial sequence
    - n_steps: number of optimization steps
    - stepsize: step size for gradient descent
    - momentum: momentum factor
    - key: jax random key
    - max_gradient_norm: maximum norm of the gradient
    - scale: proximal scaling factor for L2 regularization (or entropic regularization if logspace=True), set to > 1.0 to encourage sparsity
    - trajectory_fn: function to compute trajectory information, takes (aux, x) and returns any value.
    - logspace: whether to optimize in log space, which corresponds to a bregman proximal algorithm.

    returns:
    - x: final soft sequence after optimization
    - best_x: best soft sequence found during optimization
    - trajectory: list of trajectory information if `trajectory_fn` is provided, otherwise nothing.
    """

    if max_gradient_norm is None:
        max_gradient_norm = np.sqrt(x.shape[0])

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    best_val = np.inf
    x = projection_simplex(x) if not logspace else x
    best_x = x

    x_prev = x

    trajectory = []

    for _iter in range(n_steps):
        start_time = time.time()
        v = jax.device_put(x + momentum * (x - x_prev))
        (value, aux), g = _eval_loss_and_grad(
            x=v if not logspace else jax.nn.softmax(v),
            loss_function=loss_function,
            key=key,
        )

        n = np.sqrt((g**2).sum())
        if n > max_gradient_norm:
            g = g * (max_gradient_norm / n)

        key = jax.random.fold_in(key, 0)

        if logspace:
            x_new = scale * (v - stepsize * g)
        else:
            x_new = projection_simplex(scale * (v - stepsize * g))

        x_prev = x
        x = x_new

        if value < best_val and not np.isnan(value):
            best_val = value
            best_x = (
                x  # this isn't exactly right, because we evaluated loss at v, not x.
            )

        average_nnz = (
            (x > 0.01).sum(-1).mean()
            if not logspace
            else (jax.nn.softmax(x) > 0.01).sum(-1).mean()
        )

        aux = {
            "loss": value,
            "nnz": average_nnz,
            "time": (time.time() - start_time),
            "": aux,
        }
        if trajectory_fn is not None:
            trajectory.append(trajectory_fn(aux, x))

        _print_iter(
            _iter,
            eqx.filter(
                aux,
                lambda v: isinstance(v, float) or v.shape == (),
            ),
            value,
        )

    if logspace:
        x = jax.nn.softmax(x)
        best_x = jax.nn.softmax(best_x)

    if trajectory_fn is None:
        return x, best_x
    else:
        return x, best_x, trajectory


def batched_simplex_APGM(
    *,
    loss_function: AbstractLoss,
    x: Float[Array, "B N 20"],
    n_steps: int,
    stepsize: float,
    momentum: float = 0.0,
    key: jax.Array | None = None,
    max_gradient_norm: float | None = None,
    scale: float = 1.0,
    logspace: bool = False,
) -> tuple[Float[Array, "B N 20"], Float[Array, "B N 20"]]:
    """
    Batched accelerated projected gradient descent on the simplex.
    Runs B copies of the optimization in parallel via vmap, where B = x.shape[0].

    Args:
    - loss_function: loss function (same for all designs)
    - x: initial soft sequences [B, N, 20]
    - n_steps: number of optimization steps
    - stepsize: step size (scalar or [B, 1, 1] array for per-design values)
    - momentum: momentum factor (scalar or [B, 1, 1] array)
    - key: jax random key
    - max_gradient_norm: maximum norm of the gradient
    - scale: proximal scaling factor
    - logspace: whether to optimize in log space

    returns:
    - x: final soft sequences [B, N, 20]
    - best_x: best soft sequences found during optimization [B, N, 20]
    """
    assert x.ndim == 3, f"x must be 3D [B, N, 20], got {x.ndim}D"
    B = x.shape[0]

    if max_gradient_norm is None:
        max_gradient_norm = np.sqrt(x.shape[1])

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    if not logspace:
        flat = np.array(x).reshape(-1, x.shape[-1])
        x = jnp.array(projection_simplex(flat).reshape(x.shape), dtype=jnp.float32)

    best_vals = jnp.full(B, jnp.inf)
    best_x = x
    x_prev = x

    for _iter in range(n_steps):
        start_time = time.time()
        v = jnp.array(x + momentum * (x - x_prev), dtype=jnp.float32)
        v_eval = jax.nn.softmax(v, axis=-1) if logspace else v

        values, auxs, grads = batched_eval(loss_function, v_eval, jax.random.split(key, B))

        norms = np.sqrt((grads**2).sum(axis=(-2, -1)))
        clip = np.where(norms > max_gradient_norm, max_gradient_norm / norms, 1.0)
        grads = grads * np.asarray(clip)[:, None, None]

        key = jax.random.fold_in(key, 0)

        if logspace:
            x_new = scale * (v - stepsize * grads)
        else:
            flat = np.array(scale * (v - stepsize * grads)).reshape(-1, x.shape[-1])
            x_new = jnp.array(projection_simplex(flat).reshape(x.shape), dtype=jnp.float32)

        x_prev = x
        x = x_new

        better = (np.array(values) < np.array(best_vals)) & ~np.isnan(values)
        best_vals = jnp.where(jnp.array(better), values, best_vals)
        best_x = jnp.where(jnp.array(better)[:, None, None], x, best_x)

        for i in range(B):
            aux_i = jax.tree.map(lambda v: v[i], auxs)
            average_nnz = (
                (x[i] > 0.01).sum(-1).mean()
                if not logspace
                else (jax.nn.softmax(x[i]) > 0.01).sum(-1).mean()
            )
            _print_iter(
                f"{_iter}[{i}]",
                {"loss": values[i], "nnz": average_nnz, "time": time.time() - start_time, "": aux_i},
                values[i],
            )

    if logspace:
        x = jax.nn.softmax(x, axis=-1)
        best_x = jax.nn.softmax(best_x, axis=-1)

    return x, best_x


def _topb_unseen_mutations(seq, g, seen, b):
    """Pick up to b 1-hop neighbours of `seq` ranked by first-order predicted delta.

    Returns (candidates, predicted_deltas) with shapes (m, N) and (m,), m <= b.
    Returns None if every 1-hop neighbour has already been seen.
    """
    N, K = g.shape
    a0 = seq.astype(np.int64)
    delta = g - g[np.arange(N), a0][:, None]
    delta[np.arange(N), a0] = np.inf  # mask no-ops

    order = np.argsort(delta.ravel(), kind="stable")

    cands = []
    deltas = []
    for idx in order:
        d = delta.ravel()[idx]
        if not np.isfinite(d):
            break
        pos, aa = divmod(int(idx), K)
        cand = seq.copy()
        cand[pos] = aa
        if cand.tobytes() in seen:
            continue
        cands.append(cand)
        deltas.append(float(d))
        if len(cands) == b:
            break

    if not cands:
        return None
    return np.stack(cands), np.asarray(deltas)


def batch_greedy_descent(
    loss: AbstractLoss,
    sequence: Int[Array, " N"],
    *,
    batch_size: int = 16,
    steps: int = 100,
    alphabet_size: int = 20,
    key: jax.Array | None = None,
) -> tuple[np.ndarray, float]:
    """Greedy batch hillclimb on a discrete sequence.

    Each step: compute the gradient at the current sequence, rank all
    single-point mutations by predicted first-order delta, evaluate the
    top `batch_size` unseen candidates in parallel, and greedily accept
    the best if it improves. Stops early when the full 1-hop neighbourhood
    has been evaluated.

    Args:
    - loss: loss function (called as loss(x, key=...) returning (value, aux))
    - sequence: (N,) int starting sequence
    - batch_size: number of candidate mutations evaluated per step
    - steps: maximum number of steps
    - alphabet_size: token alphabet size
    - key: jax random key (fixed across all evals for deterministic comparison)

    Returns:
    - best_seq: best sequence found
    - best_val: loss at best sequence
    """
    sequence = np.asarray(sequence, dtype=np.int32).copy()
    assert sequence.ndim == 1, f"sequence must be 1D [N], got {sequence.ndim}D"
    B = int(batch_size)

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    # initial eval
    x0 = jax.nn.one_hot(jnp.asarray(sequence[None]), alphabet_size)
    vals, aux0, grads = batched_eval(loss, x0, jnp.broadcast_to(key, (x0.shape[0], *key.shape)))
    v = float(np.asarray(vals)[0])
    g = np.asarray(grads)[0]
    aux = jax.tree.map(lambda a: a[0], aux0)

    _print_iter("init", {"": aux}, v)

    best_seq = sequence.copy()
    best_val = v
    seen: set[bytes] = {sequence.tobytes()}

    for it in range(steps):
        start_time = time.time()

        picked = _topb_unseen_mutations(sequence, g, seen, B)
        if picked is None:
            print(f"step {it}: neighbourhood exhausted, stopping")
            break
        cands, _ = picked
        m = cands.shape[0]

        xs = jax.nn.one_hot(jnp.asarray(cands), alphabet_size)
        vals, auxs, grads_batch = batched_eval(loss, xs, jnp.broadcast_to(key, (xs.shape[0], *key.shape)))
        vals_np = np.asarray(vals)

        for c in cands:
            seen.add(c.tobytes())

        best_in_batch = int(np.argmin(vals_np[:m]))
        v_best = float(vals_np[best_in_batch])

        if v_best < v:
            sequence = cands[best_in_batch].copy()
            v = v_best
            g = np.asarray(grads_batch)[best_in_batch]
            aux = jax.tree.map(lambda a: a[best_in_batch], auxs)

        if v < best_val:
            best_val = v
            best_seq = sequence.copy()

        _print_iter(
            it,
            {"": aux, "time": time.time() - start_time},
            v,
        )

    return best_seq, best_val


def biohub_optimizer(
    *,
    loss_function,
    logits: Float[Array, "B N 20"],
    n_steps: int = 150,
    learning_rate: float = 0.1,
    temperature_min: float = 1e-2,
    mask: Float[Array, "N 20"] | None = None,
    key: jax.Array,
    verbose: bool = False,
    tail_loss_function=None,
    tail_select_steps: int = 20,
    beta: float = 0.0,
    ranking_aux_name: str = "ranking_loss"
):
    """Standard Biohub ESMFold2 optimizer (Algorithm 11 in https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1)

    Returns `(best_designs, best_loss)` sorted best-first. `best_designs` is
    argmax-ready only: improved tail iterates are stored as softmax
    probabilities, so consume it via `best_designs.argmax(-1)`.
    """
    assert 0 < tail_select_steps <= n_steps, (
        "tail_select_steps must be in (0, n_steps]; otherwise no iterate is ever selected"
    )
    logits = jnp.array(logits)

    if mask is not None:
        assert mask.ndim == 2
        mask = jnp.array(mask)[None]
        logits = jnp.where(mask > 0, logits, -1e9)

    if tail_loss_function is None:
        tail_loss_function = loss_function

    _warned_ranking = False

    def get_ranking_values(values, aux):
        nonlocal _warned_ranking
        leaf = _ranking_leaf(aux, ranking_aux_name)
        if leaf is None:
            if not _warned_ranking:
                print(f"Warning: biohub_optimizer ranking by loss value, not {ranking_aux_name}")
                _warned_ranking = True
            return values
        return leaf


    assert logits.ndim == 3, "biohub_optimizer is batched, `logits` must have shape [B, N, 20]"
    B = logits.shape[0]

    best_designs = np.asarray(logits).copy()
    best_loss = np.full(B, np.inf)
    v = np.zeros_like(logits)

    for step in range(n_steps):
        start_time = time.time()
        t = (step + 1) / n_steps
        temp = temperature_min + (1 - temperature_min) * 0.5 * (1 + np.cos(np.pi * t))

        in_tail = step >= n_steps - tail_select_steps
        lf = tail_loss_function if in_tail else loss_function

        design = jax.nn.softmax(logits / temp, axis=-1)
        values, aux, g_design = batched_eval(lf, design, jax.random.split(key, B))
        key = jax.random.fold_in(key, 0)

        # best-of-tail
        if in_tail:
            # best loss according to ranking_loss aux _or_ the loss
            ranking_values = get_ranking_values(values, aux)
            improved = best_loss > ranking_values                      
            design_np, vals_np = np.asarray(design), np.asarray(ranking_values)
            best_designs[improved] = design_np[improved]
            best_loss[improved] = vals_np[improved]

        # chain-rule through softmax(logits / T): dL/dz = (design ⊙ g − design·(design·g)) / T
        xg = (design * g_design).sum(-1, keepdims=True)
        g_z = (design * (g_design - xg)) / temp
        if mask is not None:
            g_z = g_z * mask

        # normalize per design to ‖·‖ = √(eff_L) (biohub normalized_gradient_tensor)
        eff_L = (jnp.square(g_z).sum(-1) > 0).sum(-1)                        # [B]
        gnorm = jnp.linalg.norm(g_z, axis=(-1, -2))                          # [B]
        g_z = g_z / (gnorm[:, None, None] + 1e-7) * jnp.sqrt(eff_L)[:, None, None]

        v = beta * v + (learning_rate * temp) * g_z

        logits = logits - v

        for i in range(B):
            aux_i = jax.tree.map(lambda v: v[i], aux)
            average_nnz = (
                (design[i] > 0.01).sum(-1).mean()
            )
            _print_iter(
                f"{step}[{i}]",
                {"loss": values[i], "time": time.time() - start_time, "": aux_i, "nnz": average_nnz, "temp": temp},
                values[i],

            )
        if mask is not None:
            logits = jnp.where(mask > 0, logits, -1e9)

    order = np.argsort(best_loss)

    return best_designs[order], best_loss[order]


def _colabdesign_transform(z, soft, temp, hard):
    """Map ColabDesign logits to the sequence passed to the user's loss."""
    soft_seq = jax.nn.softmax(z / temp)
    hard_seq = jax.nn.one_hot(soft_seq.argmax(-1), z.shape[-1])
    # straight thru estimator
    hard_seq = jax.lax.stop_gradient(hard_seq - soft_seq) + soft_seq
    pseudo = soft * soft_seq + (1.0 - soft) * z
    # if hard true use only hard_seq
    return hard * hard_seq + (1.0 - hard) * pseudo


# transform grad computed separately so switching optimizers doesn't recompile everything
@jax.jit
def _colabdesign_pullback(z, g, soft, temp, hard):
    """Pull a sequence gradient back to ColabDesign logits."""
    _, pullback = jax.vjp(
        lambda z: _colabdesign_transform(z, soft, temp, hard),
        z,
    )
    (g_z,) = pullback(g)
    return g_z


def _seq_grad_norm(g):
    """Lifted from ColabDesign _norm_seq_grad."""
    eff_L = (jnp.square(g).sum(-1, keepdims=True) > 0).sum(-2, keepdims=True)
    gn = jnp.linalg.norm(g, axis=(-1, -2), keepdims=True)
    return g * jnp.sqrt(eff_L) / (gn + 1e-7)


def colabdesign_stage(
    *,
    loss_function: AbstractLoss,
    x: Float[Array, "N 20"],
    n_steps: int,
    soft_start: float,
    soft_end: float,
    temp_start: float,
    temp_end: float,
    hard: bool,
    lr: float,
    norm_seq_grad: bool = True,
    max_gradient_norm: float | None = None,
    key=None,
    trajectory_fn: Callable[tuple[PyTree, Float[Array, "N 20"]], any] | None = None,
):
    """
    One ColabDesign/AfDesign stage: n_steps of normalised SGD on a logit iterate,
    with soft ramping linearly, temp annealing quadratically, and hard fixed. Logits
    in and out, so stages chain with no softmax round-trip (the BindCraft recipe).

    Args:
    - loss_function: function to minimize
    - x: initial logits (N x K), centered per row on entry
    - n_steps: number of optimization steps
    - soft_start, soft_end: linear soft ramp (0 = raw logits, 1 = softmax)
    - temp_start, temp_end: quadratic temp anneal
    - hard: if True, blend in a straight-through one-hot for the stage
    - lr: base learning rate (ColabDesign default 0.1)
    - norm_seq_grad: normalise the gradient to sqrt(L); else clip at max_gradient_norm
    - max_gradient_norm: clip norm when norm_seq_grad is False (default sqrt(N))
    - key: jax random key
    - trajectory_fn: takes (aux, x) and returns any value

    returns:
    - x: final logits
    - trajectory: list of trajectory information if `trajectory_fn` is provided
    """
    if max_gradient_norm is None:
        max_gradient_norm = float(np.sqrt(x.shape[0]))
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    x = jnp.asarray(x, dtype=jnp.float32)
    # Center logits per row: log(pssm) carries a DC offset that softmax ignores but
    # is off-distribution for the raw-logits stage (soft<1). The mosaic gradient is
    # row-centered, so x stays zero-mean (re-centering a chained stage is a no-op).
    x = x - x.mean(-1, keepdims=True)
    trajectory = []

    for _iter in range(n_steps):
        start_time = time.time()
        frac = (_iter + 1) / n_steps
        soft = soft_start + (soft_end - soft_start) * frac  # linear ramp
        temp = temp_end + (temp_start - temp_end) * (1 - frac) ** 2  # quadratic anneal

        soft = jnp.asarray(soft, dtype=jnp.float32)
        temp = jnp.asarray(temp, dtype=jnp.float32)
        hard = jnp.asarray(hard, dtype=jnp.float32)
        pseudo = _colabdesign_transform(x, soft, temp, hard)
        # compute loss gradient
        (value, aux), g = _eval_loss_and_grad(loss_function, pseudo, key)
        # apply vjp 
        g = _colabdesign_pullback(x, g, soft, temp, hard)
        value = float(value)
        key = jax.random.fold_in(key, 0)

        if norm_seq_grad:
            g = _seq_grad_norm(g)
        else:
            n = np.sqrt((g**2).sum())
            if n > max_gradient_norm:
                g = g * (max_gradient_norm / n)

        lr_scale = (1.0 - soft) + soft * temp
        x = x - (lr * lr_scale) * g

        average_nnz = float((jax.nn.softmax(x) > 0.01).sum(-1).mean())
        aux = {
            "loss": value,
            "nnz": average_nnz,
            "time": time.time() - start_time,
            "soft": float(soft),
            "temp": float(temp),
            "hard": float(hard),
            "": aux,
        }
        if trajectory_fn is not None:
            trajectory.append(trajectory_fn(aux, x))

        _print_iter(
            _iter,
            eqx.filter(aux, lambda v: isinstance(v, float) or v.shape == ()),
            value,
        )

    if trajectory_fn is None:
        return x
    return x, trajectory


def bindcraft_design(
    *,
    loss_function: AbstractLoss,
    x: Float[Array, "N 20"],
    lr: float = 0.1,
    logits_iters: tuple[int, int] = (50, 25),
    soft_iters: int = 45,
    hard_iters: int = 5,
    key=None,
    trajectory_fn: Callable[tuple[PyTree, Float[Array, "N 20"]], any] | None = None,
):
    """Run the default four-stage BindCraft gradient design schedule.

    The caller supplies initial logits. To match the paper initialization, use
    x = 0.01 * jax.random.normal(key, (N, K)).
    
    Parameters follow default_4stage_multimer.json in BindCraft package.
    See the "materials and methods" section here: https://www.biorxiv.org/content/10.1101/2024.09.30.615802v1

    Args:
    - loss_function: function to minimise.
    - x: initial logits with shape (N, K).
    - lr: base learning rate.
    - logits_iters: step counts for the two logits stages.
    - soft_iters: step count for the soft stage.
    - hard_iters: step count for the hard stage.
    - key: JAX random key.
    - trajectory_fn: optional callback receiving (aux, x).

    Returns:
    - pssm: final soft sequence with shape (N, K).
    - trajectory: concatenated callback results, when requested.
    """
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    n1, n2 = logits_iters
    stages = [
        {
            "n_steps": n1,
            "soft_start": 0.0,
            "soft_end": 0.9,
            "temp_start": 1.0,
            "temp_end": 1.0,
            "hard": False,
        },
        {
            "n_steps": n2,
            "soft_start": 0.9,
            "soft_end": 1.0,
            "temp_start": 1.0,
            "temp_end": 1.0,
            "hard": False,
        },
        {
            "n_steps": soft_iters,
            "soft_start": 1.0,
            "soft_end": 1.0,
            "temp_start": 1.0,
            "temp_end": 1e-2,
            "hard": False,
        },
        {
            "n_steps": hard_iters,
            "soft_start": 1.0,
            "soft_end": 1.0,
            "temp_start": 1e-2,
            "temp_end": 1e-2,
            "hard": True,
        },
    ]
    stage_keys = jax.random.split(key, len(stages))

    trajectory = []
    for stage, k in zip(stages, stage_keys):
        out = colabdesign_stage(
            loss_function=loss_function,
            x=x,
            lr=lr,
            key=k,
            trajectory_fn=trajectory_fn,
            **stage,
        )
        x, t = out if trajectory_fn is not None else (out, [])
        trajectory += t

    # ColabDesign uses the final (post-hard) iterate; softmax to read out a pssm.
    pssm = jax.nn.softmax(x)
    if trajectory_fn is None:
        return pssm
    return pssm, trajectory
