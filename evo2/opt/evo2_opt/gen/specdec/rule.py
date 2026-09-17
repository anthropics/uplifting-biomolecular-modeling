"""The speculative-sampling rule (Leviathan et al. 2023, Chen et al. 2023) on rows of two distributions over the vocabulary: p_i = the
target's for position i, q_i = the draft's, d_i = the draft's token drawn from q_i, u_i uniform in [0, 1). Arithmetic in float64 on the
values given (the callers pass the sampler transform's probabilities); draws through torch's generator (``torch.multinomial``,
``torch.rand`` of the caller), so ``torch.manual_seed`` governs a run.

    accept d_1..d_n while u_i · q_i(d_i) < p_i(d_i)                      (u_i < p/q without the division; q_i(d_i) > 0: d_i was drawn from q_i)
    at the first rejected position j: the token is drawn from norm(max(p_j − q_j, 0))
    all accepted: one more token is drawn from the next target row (the bonus)
    greedy (one-hot p): accept while d_i == argmax p_i; the correction / bonus is argmax of the row
Each emitted token is then distributed exactly as p at its position."""
import torch


def accepted_prefix(p_rows, q_rows, drafts, u) -> int:
    """Number of leading drafts accepted. p_rows, q_rows: (n, V); drafts: (n,) int64; u: (n,) in [0, 1)."""
    n = int(drafts.numel())
    if n == 0:
        return 0
    idx = drafts.view(n, 1)
    p_sel = p_rows.double().gather(-1, idx)[:, 0]
    q_sel = q_rows.double().gather(-1, idx)[:, 0]
    acc = (u.double() * q_sel < p_sel).tolist()
    k = 0
    for a in acc:
        if not a:
            break
        k += 1
    return k


def greedy_prefix(p_rows, drafts) -> int:
    """Number of leading drafts equal to the argmax of their target row."""
    n = int(drafts.numel())
    if n == 0:
        return 0
    same = (p_rows.argmax(-1) == drafts).tolist()
    k = 0
    for s in same:
        if not s:
            break
        k += 1
    return k


def residual(p_row, q_row):
    """norm(max(p − q, 0)) in float64; (residual, mass_before_norm). A zero mass (p == q elementwise: no rejection can occur there) returns p."""
    r = torch.clamp(p_row.double() - q_row.double(), min=0.0)
    s = float(r.sum())
    if s <= 0.0:
        p = p_row.double()
        return p / p.sum(), 0.0
    return r / s, s


def acceptance_probability(p_row, q_row) -> float:
    """Σ_x min(p(x), q(x)) = P(a draft drawn from q is accepted against p)."""
    return float(torch.minimum(p_row.double(), q_row.double()).sum())


def emitted_distribution(p_row, q_row):
    """The distribution of the token this rule emits at one position (draft from q, accept test, residual on rejection) — equals p; used by the tests."""
    p, q = p_row.double(), q_row.double()
    acc = torch.minimum(p, q)                                   # q(x) · min(1, p(x)/q(x))
    rej_mass = 1.0 - float(acc.sum())
    res, _ = residual(p, q)
    return acc + rej_mass * res


def draw(probs_row, generator=None):
    """One index from a probability row (torch.multinomial on the row as float64)."""
    return int(torch.multinomial(probs_row.double(), 1, generator=generator))
