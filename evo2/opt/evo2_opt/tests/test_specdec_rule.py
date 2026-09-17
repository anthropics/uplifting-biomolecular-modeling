"""gen/specdec/rule.py: the accept / residual / bonus rule against exact rational arithmetic and its defining property (the emitted token is
distributed as p) on recorded rows: a top-k-truncated p, a q with zeros, p == q, a one-hot p, a near-tie."""
import unittest
from fractions import Fraction

try:
    import torch
except ImportError:          # the rule runs on torch tensors; a python without torch has nothing to test here
    torch = None

if torch is not None:
    from evo2_opt.gen.specdec import rule as R

ROWS = {   # V = 6
    "topk_p":   ([0.55, 0.25, 0.15, 0.05, 0.0, 0.0], [0.40, 0.30, 0.10, 0.10, 0.05, 0.05]),
    "q_zeros":  ([0.30, 0.30, 0.20, 0.10, 0.05, 0.05], [0.70, 0.30, 0.0, 0.0, 0.0, 0.0]),
    "p_eq_q":   ([0.25, 0.25, 0.25, 0.125, 0.0625, 0.0625], [0.25, 0.25, 0.25, 0.125, 0.0625, 0.0625]),
    "one_hot":  ([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.20, 0.20, 0.20, 0.20, 0.10, 0.10]),
    "near_tie": ([0.5000001, 0.4999999, 0.0, 0.0, 0.0, 0.0], [0.4999999, 0.5000001, 0.0, 0.0, 0.0, 0.0]),
}


@unittest.skipIf(torch is None, "torch is not installed in this interpreter")
class TestRule(unittest.TestCase):
    def t(self, xs):
        return torch.tensor(xs, dtype=torch.float64)

    def test_accept_is_the_rational_comparison(self):
        for name, (p, q) in ROWS.items():
            P, Q = self.t(p), self.t(q)
            for d in range(6):
                if q[d] == 0.0:
                    continue                                            # d is drawn from q: q(d) > 0 always
                for u in (0.0, 0.1, 0.3333333333333333, 0.5, 0.74999999, 0.75, 0.9999999999):
                    got = R.accepted_prefix(P[None], Q[None], torch.tensor([d]), self.t([u]))
                    want = 1 if Fraction(u) * Fraction(q[d]) < Fraction(p[d]) else 0      # the values as given, compared exactly
                    self.assertEqual(got, want, (name, d, u))

    def test_prefix_stops_at_the_first_rejection(self):
        P = self.t([ROWS["topk_p"][0]] * 3); Q = self.t([ROWS["topk_p"][1]] * 3)
        drafts = torch.tensor([0, 4, 0])                                        # p(4) = 0: always rejected at position 2
        self.assertEqual(R.accepted_prefix(P, Q, drafts, self.t([0.0, 0.0, 0.0])), 1)
        self.assertEqual(R.accepted_prefix(P, Q, torch.tensor([0, 0, 0]), self.t([0.99, 0.99, 0.99])), 3)   # p/q = 1.375 > u
        self.assertEqual(R.accepted_prefix(P[:0], Q[:0], torch.tensor([], dtype=torch.long), self.t([])), 0)

    def test_residual(self):
        for name, (p, q) in ROWS.items():
            res, mass = R.residual(self.t(p), self.t(q))
            want = [max(Fraction(a) - Fraction(b), 0) for a, b in zip(p, q)]
            s = sum(want)
            self.assertAlmostEqual(mass, float(s), places=15, msg=name)
            if s == 0:
                self.assertEqual(name, "p_eq_q")
                self.assertTrue(torch.equal(res, self.t(p) / self.t(p).sum()))     # p == q: the residual is p itself (no rejection can occur)
            else:
                for x in range(6):
                    self.assertAlmostEqual(float(res[x]), float(want[x] / s), places=15, msg=(name, x))
            self.assertAlmostEqual(float(res.sum()), 1.0, places=14)

    def test_emitted_token_is_distributed_as_p(self):
        for name, (p, q) in ROWS.items():
            em = R.emitted_distribution(self.t(p), self.t(q))
            for x in range(6):
                self.assertAlmostEqual(float(em[x]), p[x], places=14, msg=(name, x))
            alpha = R.acceptance_probability(self.t(p), self.t(q))
            self.assertAlmostEqual(alpha, float(sum(min(Fraction(a), Fraction(b)) for a, b in zip(p, q))), places=15, msg=name)

    def test_greedy(self):
        P = self.t([ROWS["one_hot"][0], ROWS["topk_p"][0], ROWS["q_zeros"][0]])
        self.assertEqual(R.greedy_prefix(P, torch.tensor([2, 0, 5])), 2)
        self.assertEqual(R.greedy_prefix(P, torch.tensor([1, 0, 0])), 0)
        self.assertEqual(R.greedy_prefix(P, torch.tensor([2, 0, 0])), 3)

    def test_draw_uses_torch_generator(self):
        g1 = torch.Generator().manual_seed(5); g2 = torch.Generator().manual_seed(5)
        row = self.t(ROWS["topk_p"][0])
        a = [R.draw(row, g1) for _ in range(64)]; b = [R.draw(row, g2) for _ in range(64)]
        self.assertEqual(a, b)
        self.assertTrue(all(row[i] > 0 for i in a))


if __name__ == "__main__":
    unittest.main()
