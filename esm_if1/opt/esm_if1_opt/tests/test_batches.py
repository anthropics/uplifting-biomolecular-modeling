"""The driver's row batching (pure function): design order, each backbone's samples adjacent, at most B rows, one key per batch (the
backbone length; on the multichain route the (concatenated length, designed length) pair), and the multichain route's token pattern."""
import pytest

from esm_if1_opt.batched import batches, complex_pattern


def test_batches():
    assert list(batches([5, 5, 7], 2, 64)) == [[(0, 0), (0, 1), (1, 0), (1, 1)], [(2, 0), (2, 1)]]
    assert list(batches([5, 5, 7], 2, 3)) == [[(0, 0), (0, 1), (1, 0)], [(1, 1)], [(2, 0), (2, 1)]]
    assert list(batches([9], 4, 1)) == [[(0, 0)], [(0, 1)], [(0, 2)], [(0, 3)]]
    assert list(batches([], 3, 8)) == []
    rows = [r for b in batches([3, 3, 3, 4, 3], 3, 4) for r in b]
    assert rows == [(i, s) for i in range(5) for s in range(3)]                  # every (backbone, sample) row exactly once, in design order


def test_batches_on_the_multichain_key():
    keys = [(30, 10), (30, 10), (30, 12), (41, 12)]                                   # two complexes alike, then the designed length changes, then the total
    assert list(batches(keys, 2, 64)) == [[(0, 0), (0, 1), (1, 0), (1, 1)], [(2, 0), (2, 1)], [(3, 0), (3, 1)]]
    assert [len(b) for b in batches([(30, 10)], 8, 3)] == [3, 3, 2]                    # one complex, 8 sequences, 3 rows per forward


def test_complex_pattern_is_upstreams():
    """multichain_util.sample_sequence_in_complex: <mask> over the designed chain (first in the concatenation), <pad> over the rest."""
    assert complex_pattern(7, 3) == ['<mask>', '<mask>', '<mask>', '<pad>', '<pad>', '<pad>', '<pad>']
    assert complex_pattern(4, 4) == ['<mask>'] * 4 and complex_pattern(2, 0) == ['<pad>', '<pad>']
    with pytest.raises(ValueError):
        complex_pattern(3, 5)
