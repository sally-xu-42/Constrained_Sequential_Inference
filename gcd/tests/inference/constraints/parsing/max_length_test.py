import unittest
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from gcd.inference.constraints.parsing import MaxLengthConstraint


class TestMaxLengthConstraint(unittest.TestCase):
    def test_max_length_constraint(self):
        start, end = 1, 2
        nt, xx, close = 3, 4, 5
        token_to_key = {
            START_SYMBOL: start,
            END_SYMBOL: end,
            '(NT': nt,
            'XX': xx,
            ')': close
        }
        automaton = MaxLengthConstraint(5).build(None, token_to_key)

        assert automaton.accept([start, end])
        assert automaton.accept([start, nt, close, end])
        assert automaton.accept([start, nt, nt, close, close, end])
        assert automaton.accept([start, nt, xx, nt, xx, close, end])

        assert not automaton.accept([start, nt, close, nt, close, close, nt, end])
        assert not automaton.accept([start, nt, xx, nt, xx, close, close, xx, end])
