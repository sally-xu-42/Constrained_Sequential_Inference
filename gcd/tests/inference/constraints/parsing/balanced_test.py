import unittest
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from gcd.inference.constraints.parsing import BalancedParenthesesConstraint


class TestBalancedParenthesesConstraint(unittest.TestCase):
    def test_balanced_parentheses_constraint(self):
        start, end = 1, 2
        nt, xx, close = 3, 4, 5
        token_to_key = {
            START_SYMBOL: 1,
            END_SYMBOL: 2,
            '(NT': 3,
            'XX': 4,
            ')': 5
        }
        automaton = BalancedParenthesesConstraint(8).build(None, token_to_key)

        assert automaton.accept([start, nt, close, end])
        assert automaton.accept([start, nt, nt, close, close, end])
        assert automaton.accept([start, nt, xx, nt, xx, close, xx, close, end])
        # Single root
        assert not automaton.accept([])
        assert not automaton.accept([start])
        assert not automaton.accept([end])
        assert not automaton.accept([start, end])
        assert not automaton.accept([start, nt, close, nt, close, end])
        assert not automaton.accept([start, close, nt, end])
        assert not automaton.accept([start, nt, nt, close, end])
        assert not automaton.accept([start, nt, end])
        # Balanced but too long
        assert not automaton.accept([start, nt, nt, nt, nt, xx, close, close, close, close, end])
