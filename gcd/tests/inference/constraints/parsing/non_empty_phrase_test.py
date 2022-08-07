import unittest
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from gcd.inference.constraints.parsing import NonEmptyPhraseConstraint


class TestNonEmptyPhraseConstraint(unittest.TestCase):
    def test_non_empty_phrase_constraint(self):
        start, end = START_SYMBOL, END_SYMBOL
        nt, xx, close = '(NT', 'XX', ')'
        token_to_key = {
            START_SYMBOL: 1,
            END_SYMBOL: 2,
            '(NT': 3,
            'XX': 4,
            ')': 5
        }
        automaton = NonEmptyPhraseConstraint().build(None, token_to_key)

        assert automaton.accept([start, nt, xx, close, end])
        assert automaton.accept([start, nt, nt, xx, close, nt, xx, close, close, end])
        assert automaton.accept([start, nt, nt, end])
        assert not automaton.accept([start, nt, close, end])
        assert not automaton.accept([start, nt, nt, close, close, end])
        assert not automaton.accept([start, nt, xx, close, nt, close, end])
