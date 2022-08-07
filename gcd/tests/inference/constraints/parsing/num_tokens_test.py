import torch
import unittest
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from gcd.inference.constraints.parsing import NumTokensConstraint


class TestNumTokensConstraint(unittest.TestCase):
    def test_num_tokens_constraint(self):
        num_tokens = 3
        input_tokens = torch.zeros(1, num_tokens + 2)

        start, end = START_SYMBOL, END_SYMBOL
        nt, xx, close = '(NT', 'XX', ')'
        token_to_key = {
            START_SYMBOL: 1,
            END_SYMBOL: 2,
            '(NT': 3,
            'XX': 4,
            ')': 5
        }
        automaton = NumTokensConstraint().build(input_tokens, token_to_key)

        assert automaton.accept([start, xx, xx, xx, end])
        assert automaton.accept([start, nt, xx, close, close, xx, nt, xx, end])
        assert not automaton.accept([start, xx, xx, end])
        assert not automaton.accept([start, xx, nt, close, xx, end])
