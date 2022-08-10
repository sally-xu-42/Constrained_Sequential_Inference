import torch
import unittest
from allennlp.common.util import END_SYMBOL, START_SYMBOL

from rayuela.fsa.fsa import FSA
from rayuela.fsa.pda import PDA
from rayuela.fsa.state import State


class TestPDA(unittest.TestCase):
    def setUp(self):
        # Build a PDA that accepts "<bos> a^n b^k c^n <eos>" where
        # n >= 1 and k >= 0
        start, end = 1, 2
        a, b, c = 3, 4, 5
        self.token_to_key = {
            START_SYMBOL: start,
            END_SYMBOL: end,
            '(': a,
            'x': b,
            ')': c
        }
        self.bos, self.eos = start, end
        self.a, self.b, self.c = a, b, c

        fsa = FSA()
        s = [State(i) for i in range(5)]
        fsa.add_states(s)
        fsa.add_arc(s[0], start, s[1])
        fsa.add_arc(s[1], a, s[1])
        fsa.add_arc(s[1], a, s[2])
        fsa.add_arc(s[2], b, s[2])
        fsa.add_arc(s[2], c, s[3])
        fsa.add_arc(s[3], c, s[3])
        fsa.add_arc(s[3], end, s[4])
        fsa.set_I(s[0])
        fsa.set_F(s[4])

        self.pda = PDA(self.token_to_key, fsa.compile())

    def test_accepts(self):
        bos, eos = self.bos, self.eos
        a, b, c = self.a, self.b, self.c
        assert self.pda.accept([bos, a, b, c, eos])
        assert self.pda.accept([bos, a, a, b, c, c, eos])
        assert self.pda.accept([bos, a, a, a, b, b, c, c, c, eos])
        assert self.pda.accept([bos, a, c, eos])
        assert self.pda.accept([bos, a, a, c, c, eos])
        assert self.pda.accept([bos, a, a, a, c, c, c, eos])

        assert not self.pda.accept([bos, a, a, b, c, c])
        assert not self.pda.accept([a, a, b, c, c, eos])
        assert not self.pda.accept([a, a, b, b, c])
        assert not self.pda.accept([a, a, c, c, c])

    def test_step1(self):
        bos, eos = self.bos, self.eos
        a, b, c = self.a, self.b, self.c
        pda = self.pda
        state, stack = pda.get_start(), 0
        assert sorted(pda.get_valid_actions(state, stack)) == [bos]
        state, stack = pda.step(state, stack, bos)
        assert sorted(pda.get_valid_actions(state, stack)) == [a]
        state, stack = pda.step(state, stack, a)
        assert sorted(pda.get_valid_actions(state, stack)) == [a, b, c]
        state, stack = pda.step(state, stack, b)
        assert sorted(pda.get_valid_actions(state, stack)) == [b, c]
        state, stack = pda.step(state, stack, b)
        assert sorted(pda.get_valid_actions(state, stack)) == [b, c]
        state, stack = pda.step(state, stack, c)
        assert sorted(pda.get_valid_actions(state, stack)) == [eos]
        state, stack = pda.step(state, stack, eos)
        assert sorted(pda.get_valid_actions(state, stack)) == []

    def test_step2(self):
        bos, eos = self.bos, self.eos
        a, b, c = self.a, self.b, self.c
        pda = self.pda
        state, stack = pda.get_start(), 0
        assert sorted(pda.get_valid_actions(state, stack)) == [bos]
        state, stack = pda.step(state, stack, bos)
        assert sorted(pda.get_valid_actions(state, stack)) == [a]
        state, stack = pda.step(state, stack, a)
        assert sorted(pda.get_valid_actions(state, stack)) == [a, b, c]
        state, stack = pda.step(state, stack, a)
        assert sorted(pda.get_valid_actions(state, stack)) == [a, b, c]
        state, stack = pda.step(state, stack, b)
        assert sorted(pda.get_valid_actions(state, stack)) == [b, c]
        state, stack = pda.step(state, stack, c)
        assert sorted(pda.get_valid_actions(state, stack)) == [c]
        state, stack = pda.step(state, stack, c)
        assert sorted(pda.get_valid_actions(state, stack)) == [eos]
        state, stack = pda.step(state, stack, eos)
        assert sorted(pda.get_valid_actions(state, stack)) == []

    def test_intersection(self):
        bos, eos = self.bos, self.eos
        a, b, c = self.a, self.b, self.c

        s = [State(i) for i in range(4)]
        fsa = FSA()
        fsa.add_states(s)
        fsa.add_arc(s[0], bos, s[1])
        fsa.add_arc(s[1], a, s[1])
        fsa.add_arc(s[1], b, s[1])
        fsa.add_arc(s[1], c, s[2])
        fsa.add_arc(s[2], a, s[2])
        fsa.add_arc(s[2], b, s[2])
        fsa.add_arc(s[2], eos, s[3])
        fsa.set_I(s[0])
        fsa.set_F(s[3])

        assert fsa.accept([bos, a, b, c, a, b, b, eos])
        assert fsa.accept([bos, c, eos])
        assert not fsa.accept([bos, a, a, b, c, c, eos])

        isect = self.pda.intersect(fsa)
        
        assert isect.accept([bos, a, b, c, eos])
        assert isect.accept([bos, a, c, eos])
        assert not isect.accept([bos, eos])
        assert not isect.accept([])
        assert not isect.accept([eos])
        assert not isect.accept([bos])
        assert not isect.accept([bos, c, eos])
        assert not isect.accept([bos, a, a, b, c, c, eos])
        assert not isect.accept([bos, a, a, b, c, a, b, eos])
