import unittest
from allennlp.common.util import END_SYMBOL, START_SYMBOL

from rayuela.fsa.fsa import FSA
from rayuela.fsa.pda import PDA
from rayuela.fsa.state import State


class TestUtil(unittest.TestCase):
    def test_intersect_fsa_fsa(self):
        # Build an FSA that accepts "123" and "124" and another that
        # only accepts "123"
        a1, a2, a3, a4, a5 = 1, 2, 3, 4, 5
        states = [State(i) for i in range(4)]
        s0, s1, s2, s3 = states

        fsa1 = FSA()
        fsa1.add_states(states)
        fsa1.set_I(s0)
        fsa1.set_F(s3)
        fsa1.add_arc(s0, a1, s1)
        fsa1.add_arc(s1, a2, s2)
        fsa1.add_arc(s2, a3, s3)
        fsa1.add_arc(s2, a4, s3)
        fsa1 = fsa1.compile()

        fsa2 = FSA()
        fsa2.add_states(states)
        fsa2.set_I(s0)
        fsa2.set_F(s3)
        fsa2.add_arc(s0, a1, s1)
        fsa2.add_arc(s1, a2, s2)
        fsa2.add_arc(s2, a3, s3)
        fsa2 = fsa2.compile()

        intersection = fsa1.intersect(fsa2)
        # Check the language
        assert intersection.accept([a1, a2, a3])
        assert not intersection.accept([])
        assert not intersection.accept([a1, a2])
        assert not intersection.accept([a1, a2, a3, a1])
        assert not intersection.accept([a2])

    def test_intersect_fsa_pda(self):
        # Create a PDA that accepts "a^n b^n" and an FSA that accepts
        # inputs >= length 5
        start, end = 1, 2
        a, b = 5, 6
        token_to_key = {
            START_SYMBOL: start,
            END_SYMBOL: end,
            '(NT': a,
            ')': b
        }

        pda = PDA(token_to_key)

        assert pda.accept([start, a, b, end])
        assert pda.accept([start, a, a, b, b, end])
        assert pda.accept([start, a, a, a, b, b, b, end])
        assert pda.accept([start, a, a, a, a, b, b, b, b, end])

        fsa = FSA()
        s = [State(i) for i in range(8)]
        fsa.add_states(s)
        fsa.set_I(s[0])
        fsa.set_F(s[7])

        # Use the PDA's symbol table
        fsa.add_arc(s[0], start, s[1])
        for i in range(1, 6):
            fsa.add_arc(s[i], a, s[i+1])
            fsa.add_arc(s[i], b, s[i+1])
        fsa.add_arc(s[6], a, s[6])
        fsa.add_arc(s[6], b, s[6])
        fsa.add_arc(s[6], end, s[7])
        fsa = fsa.compile()

        intersection = fsa.intersect(pda)
        assert not intersection.accept([start, a, b, end])
        assert not intersection.accept([start, a, a, b, b, end])
        assert intersection.accept([start, a, a, a, b, b, b, end])
        assert intersection.accept([start, a, a, a, a, b, b, b, b, end])
