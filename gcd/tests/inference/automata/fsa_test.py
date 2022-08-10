import unittest

from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State


class TestFSA(unittest.TestCase):
    def setUp(self):
        # Build an FSA that accepts "123" and "124"
        self.fsa = FSA() 
        states = [State(i) for i in range(4)]
        self.s0, self.s1, self.s2, self.s3 = states
        self.fsa.add_states(states)
        self.a1, self.a2, self.a3, self.a4, self.a5 = [i for i in range(1, 6)]
        self.fsa.set_I(self.s0)
        self.fsa.set_F(self.s3)
        self.fsa.add_arc(self.s0, self.a1, self.s1)
        self.fsa.add_arc(self.s1, self.a2, self.s2)
        self.fsa.add_arc(self.s2, self.a3, self.s3)
        self.fsa.add_arc(self.s2, self.a4, self.s3)
        self.fsa = self.fsa.compile()
        self.s0, self.s1, self.s2, self.s3 = 2, 1, 3, 0  # ! bad

    def test_accepts(self):
        assert self.fsa.accept([self.a1, self.a2, self.a3])
        assert self.fsa.accept([self.a1, self.a2, self.a4])
        assert not self.fsa.accept([self.a1, self.a2])
        assert not self.fsa.accept([self.a1, self.a2, self.a5])
        assert not self.fsa.accept([])

    def test_step(self):
        s0 = self.fsa.get_start()
        assert self.fsa.get_valid_actions(s0, 0) == [self.a1]
        s1, _  = self.fsa.step(s0, 0, self.a1)
        assert self.fsa.get_valid_actions(s1, 0) == [self.a2]
        s2, _  = self.fsa.step(s1, 0, self.a2)
        assert self.fsa.get_valid_actions(s2, 0) == [self.a3, self.a4]
        s3, _  = self.fsa.step(s2, 0, self.a3)
        assert self.fsa.get_valid_actions(s3, 0) == []
        s4, _  = self.fsa.step(s2, 0, self.a4)
        assert s3 == s4
