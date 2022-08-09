from rayuela.base.automaton import Automaton
from rayuela.base.semiring import Boolean, Semiring
from collections import defaultdict as dd
from typing import Tuple


# Deterministic FSA
class DFSA(Automaton):
    # Construct from fsa
    def __init__(self, fsa):
        from rayuela.fsa.fsa import FSA
        assert(isinstance(fsa, FSA) and fsa.deterministic)
        self.fsa = fsa  # save it for intersection
        self.R: Semiring = fsa.R

        state_map = {q: i for i, q in enumerate(fsa.Q)}  # from fsa state to dfsa

        self.Q: set = set(state_map.values())

        self.initial_states = set(state_map[q] for q in fsa.Q if fsa.λ[q] != self.R.zero)
        self.final_states = set(state_map[q] for q in fsa.Q if fsa.ρ[q] != self.R.zero)

        # delta[i][a] = (j, w)
        self.delta = dd(lambda: dd(lambda: (None, self.R.zero)))

        # Build deterministic transition arcs
        for i in fsa.Q:
            for a, j, w in fsa.arcs(i):
                self.delta[state_map[i]][a.sym] = (state_map[j], w)

    def accept(self, tokens) -> bool:
        """ determines whether a string is in the language """
        assert isinstance(tokens, list)
        for cur in self.initial_states:
            for a in tokens:
                nxt, w = self.delta[cur][a]
                if nxt is not None:
                    cur = nxt
                else:  # no arc
                    return False

            if cur in self.final_states:
                return True

        return False
    
    def get_start(self) -> int:
        assert(len(self.initial_states) == 1)
        return list(self.initial_states)[0]

    def get_valid_actions(self, state: int, stack: int) -> list:
        return list(self.delta[state].keys())
    
    def step(self, state: int, stack: int, action) -> Tuple[int, int]:  # returns to state, to stack
        nxt, w = self.delta[state][action]
        assert nxt is not None
        return nxt, stack
