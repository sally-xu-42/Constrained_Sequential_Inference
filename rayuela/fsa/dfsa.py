from rayuela.base.semiring import Boolean, Semiring
from rayuela.base.symbol import Sym
from rayuela.fsa.state import State
from collections import defaultdict as dd
from copy import copy


# Deterministic FSA
class DFSA:
    # Construct from fsa
    def __init__(self, fsa):
        assert(fsa.deterministic)
        self.R: Semiring = fsa.R

        state_map = {q: State(i) for i, q in enumerate(fsa.Q)}

        self.Q: set = set(state_map.values())

        self.initial_states = set(state_map[q] for q in fsa.Q if fsa.λ[q] != self.R.zero)
        self.final_states = set(state_map[q] for q in fsa.Q if fsa.ρ[q] != self.R.zero)

        # delta[i][a] = (j, w)
        self.delta = dd(lambda: dd(lambda: (None, self.R.zero)))

        # Build deterministic transition arcs
        for i in fsa.Q:
            for a, j, w in fsa.arcs(i):
                self.delta[state_map[i]][a] = (state_map[j], w)

    def accept(self, tokens) -> bool:
        """ determines whether a string is in the language """
        assert isinstance(tokens, list)
        for cur in self.initial_states:
            for a in tokens:
                if not isinstance(a, Sym):
                    a = Sym(a)
                nxt, w = self.delta[cur][a]
                if nxt is not None:
                    cur = nxt
                else:  # no arc
                    return False

            if cur in self.final_states:
                return True

        return False
