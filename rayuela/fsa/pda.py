from frozendict import frozendict
from collections import defaultdict as dd
from numpy import isin
from paren import parenthesis_match

from rayuela.base.semiring import Boolean
from rayuela.base.symbol import Sym, ε

from rayuela.base.automaton import Automaton
from rayuela.fsa.state import State
from gcd.inference.constraints.parsing import util


class PDA(Automaton):
    def __init__(self, R=Boolean):
        from rayuela.fsa.fsa import FSA

		# DEFINITION
		# A weighted finite-state transducer is a 8-tuple <Σ, Δ, Q, F, I, δ, λ, ρ> where
		# • Σ is an alphabet of symbols;
		# • Δ is an alphabet of symbols;
		# • Q is a finite set of states;
		# • I ⊆ Q is a set of initial states;
		# • F ⊆ Q is a set of final states;
		# • δ is a finite relation Q × Σ × Δ × Q × R;
		# • λ is an initial weight function;
		# • ρ is a final weight function.

		# NOTATION CONVENTIONS
		# • single states (elements of Q) are denoted q
		# • multiple states not in sequence are denoted, p, q, r, ...
		# • multiple states in sequence are denoted i, j, k, ...
		# • symbols (elements of Σ and Δ) are denoted lowercase a, b, c, ...
		# • single weights (elements of R) are denoted w
		# • multiple weights (elements of R) are denoted u, v, w, ...

        # semiring
        # self.R = R

		# # alphabet of symbols
        # self.Sigma = set([])

		# # a finite set of states
        # self.Q = set([])

		# # transition function : Q × Σ × Q → R
        # self.δ = dd(lambda : dd(lambda : dd(lambda : self.R.zero)))

		# # initial weight function
        # self.λ = R.chart()

		# # final weight function
        # self.ρ = R.chart()
        self.Fsa = FSA(R=R)

    def get_Fsa(self):
        return self.Fsa
    
    def parenthesis_match(tokens: list) -> bool:
        stack = 0
        for i, token in enumerate(tokens):
            if util.is_token_open_paren(token):
                stack += 1
            elif util.is_token_close_paren(token):
                if stack == 0:
                    return False
                stack -= 1
                if stack == 0:
                    return all(not util.is_token_open_paren(t) and not util.is_token_close_paren(t) for t in tokens[i+1:])
            else: # terminal symbols
                pass
        return stack == 0
    
    def accept(self, tokens) -> bool:
        return PDA.parenthesis_match(tokens)
        print(PDA.parenthesis_match(tokens))
        print(self.Fsa.accept(tokens))
        return PDA.parenthesis_match(tokens) and self.Fsa.accept(tokens)

