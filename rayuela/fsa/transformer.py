from collections import defaultdict as dd
from itertools import chain, product
from sys import float_repr_style
from frozendict import frozendict

from rayuela.base.automaton import Automaton
from rayuela.base.misc import epsilon_filter
from rayuela.base.symbol import ε, ε_1, ε_2
from rayuela.fsa.fsa import FSA
from rayuela.fsa.dfsa import DFSA
from rayuela.fsa.pda import PDA
from rayuela.fsa.state import PairState, PowerState, State
from rayuela.fsa.pathsum import Pathsum, Strategy


class Transformer:

    def trim(fsa):
        raise NotImplementedError

    def _powerarcs(fsa, Q):
        """ This helper method group outgoing arcs for determinization. """

        symbol2arcs, unnormalized_residuals = dd(set), fsa.R.chart()

        for q, old_residual in Q.residuals.items():
            for a, p, w in fsa.arcs(q):
                symbol2arcs[a].add(p)
                unnormalized_residuals[(a, p)] += old_residual * w

        for a, ps in symbol2arcs.items():
            normalizer = fsa.R.zero
            for p in ps:
                normalizer += unnormalized_residuals[(a, p)]
            residuals = {p : ~normalizer * unnormalized_residuals[(a, p)] for p in ps}

            yield a, PowerState(residuals), normalizer

    def push(fsa):
        from rayuela.fsa.pathsum import Strategy
        W = Pathsum(fsa).backward(Strategy.LEHMANN)
        pfsa = Transformer._push(fsa, W)
        assert pfsa.pushed # sanity check
        return pfsa

    def _push(fsa, V):
        """
        Mohri (2001)'s weight pushing algorithm. See Eqs 1, 2, 3.
        Link: https://www.isca-speech.org/archive_v0/archive_papers/eurospeech_2001/e01_1603.pdf.
        """

        pfsa = fsa.spawn()
        for i in fsa.Q:
            pfsa.set_I(i, fsa.λ[i] * V[i])
            pfsa.set_F(i, ~V[i] * fsa.ρ[i])
            for a, j, w in fsa.arcs(i):
                pfsa.add_arc(i, a, j, ~V[i] * w * V[j])

        return pfsa

    def _eps_partition(fsa):
        """ partition fsa into two (one with eps arcs and one with all others) """

        E = fsa.spawn()
        N = fsa.spawn(keep_init=True, keep_final=True)

        for q in fsa.Q:
            E.add_state(q)
            N.add_state(q)

        for i in fsa.Q:
            for a, j, w in fsa.arcs(i):
                if a == ε:
                    E.add_arc(i, a, j, w)
                else:
                    N.add_arc(i, a, j, w)

        return N, E

    def epsremoval(fsa):

        # note that N keeps same initial and final weights
        N, E = Transformer._eps_partition(fsa)
        W = Pathsum(E).lehmann(zero=False)

        for i in fsa.Q:
            for a, j, w in fsa.arcs(i, no_eps=True):
                for k in fsa.Q:
                    N.add_arc(i, a, k, w * W[j, k])

        # additional initial states
        for i, j in product(fsa.Q, repeat=2):
            N.add_I(j, fsa.λ[i] * W[i, j])


        return N

    def intersect(a1: Automaton, a2: Automaton) -> Automaton:
        """Router for intersecting two automata"""
        if isinstance(a1, FSA) and isinstance(a2, FSA):
            return Transformer._fsa_fsa_intersect(a1, a2)

        if isinstance(a1, DFSA) and isinstance(a2, DFSA):
            return Transformer._dfsa_dfsa_intersect(a1, a2)
        if isinstance(a1, DFSA) and isinstance(a2, FSA):
            return Transformer._dfsa_dfsa_intersect(a1, a2.compile())
        if isinstance(a1, FSA) and isinstance(a2, DFSA):
            return Transformer._dfsa_dfsa_intersect(a1.compile(), a2)

        if isinstance(a2, PDA): a1, a2 = a2, a1
        assert(isinstance(a1, PDA)), f"Unkown automaton type {type(a1)}"

        if isinstance(a2, PDA):
            return Transformer._pda_pda_intersect(a1, a2)
        if isinstance(a2, DFSA):
            return Transformer._pda_dfsa_intersect(a1, a2)
        assert(isinstance(a2, FSA)), f"Unkown automaton type {type(a2)}"
        return Transformer._pda_dfsa_intersect(a1, a2.compile())
    
    def _fsa_fsa_intersect(f1: FSA, f2: FSA) -> FSA:
        # the two machines need to be in the same semiring
        assert f1.R == f2.R

        # add initial states
        product_fsa = FSA(R=f1.R)
        for (q1, w1), (q2, w2) in product(f1.I, f2.I):
            product_fsa.add_I(PairState(q1, q2), w=w1 * w2)
        
        self_initials = {q: w for q, w in f1.I}
        fsa_initials = {q: w for q, w in f2.I}

        visited = set([(i1, i2, State('0')) for i1, i2 in product(self_initials, fsa_initials)])
        stack = [(i1, i2, State('0')) for i1, i2 in product(self_initials, fsa_initials)]

        self_finals = {q: w for q, w in f1.F}
        fsa_finals = {q: w for q, w in f2.F}

        while stack:
            q1, q2, qf = stack.pop()

            E1 = [(a if a != ε else ε_2, j, w) for (a, j, w) in f1.arcs(q1)] + \
                            [(ε_1, q1, f1.R.one)]
            E2 = [(a if a != ε else ε_1, j, w) for (a, j, w) in f2.arcs(q2)] + \
                            [(ε_2, q2, f1.R.one)]

            M = [((a1, j1, w1), (a2, j2, w2))
                    for (a1, j1, w1), (a2, j2, w2) in product(E1, E2)
                    if epsilon_filter(a1, a2, qf) != State('⊥')]

            for (a1, j1, w1), (a2, j2, w2) in M:

                product_fsa.set_arc(PairState(q1, q2), a1, PairState(j1, j2), w=w1*w2)

                _qf = epsilon_filter(a1, a2, qf)
                if (j1, j2, _qf) not in visited:
                    stack.append((j1, j2, _qf))
                    visited.add((j1, j2, _qf))

            # final state handling
            if q1 in self_finals and q2 in fsa_finals:
                product_fsa.add_F(PairState(q1, q2), w=self_finals[q1] * fsa_finals[q2])

        return product_fsa
    
    def _dfsa_dfsa_intersect(d1: DFSA, d2: DFSA) -> DFSA:
        print(f"Intersecting a DFSA with {d1.num_states} and a DFSA with {d2.num_states} ...")
        product_fsa = FSA()
        product_fsa.set_I(PairState(d1.initial_state, d2.initial_state)) 

        stack = [(d1.initial_state, d2.initial_state)]
        visited = {(d1.initial_state, d2.initial_state)}

        while stack:
            i1, i2 = stack.pop()
            def common_arcs(arcs1, arcs2):
                # res = []
                if len(arcs1) <= len(arcs2):
                    for a, j1 in arcs1.items():
                        j2 = arcs2.get(a)
                        if j2 is not None:
                            yield a, j1, j2
                            # res.append((a, j1, j2))
                else:
                    for a, j2 in arcs2.items():
                        j1 = arcs1.get(a)
                        if j1 is not None:
                            yield a, j1, j2
                            # res.append((a, j1, j2))
                # for b in d1.Sigma:
                #     assert sum(1 for a, j1, j2 in res if a == b) <= 1
                # return res
            
            for a, j1, j2 in common_arcs(d1.delta[i1], d2.delta[i2]):
                product_fsa.add_arc(PairState(i1, i2), a, PairState(j1, j2))
                if (j1, j2) not in visited:
                    stack.append((j1, j2))
                    visited.add((j1, j2))
            
            if i1 == d1.final_state and i2 == d2.final_state:
                product_fsa.set_F(PairState(i1, i2))
        
        return product_fsa.compile()


    def _pda_dfsa_intersect(pda: PDA, dfsa: DFSA) -> PDA:
        return PDA(pda.token_to_key, Transformer._dfsa_dfsa_intersect(pda.dfsa, dfsa))
    
    def _pda_pda_intersect(p1: PDA, p2: PDA) -> PDA:
        assert(p1.token_to_key == p2.token_to_key)
        return PDA(p1.token_to_key, Transformer._dfsa_dfsa_intersect(p1.dfsa, p2.dfsa))
