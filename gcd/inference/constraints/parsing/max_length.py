from collections import defaultdict
import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Dict

from rayuela.base.automaton import Automaton
from rayuela.fsa.dfsa import DFSA
from rayuela.fsa.fsa import FSA, State
from gcd.inference.constraints import Constraint
from gcd.inference.constraints.parsing import util


@Constraint.register('max-length')
class MaxLengthConstraint(Constraint):
    cache: Dict[str, Dict[int, Automaton]] = defaultdict(lambda: {})
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    def build(self,
              input_tokens: torch.Tensor,
              token_to_key: Dict[str, int],
              dict_hash: str = None, *args, **kwargs) -> Automaton:
        if dict_hash is None: dict_hash = util.hash_dict(token_to_key)
        dfsa = self.cache[self.max_length].get(dict_hash)
        if dfsa is None:
            print(f'Compiling a MaxLengthConstraint with max_length {self.max_length} and size {len(token_to_key)} vocab ...')
            fsa = FSA()

            states = [State(i) for i in range(self.max_length + 3)]
            fsa.add_states(states)
            fsa.set_I(states[0])
            fsa.set_F(states[-1])

            # Add starting and ending transitions
            start, end = token_to_key[START_SYMBOL], token_to_key[END_SYMBOL]
            fsa.add_arc(states[0], start, states[1])
            fsa.add_arc(states[-2], end, states[-1])

            # Add all of the intermediate transitions
            for state1, state2 in zip(states[1:-2], states[2:-1]):
                for token, key in token_to_key.items():
                    if util.is_stack_token(token) or token in [START_SYMBOL, END_SYMBOL]:
                        continue

                    fsa.add_arc(state1, key, state2)

            # Add a transition from the intermediate states to the end
            for state in states[1:-2]:
                fsa.add_arc(state, end, states[-1])

            # Finalize
            dfsa = fsa.compile()
            self.cache[self.max_length][dict_hash] = dfsa
        
        return dfsa

    def get_name(self) -> str:
        return f'max-length-{self.max_length}'
