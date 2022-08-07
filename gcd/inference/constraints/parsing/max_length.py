import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Dict

from rayuela.fsa.fsa import FSA, DFSA, State
from gcd.inference.constraints import Constraint
from gcd.inference.constraints.parsing import util


@Constraint.register('max-length')
class MaxLengthConstraint(Constraint):
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    def build(self,
              input_tokens: torch.Tensor,
              token_to_key: Dict[str, int], *args, **kwargs) -> DFSA:
        fsa = FSA()

        states = [State(i) for i in range(self.max_length + 3)]
        fsa.add_states(states)
        fsa.set_I(states[0])
        fsa.set_F(states[-1])

        # Add starting and ending transitions
        fsa.add_arc(states[0], START_SYMBOL, states[1])
        fsa.add_arc(states[-2], END_SYMBOL, states[-1])

        # Add all of the intermediate transitions
        for state1, state2 in zip(states[1:-2], states[2:-1]):
            for token in token_to_key.keys():
                if util.is_stack_token(token) or token in [START_SYMBOL, END_SYMBOL]:
                    continue

                fsa.add_arc(state1, token, state2)

        # Add a transition from the intermediate states to the end
        for state in states[1:-2]:
            fsa.add_arc(state, END_SYMBOL, states[-1])

        # Finalize
        return fsa.compile()

    def get_name(self) -> str:
        return f'max-length-{self.max_length}'
