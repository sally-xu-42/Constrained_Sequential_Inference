import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Dict

from rayuela.fsa.fsa import FSA, State
from gcd.inference.constraints import Constraint
from gcd.inference.constraints.parsing import util


@Constraint.register('num-tokens')
class NumTokensConstraint(Constraint):
    def build(self,
              input_tokens: torch.Tensor,
              token_to_key: Dict[str, int], *args, **kwargs) -> FSA:
        batch_size, num_tokens = input_tokens.size()
        assert batch_size == 1, batch_size
        num_tokens -= 2  # <bos>, <eos>

        fsa = FSA()

        # There is a state for seeing 0, 1, ..., num_tokens tokens plus one
        # start state and one final state
        states = [State(i) for i in range(num_tokens + 3)]
        fsa.add_states(states)

        # Set the start and final states
        fsa.set_I(states[0])
        fsa.set_F(states[-1])

        # Set the transitions from and to the start and final
        fsa.add_arc(states[0], START_SYMBOL, states[1])
        fsa.add_arc(states[-2], END_SYMBOL, states[-1])

        # For the middle states, set the self loop for any symbol
        # except for a preterminal, start, end, open, or close.
        for state in states[1:-1]:
            for token in token_to_key.keys():
                if util.is_stack_token(token) or token in [START_SYMBOL, END_SYMBOL]:
                    continue

                # Now we only have vocabulary items left
                if util.is_token_preterminal(token):
                    continue
                    
                fsa.add_arc(state, token, state)

        # Add a transition between the intermediate states using a preterminal
        for state1, state2 in zip(states[1:-2], states[2:-1]):
            for token in token_to_key.keys():
                if util.is_token_preterminal(token):
                    fsa.add_arc(state1, token, state2)

        # Finalize
        fsa.compile()
        return fsa

    def get_name(self) -> str:
        return 'num-tokens'
