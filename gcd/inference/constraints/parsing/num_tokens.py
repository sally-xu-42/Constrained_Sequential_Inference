from collections import defaultdict
import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Dict

from rayuela.base.automaton import Automaton
from rayuela.fsa.dfsa import DFSA
from rayuela.fsa.fsa import FSA, State
from gcd.inference.constraints import Constraint
from gcd.inference.constraints.parsing import util


@Constraint.register('num-tokens')
class NumTokensConstraint(Constraint):
    cache: Dict[int, Dict[str, DFSA]] = defaultdict(lambda: {})
    # cache = {} cache[123] =2
    # cache      
    def build(self,
              input_tokens: torch.Tensor,
              token_to_key: Dict[str, int],
              dict_hash: str, *args, **kwargs) -> Automaton:
        batch_size, num_tokens = input_tokens.size()
        assert batch_size == 1, batch_size
        num_tokens -= 2  # <bos>, <eos>
        dfsa = self.cache[num_tokens].get(dict_hash)
        if dfsa is None:
            print(f'Compiling a NumTokensConstraint with {num_tokens} tokens and size {len(token_to_key)} vocab...')
            fsa = FSA()

            # There is a state for seeing 0, 1, ..., num_tokens tokens plus one
            # start state and one final state
            states = [State(i) for i in range(num_tokens + 3)]
            fsa.add_states(states)

            # Set the start and final states
            fsa.set_I(states[0])
            fsa.set_F(states[-1])

            # Set the transitions from and to the start and final
            start, end = token_to_key[START_SYMBOL], token_to_key[END_SYMBOL]
            fsa.add_arc(states[0], start, states[1])
            fsa.add_arc(states[-2], end, states[-1])

            # For the middle states, set the self loop for any symbol
            # except for a preterminal, start, end, open, or close.
            for state in states[1:-1]:
                for token, key in token_to_key.items():
                    if util.is_stack_token(token) or token in [START_SYMBOL, END_SYMBOL]:
                        continue

                    # Now we only have vocabulary items left
                    if util.is_token_preterminal(token):
                        continue
                        
                    fsa.add_arc(state, key, state)

            # Add a transition between the intermediate states using a preterminal
            for state1, state2 in zip(states[1:-2], states[2:-1]):
                for token, key in token_to_key.items():
                    if util.is_token_preterminal(token):
                        fsa.add_arc(state1, key, state2)

            # Finalize
            dfsa = fsa.compile()
            self.cache[num_tokens][dict_hash] = dfsa
        
        return dfsa

    def get_name(self) -> str:
        return 'num-tokens'
