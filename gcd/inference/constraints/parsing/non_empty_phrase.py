import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Dict

from rayuela.base.automaton import Automaton
from rayuela.fsa.fsa import FSA, State
from gcd.inference.constraints import Constraint
from gcd.inference.constraints.parsing import util


@Constraint.register('non-empty-phrase')
class NonEmptyPhraseConstraint(Constraint):
    def __init__(self) -> None:
        super().__init__()

    def build(self,
              input_tokens: torch.Tensor,
              token_to_key: Dict[str, int], *args, **kwargs) -> Automaton:
        fsa = FSA()

        # To write this automaton, we will first write an automaton
        # to match emtpy phrases and then negate it
        s0 = State(0)
        s1 = State(1)
        s2 = State(2)
        s3 = State(3)

        # Set the start and final states
        fsa.set_I(s0)
        fsa.set_F(s3)

        # Set the transitions from and to the start and final
        fsa.add_arc(s0, START_SYMBOL, s1)
        fsa.add_arc(s1, END_SYMBOL, s3)
        fsa.add_arc(s2, END_SYMBOL, s3)

        for token in token_to_key.keys():
            if util.is_stack_token(token) or token in [START_SYMBOL, END_SYMBOL]:
                continue

            # Loop around s1 with any token but an open paren
            if not util.is_token_open_paren(token):
                fsa.add_arc(s1, token, s1)

            if util.is_token_open_paren(token):
                # Go from s1 to s2 if there's an open paren
                fsa.add_arc(s1, token, s2)

                # Loop on s2 if there's an open paren
                fsa.add_arc(s2, token, s2)

            # Go back from s2 to s1 on anything but a close paren or open paren
            if not util.is_token_close_paren(token) and not util.is_token_open_paren(token):
                fsa.add_arc(s2, token, s1)

        # Finalize
        return fsa.compile()

    def get_name(self) -> str:
        return 'non-empty-phrase'
