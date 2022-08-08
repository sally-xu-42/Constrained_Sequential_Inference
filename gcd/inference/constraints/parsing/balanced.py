import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Dict

# from gcd.inference.automata import PDA
from rayuela.fsa.pda import PDA
from rayuela.fsa.state import State
from gcd.inference.constraints import Constraint
from gcd.inference.constraints.parsing import util
from gcd.inference.constraints.parsing.util import \
    CLOSE_PAREN_SYMBOL, OPEN_PAREN_SYMBOL, \
    EMPTY_STACK_OPEN_SYMBOL, EMPTY_STACK_CLOSE_SYMBOL, is_token_open_paren


@Constraint.register('balanced-parens')
class BalancedParenthesesConstraint(Constraint):
    def __init__(self, max_length: int) -> None:
        from gcd.inference.constraints.parsing import MaxLengthConstraint
        self.max_length_constraint = MaxLengthConstraint(max_length)

    def build(self,
              input_tokens: torch.Tensor,
              token_to_key: Dict[str, int], *args, **kwargs) -> PDA:
        pda = PDA()

        # start_key = symbol_table[START_SYMBOL]
        # end_key = symbol_table[END_SYMBOL]
        # open_key = symbol_table[OPEN_PAREN_SYMBOL]
        # close_key = symbol_table[CLOSE_PAREN_SYMBOL]
        # empty_open_key = symbol_table[EMPTY_STACK_OPEN_SYMBOL]
        # empty_close_key = symbol_table[EMPTY_STACK_CLOSE_SYMBOL]

        states = [State(i) for i in range(5)]
        pda.get_Fsa().add_states(states)

        # Set the start and final states
        pda.get_Fsa().set_I(states[0])
        pda.get_Fsa().set_F(states[-1])

        # Add the start transition and empty stack push
        pda.get_Fsa().add_arc(State(0), START_SYMBOL, State(1))
        for token in token_to_key.keys():
            if util.is_token_open_paren(token):
                pda.get_Fsa().add_arc(State(1), token, State(2))

        # Emit any preterminal any number of times
        for token in token_to_key.keys():
            if util.is_stack_token(token) or token in [START_SYMBOL, END_SYMBOL]:
                continue
            pda.get_Fsa().add_arc(State(2), token, State(2))

        for token in token_to_key.keys():
            if util.is_token_close_paren(token):
                pda.get_Fsa().add_arc(State(2), token, State(3))

        # pda.get_Fsa().add_arc(State(2), CLOSE_PAREN_SYMBOL, State(3))
        # Get the last closing phrase
        # pda.get_Fsa().add_arc(State(3), EMPTY_STACK_CLOSE_SYMBOL, State(6))
        pda.get_Fsa().add_arc(State(3), END_SYMBOL, State(4))

        # Finalize
        # pda.compile()

        # For now, we don't allow PDAs with unbounded stacks, so we have
        # to intersect this constraint with a maximum length constraint. This does
        # not change the expressibility of the model
        # max_length_constraint = self.max_length_constraint.build(input_tokens, token_to_key)
        # pda = pda.intersect(max_length_constraint)
        return pda

    def get_name(self) -> str:
        return 'balanced-parens'
