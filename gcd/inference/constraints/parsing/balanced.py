import torch
from typing import Dict

from rayuela.fsa.pda import PDA
from gcd.inference.constraints import Constraint


@Constraint.register('balanced-parens')
class BalancedParenthesesConstraint(Constraint):
    def __init__(self, max_length: int) -> None:
        from gcd.inference.constraints.parsing import MaxLengthConstraint
        self.max_length_constraint = MaxLengthConstraint(max_length)

    def build(self,
              input_tokens: torch.Tensor,
              token_to_key: Dict[str, int], *args, **kwargs) -> PDA:
        pda = PDA(token_to_key)
        # For now, we don't allow PDAs with unbounded stacks, so we have
        # to intersect this constraint with a maximum length constraint. This does
        # not change the expressibility of the model
        max_length_constraint = self.max_length_constraint.build(input_tokens, token_to_key)
        return pda.intersect(max_length_constraint)

    def get_name(self) -> str:
        return 'balanced-parens'
