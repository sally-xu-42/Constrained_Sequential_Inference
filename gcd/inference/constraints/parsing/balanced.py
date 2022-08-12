from collections import defaultdict
import torch
from typing import Dict
from gcd.inference.constraints.parsing import util
from rayuela.base.automaton import Automaton

from rayuela.fsa.pda import PDA
from gcd.inference.constraints import Constraint


@Constraint.register('balanced-parens')
class BalancedParenthesesConstraint(Constraint):
    cache: Dict[str, Dict[int, PDA]] = defaultdict(lambda: {})
    def __init__(self, max_length: int) -> None:
        # max_length = 80
        self.max_length = max_length
        from gcd.inference.constraints.parsing import MaxLengthConstraint
        self.max_length_constraint = MaxLengthConstraint(max_length)

    def build(self,
              input_tokens: torch.Tensor,
              token_to_key: Dict[str, int],
              dict_hash: str = None, *args, **kwargs) -> Automaton:
        if dict_hash is None: dict_hash = util.hash_dict(token_to_key)
        pda = self.cache[self.max_length].get(dict_hash)
        if pda is None:
            print(f'Compiling a BalancedParenthesesConstraint with size {len(token_to_key)} vocab...')
            # pda = PDA(token_to_key, max_length=self.max_length)
            pda = PDA(token_to_key)
            # For now, we don't allow PDAs with unbounded stacks, so we have
            # to intersect this constraint with a maximum length constraint. This does
            # not change the expressibility of the model
            max_length_constraint = self.max_length_constraint.build(input_tokens, token_to_key, dict_hash)
            pda = pda.intersect(max_length_constraint)
            self.cache[self.max_length][dict_hash] = pda
            
        return pda

    def get_name(self) -> str:
        return 'balanced-parens'
