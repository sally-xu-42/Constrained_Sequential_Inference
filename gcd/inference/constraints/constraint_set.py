import torch
from allennlp.common.from_params import FromParams
from allennlp.data import Vocabulary
from typing import Dict, List, Optional, Tuple

from gcd.inference.constraints import Constraint
from gcd.inference.constraints.parsing.util import hash_dict
from rayuela.base.automaton import Automaton


class ConstraintSet(FromParams):
    def __init__(self,
                 constraints: List[Constraint],
                 vocab: Vocabulary,
                 namespace: str) -> None:
        self.constraints = constraints
        self.automata: List[Automaton] = []
        self.constraint_automaton: Automaton = None
        self.working_set = set()
        self.non_working_set = set(range(len(constraints)))

        self.vocab = vocab
        self.namespace = namespace

        # Build indices:
        self.token_to_key: Dict[str, int] = vocab.get_token_to_index_vocabulary(namespace)
        self.all_indices = list(self.token_to_key.values())

    def setup(self, input_tokens: torch.Tensor, *args, **kwargs) -> None:
        dhash = hash_dict(self.token_to_key)
        self.automata = []
        for constraint in self.constraints:
            automaton = constraint.build(input_tokens, self.token_to_key, dhash, *args, **kwargs)
            self.automata.append(automaton)
        self.constraint_automaton = None
        self.working_set = set()
        self.non_working_set = set(range(len(self.constraints)))

    def force_full_intersection(self) -> None:
        for idx, _ in enumerate(self.constraints):
            self.add_contraint_to_working_set(idx)

    def is_valid(self, tokens: List[int]) -> bool:
        return all(automaton.accepts(tokens) for automaton in self.automata)

    def get_start(self) -> int:
        if self.constraint_automaton is None:
            return None
        start = self.constraint_automaton.get_start()
        # print(f'Getting start {start}')
        return start

    def step(self, state: int, stack: int, action: int) -> Tuple[int, int]:
        if self.constraint_automaton is None:
            return None, None
        # print(f'Step with state {state}, stack {stack} and action {action}')
        return self.constraint_automaton.step(state, stack, action)

    def get_valid_actions(self, state: int, stack: int) -> List[int]:
        if self.constraint_automaton is None:
            return self.all_indices
        return self.constraint_automaton.get_valid_actions(state, stack)

    def get_violated_constraint(self, tokens: List[int]) -> Optional[int]:
        """Returns the index of the constraint in the non-working set which is violated."""
        if len(self.non_working_set) == 0:
            return None

        # Optimization hack: make the output_tokens into an automaton exactly once
        # so it doesn't need to be done every time in accepts
        for index in self.non_working_set:
            if not self.automata[index].accept(tokens):
                return index
        return None

    def get_all_violated_constraints(self, tokens: List[int]) -> List[int]:
        return [i for i, automaton in enumerate(self.automata) if not automaton.accept(tokens)]

    def add_contraint_to_working_set(self, index: int) -> None:
        print(f'Adding {self.constraints[index].get_name()} to the working set')
        self.working_set.add(index)
        self.non_working_set.remove(index)
        if self.constraint_automaton is None:
            self.constraint_automaton = self.automata[index]
        else:
            self.constraint_automaton = self.constraint_automaton.intersect(self.automata[index])

    def add_all_constraints_to_working_set(self) -> None:
        """Adds all of the constraints in the non-working set to the working set."""
        for index in reversed(list(self.non_working_set)):
            self.add_contraint_to_working_set(index)

    def get_working_set(self) -> List[str]:
        return [self.constraints[index].get_name() for index in self.working_set]

    def get_all_violated_constraints_names(self, output_tokens: List[int]) -> List[str]:
        return [self.constraints[index].get_name() for index in self.get_all_violated_constraints(output_tokens)]
