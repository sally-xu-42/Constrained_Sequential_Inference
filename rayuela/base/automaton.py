from typing import Tuple

class Automaton:
    def accept(self, tokens: list) -> bool:
        """ determines whether a string is in the language """
        raise NotImplemented

    def intersect(self, other: 'Automaton') -> 'Automaton':
        from rayuela.fsa.transformer import Transformer
        return Transformer.intersect(self, other)
    
    def get_start(self) -> int:
        raise NotImplemented

    def get_valid_actions(self, state: int, stack: int) -> list:
        raise NotImplemented
    
    def step(self, state: int, stack: int, action) -> Tuple[int, int]:  # returns to state, to stack
        raise NotImplemented

