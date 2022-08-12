from typing import Dict, Tuple
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from rayuela.base.automaton import Automaton
from rayuela.base.misc import is_token_open_paren, is_token_close_paren
from rayuela.fsa.state import State


# Push down automaton for parenthesis matching
# accepts: s ( xx ( xx () ) ) e, s () e, ...
# not accepts: s e, s () () e, s ( xx () e, ...
class PDA(Automaton):
    def __init__(self, token_to_key: Dict[str, int], dfsa=None, max_length=float('inf')) -> 'PDA':
        from rayuela.fsa.fsa import FSA
        from rayuela.fsa.dfsa import DFSA
        assert(isinstance(token_to_key, dict))
        self.token_to_key = token_to_key
        self.key_to_token = {k: t for t, k in token_to_key.items()}
        self.max_length = max_length
        if isinstance(dfsa, DFSA):
            self.dfsa = dfsa
        elif dfsa is None: # a dfsa that accepts all languages
            fsa = FSA()
            q = State(0)
            fsa.add_state(q)
            fsa.set_I(q)
            fsa.set_F(q)
            for a in token_to_key.values():
                fsa.add_arc(q, a, q)
            self.dfsa = fsa.compile()
        else:
            raise TypeError(f'Unknown automaton type {type(dfsa)}')

    def pda_accept(tokens: list) -> bool:
        state = 0
        stack = 0
        # 0 -s-> 1 -(-> 2 [self loop ()] -)-> 3 -e-> 4
        for token in tokens:
            if state == 0:
                if token == START_SYMBOL:
                    state = 1
                else:
                    return False
            elif state == 1:
                if is_token_open_paren(token):
                    state = 2
                    stack = 1
                else:
                    return False
            elif state == 2:
                if is_token_open_paren(token):
                    stack += 1
                elif is_token_close_paren(token):
                    if stack >= 2:
                        stack -= 1
                    else:  # stack == 1
                        stack = 0
                        state = 3
                elif token not in [START_SYMBOL, END_SYMBOL]:  # terminal symbols
                    pass
                else:
                    return False
            elif state == 3:
                if token == END_SYMBOL:
                    state = 4
                else:
                    return False
            else:  # state == 4
                return False
        return state == 4

    def accept(self, keys) -> bool:
        if len(keys) > self.max_length: return False
        return PDA.pda_accept([self.key_to_token[k] for k in keys]) and self.dfsa.accept(keys)
    
    def get_start(self) -> int:
        return PDA.encode_state(self.dfsa.get_start(), 0)

    def get_valid_actions(self, state: int, stack: int) -> list:
        fsa_state, pda_state = PDA.decode_state(state)
        fsa_actions = self.dfsa.get_valid_actions(fsa_state, stack)
        if pda_state == 0:
            return [a for a in fsa_actions if self.key_to_token[a] == START_SYMBOL]
        elif pda_state == 1:
            return [a for a in fsa_actions if is_token_open_paren(self.key_to_token[a])]
        elif pda_state == 2:
            return [a for a in fsa_actions if self.key_to_token[a] not in [START_SYMBOL, END_SYMBOL]]
        elif pda_state == 3:
            return [a for a in fsa_actions if self.key_to_token[a] == END_SYMBOL]
        else:  # pda_state == 4
            return []
    
    def step(self, state: int, stack: int, action) -> Tuple[int, int]:  # returns to state, to stack
        fsa_state, pda_state = PDA.decode_state(state)
        fsa_to_state, _ = self.dfsa.step(fsa_state, stack, action)
        action = self.key_to_token[action]
        # assert(pda_state <= 3)
        if pda_state == 0:
            assert(action == START_SYMBOL)
            pda_to_state = 1
            to_stack = 0
        elif pda_state == 1:
            assert(is_token_open_paren(action))
            pda_to_state = 2
            to_stack = 1
        elif pda_state == 2:
            # assert(action not in [START_SYMBOL, END_SYMBOL])
            if is_token_open_paren(action):
                pda_to_state = 2
                to_stack = stack + 1
            elif is_token_close_paren(action):
                assert(stack >= 1)
                if stack >= 2:
                    pda_to_state = 2
                    to_stack = stack - 1
                else:
                    pda_to_state = 3
                    to_stack = 0
            else:
                pda_to_state = 2
                to_stack = stack
        else:  # pda_state == 3
            if action != END_SYMBOL:
                print("PDA step stuck")
                # import dill
                # dill.dump(self, open('bad_pda.dill', 'wb'))
                # raise ValueError(f'Bad dfsa step with state {state}, stack {stack} and action {action}')
            pda_to_state = 4
            to_stack = 0
        return PDA.encode_state(fsa_to_state, pda_to_state), to_stack

    def encode_state(fsa_state: int, pda_state: int) -> int:
        return fsa_state * 5 + pda_state
    
    def decode_state(state: int) -> Tuple[int, int]:
        return state // 5, state % 5
