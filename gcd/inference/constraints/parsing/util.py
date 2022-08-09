from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from typing import Dict
import hashlib
import json

from gcd.inference.constraints.parsing.common import \
    EMPTY_STACK_OPEN_SYMBOL, EMPTY_STACK_CLOSE_SYMBOL, \
    OPEN_PAREN_SYMBOL, CLOSE_PAREN_SYMBOL


STACK_SYMBOLS = [OPEN_PAREN_SYMBOL, CLOSE_PAREN_SYMBOL,
                 EMPTY_STACK_OPEN_SYMBOL, EMPTY_STACK_CLOSE_SYMBOL]


def is_stack_token(token: str) -> bool:
    return token in STACK_SYMBOLS


def is_token_preterminal(token: str) -> bool:
    if token in [START_SYMBOL, END_SYMBOL,
                 DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN]:
        return False
    if token in STACK_SYMBOLS:
        return False
    if token.startswith('(') or token.startswith(')'):
        return False
    return True


def is_token_open_paren(token: str) -> bool:
    return token.startswith('(')


def is_token_close_paren(token: str) -> bool:
    return token.endswith(')')


def hash_dict(d: dict) -> str:
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(d, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()
