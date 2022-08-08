def parenthesis_match(tokens: list) -> bool:
    stack = 0
    for token in tokens:
        if token == '(':
            stack += 1
        elif token == ')':
            if stack == 0:
                return False
            stack -= 1
        else:  # terminal symbols
            pass
    return stack == 0


def test():
    assert(parenthesis_match(list('')))
    assert(parenthesis_match(list('abcdef')))
    assert(parenthesis_match(list('()')))
    assert(parenthesis_match(list('(acsd(())dsa)()')))
    assert(parenthesis_match(list('((ad(dsfsa(d))ds)as)saewhg41(dsa()())')))
    assert(not parenthesis_match(list('(()')))
    assert(not parenthesis_match(list(')')))
    assert(not parenthesis_match(list('dsa)dsaf)((')))
    assert(not parenthesis_match(list(')()(')))
    assert(not parenthesis_match(list('((ad(dsfsa(d))ds)as)saewhg41(dsa(()())')))
