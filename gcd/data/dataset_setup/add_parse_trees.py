from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import argparse
import benepar
import tqdm


def _load_srl_data(filename, field=None):
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            if field is not None:
                dataset.append(data[field])
            else:
                dataset.append(data)
    return dataset


def _save_data(data, json_file):
    with open(json_file, 'w') as f:
        for inst in data:
            f.write(json.dumps(inst) + "\n")


def prune_multiple_spaces(mystring: str):
    """ Prune multiple spaces in a sentence and replace with single space
    Parameters:
    -----------
    sentence: Sentence string with mulitple spaces

    Returns:
    --------
    cleaned_sentence: String with only single spaces.
    """

    mystring = mystring.strip()
    tokens = mystring.split(' ')
    tokens = [t for t in tokens if t != '']
    if len(tokens) == 1:
        return tokens[0]
    else:
        return ' '.join(tokens)


def add_parse_to_instances(json_file):
    parser = benepar.Parser("benepar_en2")
    data = _load_srl_data(json_file)
    for idx, inst in tqdm.tqdm(enumerate(data)):
        sentence = inst["words"]
        # print(idx)
        tree = parser.parse(sentence)
        treestring = str(tree)
        treestring = treestring.replace("\n", " ")
        treestring = prune_multiple_spaces(treestring)
        # print(treestring)
        inst["parse"] = treestring
    return data


def main(args):
    data = add_parse_to_instances(args["in"])
    _save_data(data, args["out"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--in', required=True, help="input json file")
    parser.add_argument('--out', required=True, help="output json file")
    args = parser.parse_args()
    args = vars(args)
    main(args)
