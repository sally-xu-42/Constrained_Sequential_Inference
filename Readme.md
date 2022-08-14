# AFLT Project
This repository contains the code for our AFLT project. Our codes are adapted from https://github.com/CogComp/gcd.git and https://github.com/rycolab/aflt-f2022.git.

## Experiments
We experiment on constituency parsing.

More details on how to reproduce the results can be found under `data/ptb` and `experiments/parsing`.

## Installing Dependencies
The Python dependencies which can be directly installed via pip are in `requirements.txt`.

We recommend to use python 3.7

## Testing
We provide some unit tests under `gcd/tests`. You can run `pytest` to check if automata functionalities work as expected.

### Installation Issues
Here is one common error and what worked for us to fix it.

#### TypeError: Couldn't build proto file into descriptor pool!
```
pip uninstall protobuf
pip install --no-binary=protobuf protobuf
```
Solution via https://github.com/ValvePython/csgo/issues/8