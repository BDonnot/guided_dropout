# guided_dropout
Implementation of the "guided dropout", an algorithm presented in "Fast Power system security analysis with Guided Dropout" at the ESANN 2018 conference and used in the paper "Anticipating contingengies in power grids using fast neural net screening" presented at the IEEE WCCI 2018 conference. (see bellow for the bibtex references).

## Outlines
The main idea of this architecture, is to adapt the structure of a neural network (switching on / off units) depending on some categorical variable.

It has been applied with succes for predicting flows on powergrid, when the reference grid suffers some contingencies (line disconnection).

It shows interesting properties: when trained on single line disconnection, it is able to accurately predict flows when multiple lines are disconnected.

We believe it has a wider range of applications.

Let's imagine you have a problem with two kinds of variable:
- x : continuous variables (in the papers it was the  productions and loads of the power grid)
- `$\tau$` : a vector of {0,1} (not necessarily one hot), in these paper it was which line was connected (`$\tau[i]=0$` -> line i is connected)
- y : continous variables (in the papers it was the flows on each power line of the grid)

Suppose you have a function S, that given `$x$` and $\tau$ is able to compute `$y$`: `$y = S(x;\tau)$`.

In that case, you can apply guided dropout in a neural network to predict `$\hat{y}$` from `$x$` and `$\tau$`.

This package, as well as the readme is still in development.

## Requirements and installation
It requires python3, tensorflow and numpy.

To install it, clone this repository, cd in the proper repository then:
```python
pip install .
```

## Code example
Let's say you want to apply guided dropout on the hidden layer h, masking some of its neuron depending on the condition of the tensor tau.

Here what you have to add in the definition of your tensflow graph
```python
# create the operator for the guided dropout, we suppose that tau has a size of "sizeinputonehot"
# and h has a dimension of "latentdim"
gd_op = SpecificGDOEncoding(sizeinputonehot=sizeinputonehot,
                                sizeout=latentdim,
                                )
# set the mask in this operator
gd_op.set_mask(tau)
# effectively mask the unit in "h" depending on the configuration of "tau"
h_after_gd = gd_op(h)  

# h_after_gd will have the same shape than h, and can be use exactly the same manner.
```

A more detailed version of this usage, as well as a example is shown in the code.

## Use example and results
This part has not been documented yet, but a working example can be found by running:
```python
python GuidedDropout/GuidedDropout.py
```
From inside this repository.


## References
@inproceedings{donnot:hal-01695793,
  TITLE = {{Fast Power system security analysis with Guided Dropout}},
  AUTHOR = {Donnot, Benjamin and Guyon, Isabelle and Schoenauer, Marc and Marot, Antoine and Panciatici, Patrick},
  URL = {https://hal.archives-ouvertes.fr/hal-01695793},
  BOOKTITLE = {{European Symposium on Artificial Neural Networks}},
  ADDRESS = {Bruges, Belgium},
  YEAR = {2018},
  MONTH = Apr,
  PDF = {https://hal.archives-ouvertes.fr/hal-01695793/file/main.pdf},
  HAL_ID = {hal-01695793},
  HAL_VERSION = {v1},
}

@inproceedings{donnot:hal-01783669,
  TITLE = {{Anticipating contingengies in power grids using fast neural net screening}},
  AUTHOR = {Donnot, Benjamin and Guyon, Isabelle and Schoenauer, Marc and Marot, Antoine and Panciatici, Patrick},
  URL = {https://hal.archives-ouvertes.fr/hal-01783669},
  BOOKTITLE = {{ IEEE WCCI 2018}},
  ADDRESS = {Rio de Janeiro, Brazil},
  YEAR = {2018},
  MONTH = Jul,
  PDF = {https://hal.archives-ouvertes.fr/hal-01783669/file/main.pdf},
  HAL_ID = {hal-01783669},
  HAL_VERSION = {v1},
}


