# guided_dropout
Implementation of the "guided dropout", an algorithm presented in "Fast Power system security analysis with Guided Dropout" at the ESANN 2018 conference and used in the paper "Anticipating contingengies in power grids using fast neural net screening" presented at the IEEE WCCI 2018 conference. (see bellow for the bibtex references).

## Outlines
The main idea of this architecture, is to adapt the structure of a neural network (switching on / off units) depending on some categorical variable.

It has been applied with succes for predicting flows on powergrid, when the reference grid suffers some contingencies (line disconnection).

It shows interesting properties: when trained on single line disconnection, it is able to accurately predict flows when multiple lines are disconnected.

We believe it has a wider range of applications.

## Code
It requires tensorflow, and python3.

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


