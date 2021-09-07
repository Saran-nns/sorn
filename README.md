
# Self-Organizing Recurrent Neural Networks

SORN is a class of neuro-inspired artificial network build based on plasticity mechanisms in biological brain and mimic neocortical circuits ability of learning and adaptation. SORN consists of pool of excitatory neurons and small population of inhibitory neurons which are controlled by 5 plasticity mechanisms found in neocortex, namely Spike Timing Dependent Plasticity (STDP), Intrinsic Plasticity (IP), Synaptic Scaling (SS),Synaptic Normalization(SN) and inhibitory Spike Timing Dependent Plasticity (iSTDP). Using mathematical tools, SORN network simplifies the underlying structural and functional connectivity mechanisms responsible for learning and memory in the brain

'sorn' is a Python package designed for Self Organizing Recurrent Neural Networks. It provides a research environment for computational neuroscientists to study the self-organization, adaption, learning,memory and behavior of brain circuits by reverse engineering neural plasticity mechanisms. Further to extend the potential applications of `sorn`, a demostrative example of a neuro-robotics experiment using OpenAI gym is also [documented](https://self-organizing-recurrent-neural-networks.readthedocs.io/en/latest/usage.html).


[![Build Status](https://github.com/saran-nns/sorn/workflows/Build/badge.svg)](https://github.com/saran-nns/sorn/actions)
[![codecov](https://codecov.io/gh/Saran-nns/sorn/branch/master/graph/badge.svg)](https://codecov.io/gh/Saran-nns/sorn)
[![Documentation Status](https://readthedocs.org/projects/self-organizing-recurrent-neural-networks/badge/?version=latest)](https://self-organizing-recurrent-neural-networks.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/sorn.svg)](https://badge.fury.io/py/sorn)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://pepy.tech/badge/sorn)](https://pepy.tech/project/sorn)
[![DOI](https://zenodo.org/badge/174756058.svg)](https://zenodo.org/badge/latestdoi/174756058)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/164AKTA-iCVLq-iR-treLA_Y9keRYrQkH#scrollTo=Rt2YZptMtC14)
[![status](https://joss.theoj.org/papers/7dc447f7a0d17d774b59c8ee15c223c2/status.svg)](https://joss.theoj.org/papers/7dc447f7a0d17d774b59c8ee15c223c2)

<h4 align="Left">SORN Reservoir and the evolution of synaptic efficacies</h4>
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/PySORN_0.1/master/v0.1.0/doc/images/SORN1.png" height="320" width="430"></a>

## Installation

```python
pip install sorn
```

The library is still in alpha stage, so you may also want to install the latest version from the development branch

```python
pip install git+https://github.com/Saran-nns/sorn
```

## Dependencies
SORN supports Python 3.5+ ONLY. For older Python versions please use the official Python client.
To install all optional dependencies,

```python
  pip install 'sorn[all]'
```
## Usage
### Plasticity Phase

```python
import sorn
from sorn import Simulator
import numpy as np

# Sample input
num_features = 10
time_steps = 200
inputs = np.random.rand(num_features,time_steps)

# Simulate the network with default hyperparameters under gaussian white noise
state_dict, E, I, R, C = Simulator.simulate_sorn(inputs = inputs, phase='plasticity',
                                                matrices=None, noise = True,
                                                time_steps=time_steps)

```
```
Network Initialized
Number of connections in Wee 3909 , Wei 1574, Wie 8000
Shapes Wee (200, 200) Wei (40, 200) Wie (200, 40)
```
### Training Phase
```python
from sorn import Trainer
# NOTE: During training phase, input to `sorn` should have second (time) dimension set to 1. ie., input shape should be (input_features,1).

inputs = np.random.rand(num_features,1)

# SORN network is frozen during training phase
state_dict, E, I, R, C = Trainer.train_sorn(inputs = inputs, phase='training',
                                            matrices=state_dict, noise= False,
                                            time_steps=1,
                                            ne = 100, nu=num_features,
                                            lambda_ee = 10, eta_stdp=0.001 )
```
### Network Output Descriptions
  `state_dict`  - Dictionary of connection weights (`Wee`,`Wei`,`Wie`) , Excitatory network activity (`X`), Inhibitory network activities(`Y`), Threshold values (`Te`,`Ti`)

  `E` - Excitatory network activity of entire simulation period

  `I` - Inhibitory network activity of entire simulation period

  `R` - Recurrent network activity of entire simulation period

  `C` - Number of active connections in the Excitatory pool at each time step

### Documentation
For detailed documentation about development, analysis, plotting methods and a sample experiment with OpenAI Gym, please visit [SORN Documentation](https://self-organizing-recurrent-neural-networks.readthedocs.io/en/latest/)

### Citation

```Python
@software{saranraj_nambusubramaniyan_2020_4184103,
  author       = {Saranraj Nambusubramaniyan},
  title        = {Saran-nns/sorn: Stable alpha release},
  month        = nov,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.3.1},
  doi          = {10.5281/zenodo.4184103},
  url          = {https://doi.org/10.5281/zenodo.4184103}
}
```

### Contributions
I am welcoming contributions. If you wish to contribute, please create a branch with a pull request and the changes can be discussed there.
If you find a bug in the code or errors in the documentation, please open a new issue in the Github repository and report the bug or the error. Please provide sufficient information for the bug to be reproduced.


