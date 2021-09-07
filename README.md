
# Self-Organizing Recurrent Neural Networks

SORN is a class of neuro-inspired artificial network build based on plasticity mechanisms in biological brain and mimic neocortical circuits ability of learning and adaptation. SORN consists of pool of excitatory neurons and small population of inhibitory neurons which are controlled by 5 plasticity mechanisms found in neocortex, namely Spike Ti ming Dependent Plasticity (STDP), Intrinsic Plasticity (IP), Synaptic Scaling (SS),Synaptic Normalization(SN) and inhibitory Spike Timing Dependent Plasticity (iSTDP).

'sorn' is a Python package designed for Self Organizing Recurrent Neural Networks. It encapsulates all 5 plastcity mechansims and allows researchers to develop the network of their interest, provided they have the freedom to choose the combination of plasticity rules of their choice While it was originally developed for SORN networks, it can also serve as an ideal research package for Liquid State Machines.

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

## Contents
- [Self-Organizing Recurrent Neural Networks](#self-organizing-recurrent-neural-networks)
- [Getting Started](#)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
- [Usage](#usage)
  - [Plasticity Phase](#plasticity-phase)
  - [Training phase](#training-phase)
  - [Network Output Descriptions](#network-output-descriptions)

- [Citation](#citation)
  - [Package](#package)
  - [Thesis](#thesis)
- [Contributions](#contributions)

## Installation

```python
pip install sorn
```

The library is still in alpha stage, so you may also want to install the latest version from the development branch:

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

### Usage and sample experiment with OpenAIGym

For detailed documentation about development, analysis, plotting methods and a sample experiment with OpenAI Gym, please visit[SORN-Documentation](https://self-organizing-recurrent-neural-networks.readthedocs.io/en/latest/usage.html)

Sample call for few plotting and statistical methods in `sorn` package are shown below;

```python
from sorn import Plotter

# Plot Spike train of all neurons in the network
E = np.random.randint(2, size=(200,1000)) # For example, activity of 200 excitatory neurons in 1000 time steps
Plotter.scatter_plot(spike_train = E, savefig=True)
```
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/sorn/master/imgs/ScatterSpikeTrain.png" height="320" width="430"></a>

```python
# Inter spike intervals with exponential curve fit for neuron 1 in the Excitatory pool
Plotter.isi_exponential_fit(E,neuron=1,bin_size=5, savefig=True)
```
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/sorn/master/imgs/isi_exponential_fit.png" height="320" width="430"></a>

```python
# Distribution of connection weights in linear and lognormal scale
Plotter.linear_lognormal_fit(weights=Wee,num_points=100, savefig=True)
```
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/sorn/master/imgs/LinearLognormalFit.png" height="240" width="480"></a>

Sample simulation and training runs with few plotting functions are found at [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/164AKTA-iCVLq-iR-treLA_Y9keRYrQkH#scrollTo=Rt2YZptMtC14)

## Statistics and Analysis functions

```Python
from sorn import Statistics
#t-lagged auto correlation between neural activity
pearson_corr_matrix = Statistics.autocorr(firing_rates = [1,1,5,6,3,7], t= 2)

# Fano factor: To verify poissonian process in spike generation of neuron 10
mean_firing_rate, variance_firing_rate, fano_factor = Statistics.fanofactor(spike_train= E,
                                                                            neuron = 10,
                                                                            window_size = 10)

# Spike Source Entropy: To measure the uncertainty about the origin of spike from the network using entropy
sse = Statistics.spike_source_entropy(spike_train= E, num_neurons=200)
```
## Citation
### Package
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

## Contributions
I am welcoming contributions. If you wish to contribute, please create a branch with a pull request and the changes can be discussed there.
If you find a bug in the code or errors in the documentation, please open a new issue in the Github repository and report the bug or the error. Please provide sufficient information for the bug to be reproduced.


