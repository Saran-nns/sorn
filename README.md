
# Self-Organizing Recurrent Neural Networks

SORN is a class of neuro-inspired artificial network build based on plasticity mechanisms in biological brain and mimic neocortical circuits ability of learning and adaptation through neuroplasticity mechanisms.

The network is developed as part of my Master thesis at Universität Osnabrück, Germany. For the ease of maintainance, the notebooks and the use cases are moved to [SORN-Notebook](https://github.com/Saran-nns/PySORN_0.1)

[![Build Status](https://travis-ci.org/Saran-nns/sorn.svg?branch=master)](https://travis-ci.org/Saran-nns/sorn)
[![codecov](https://codecov.io/gh/Saran-nns/sorn/branch/master/graph/badge.svg)](https://codecov.io/gh/Saran-nns/sorn)
[![Documentation Status](https://readthedocs.org/projects/self-organizing-recurrent-neural-networks/badge/?version=latest)](https://self-organizing-recurrent-neural-networks.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/sorn.svg)](https://badge.fury.io/py/sorn)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sorn.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2590449.svg)](https://doi.org/10.5281/zenodo.2590449)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Join the chat at https://gitter.im/Self-Organizing-Recurrent-Neural-Networks](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/Self-Organizing-Recurrent-Neural-Networks?utm_source=badge&utm_medium=badge&utm_content=badge)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10TElAAE1dsgzuvaHO_NjgMAE5Pic3_jL#scrollTo=VDa0U4mf1Z75)

<h4 align="Left">SORN Reservoir and the evolution of synaptic efficacies</h4>
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/PySORN_0.1/master/v0.1.0/doc/images/SORN1.png" height="320" width="430"></a> <a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/PySORN_0.1/master/v0.1.0/doc/images/weights.png" height="375" width="425" ></a> <a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/PySORN_0.1/master/v0.1.0/doc/images/networkxx.jpg" height="375" width="425" ></a>

## Contents
- [Self-Organizing Recurrent Neural Networks](#self-organizing-recurrent-neural-networks)
- [Getting Started](#)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
- [Simulate and Train](#usage)
  - [Update Network configurations](#update-network-configurations)
  - [Plasticity Phase](#plasticity-phase)
  - [Training phase](#training-phase)
  - [Network Output Descriptions](#network-output-descriptions)
  - [Colaboratory Notebook](#colaboratory-notebook)
- [Integrate with OpenAI gym](#usage-with-openai-gym)
  - [Cartpole balance problem](#cartpole-balance-problem)
- [Plotting functions](#plotting-functions)
- [Statistics and Analysis functions](#statistics-and-analysis-functions)
- [Citation](#citation)
  - [Package](#package)
  - [Thesis](#thesis)
- [Contributions](#contributions)
- [References](#references)

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
For detailed documentation about usage and development, please visit [SORN-Documentation](https://self-organizing-recurrent-neural-networks.readthedocs.io/)

## Usage

### Update Network configurations
There are two ways to update/configure the network parameters,
1. Navigate to home/conda/envs/ENVNAME/Lib/site-packages/sorn
 ```or``` if you are unsure about the directory of ```sorn```

Run

```python
import sorn

sorn.__file__
```
to find the location of the sorn package

Then, update/edit arguments in ```configuration.ini```

2. Pass the arguments with valid names (listed below). This will override the default values at ```configuration.ini```
. The allowed ```kwargs``` are,
```Python
kwargs = {'_ne', '_nu', '_network_type_ee', '_network_type_ei', '_network_type_ie', '_lambda_ee','_lambda_ei', '_lambda_ie', '_eta_stdp','_eta_inhib', '_eta_ip', '_te_max', '_ti_max', '_ti_min', '_te_min', '_mu_ip','_sigma_ip'}
```
### Plasticity Phase
The default ```_ne, _nu``` values are overriden by passing them as kwargs inside ```simulate_sorn``` method.

```Python
from sorn import Simulator
import numpy as np

# Sample input
num_features = 10
time_steps = 200
inputs = np.random.rand(num_features,time_steps)

# To simulate the network;
matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Simulator.simulate_sorn(inputs = inputs, phase='plasticity', matrices=None, noise = True, time_steps=time_steps, _ne = 200, _nu=num_features)

# To resume the simulation, load the matrices_dict from previous simulation;
matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Simulator.simulate_sorn(inputs = inputs, phase='plasticity', matrices=matrices_dict, noise= True, time_steps=time_steps,_ne = 200, _nu=num_features)
```
### Training phase

```Python
from sorn import Trainer
inputs = np.random.rand(num_features,1)

# SORN network is frozen during training phase
matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Trainer.train_sorn(inputs = inputs, phase='Training', matrices=matrices_dict,_nu=num_features, time_steps=1)
```
To turn off any plasticity mechanisms during simulation or training phase, you can use `freeze` argument.
For example to stop intrinsic plasticity during training phase,

```python
matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Simulator.simulate_sorn(inputs = inputs, phase='plasticity', matrices=None, noise = True, time_steps=time_steps, _ne = 200, _nu=num_features, freeze=['ip'])
```

The other options are,

`'stdp'` - Spike Timing Dependent Plasticity

`'ss'` - Synaptic Scaling

`'sp'` - Structural Plasticity

`'istdp'` - Inhibitory Spike Timing Dependent Plasticity

Note: If you pass all above options to `freeze`, then the network will behave as Liquid State Machine(LSM)

### Network Output Descriptions
  ```matrices_dict```  - Dictionary of connection weights ('Wee','Wei','Wie') , Excitatory network activity ('X'), Inhibitory network activities('Y'), Threshold values ('Te','Ti')

  ```Exc_activity``` - Collection of Excitatory network activity of entire simulation period

  ```Inh_activity``` - Collection of Inhibitory network activity of entire simulation period

  ```Rec_activity``` - Collection of Recurrent network activity of entire simulation period

  ```num_active_connections``` - List of number of active connections in the Excitatory pool at each time step

### Colaboratory Notebook
Sample simulation and training runs with few plotting functions are found at [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10TElAAE1dsgzuvaHO_NjgMAE5Pic3_jL#scrollTo=VDa0U4mf1Z75)

## Usage with OpenAI gym
### Cartpole balance problem
With default network parameters.

```python
from sorn import Simulator, Trainer
import gym

# Load the simulated network matrices
# Note that these matrices are obtained after the network achieved convergence under random inputs and noise

with open('simulation_matrices.pkl','rb') as f:
    sim_matrices,excit_states,inhib_states,recur_states,num_reservoir_conn = pickle.load(f)

# Training parameters

NUM_EPISODES = 2e6
NUM_PLASTICITY_EPISODES = 20000

env = gym.make('CartPole-v0')

# Policy
def policy(state,w):
    "Implementation of softmax policy"
    z = state.dot(w)
    exp = np.exp(z)
    return exp/np.sum(exp)

for EPISODE in range(NUM_EPISODES):

    # Environment observation; Input to sorn should be of shape (input_features,time_steps)
    state = env.reset()[:, None] # (4,) --> (4,1)
    state = np.array(state)

    # Play the episode
    while True:
      state = np.array(state[:,None])
      if EPISODE < NUM_PLASTICITY_EPISODE:

        # Plasticity phase
        sim_matrices, excit_states, inhib_states, recur_states, num_reservoir_conn = Simulator.simulate_sorn(inputs = state, phase ='plasticity', matrices = sim_matrices, time_steps = 1, noise=False)

      else:
        # Training phase with frozen reservoir connectivity
        sim_matrices,excit_states,inhib_states,recur_states,num_reservoir_conn = Trainer.train_sorn(inputs = state, phase = 'training', matrices = sim_matrices, noise= False)

      # Feed excit_states as input states to your RL algorithm, below goes for simple policy gradient algorithm
      # Sample policy w.r.t excitatory states and take action in the environment
      probs = policy(np.asarray(excit_states),output_layer_weights))
      action = np.random.choice(action_space,probs)

      state,reward,done,_ = env.step(action)

      if done:
        break

  # YOUR CODE HERE
  # COMPUTE GRADIENTS BASED ON YOUR OBJECTIVE FUNCTION
  # OPTIMIZE `output_layer_weights` BASED ON YOUR OPTIMIZATION METHOD
```
There are several neural data analysis and visualization methods inbuilt with `sorn` package. Sample call for few plotting and statistical methods are shown below;

## Plotting functions

```Python
from sorn import Plotter
# Plot weight distribution in the network
Plotter.weight_distribution(weights= matrices_dict['Wee'], bin_size = 5, savefig = False)

# Plot Spike train of all neurons in the network
Plotter.scatter_plot(spike_train = np.asarray(Exc_activity), savefig=False)

Plotter.raster_plot(spike_train = np.asarray(Exc_activity), savefig=False)
```

## Statistics and Analysis functions

```Python
from sorn import Statistics
#t-lagged auto correlation between neural activity
Statistics.autocorr(firing_rates = [1,1,5,6,3,7],t= 2)

# Fano factor: To verify poissonian process in spike generation of neuron 10
Statistics.fanofactor(spike_train= np.asarray(Exc_activity),neuron = 10,window_size = 10)

# Measure the uncertainty about the origin of spike from the network using entropy
Statistics.spike_source_entropy(spike_train= np.asarray(Exc_activity), num_neurons=200)
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

### Thesis
Saranraj Nambusubramaniyan(2019): Prospects of Biologically Plausible Artificial Brain Circuits Solving General Intelligence Tasks at the Imminence of Chaos
DOI: 10.13140/RG.2.2.25393.81762

## Contributions
I am welcoming contributions. If you wish to contribute, please create a branch with a pull request and the changes can be discussed there.
If you find a bug in the code or errors in the documentation, please open a new issue in the Github repository and report the bug or the error. Please provide sufficient information for the bug to be reproduced.

## References

Lazar, A. (2009). SORN: a Self-organizing Recurrent Neural Network. Frontiers in Computational Neuroscience, 3. https://doi.org/10.3389/neuro.10.023.2009

Hartmann, C., Lazar, A., Nessler, B., & Triesch, J. (2015). Where’s the Noise? Key Features of Spontaneous Activity and Neural Variability Arise through Learning in a Deterministic Network. PLoS Computational Biology, 11(12). https://doi.org/10.1371/journal.pcbi.1004640

Del Papa, B., Priesemann, V., & Triesch, J. (2017). Criticality meets learning: Criticality signatures in a self-organizing recurrent neural network. PLoS ONE, 12(5). https://doi.org/10.1371/journal.pone.0178683

Zheng, P., Dimitrakakis, C., & Triesch, J. (2013). Network Self-Organization Explains the Statistics and Dynamics of Synaptic Connection Strengths in Cortex. PLoS Computational Biology, 9(1). https://doi.org/10.1371/journal.pcbi.1002848

