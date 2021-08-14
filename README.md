
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
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10TElAAE1dsgzuvaHO_NjgMAE5Pic3_jL#scrollTo=VDa0U4mf1Z75)
[![status](https://joss.theoj.org/papers/7dc447f7a0d17d774b59c8ee15c223c2/status.svg)](https://joss.theoj.org/papers/7dc447f7a0d17d774b59c8ee15c223c2)

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
### Plasticity Phase
The default ```ne, nu``` values are overriden by passing them as kwargs inside ```simulate_sorn``` method.

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
The network defaults are,

Keyword argument | Description | Default Value |
--- | --- | --- |
ne              | 200      | Number of Encitatory neurons in the reservoir                              |
nu              | 10       | Number of Input neurons in the reservoir                                   |
network_type_ee | "Sparse" | `Sparse` or `Dense` connectivity between Excitatory neurons                |
network_type_ie | "Dense"  | `Sparse` or `Dense` connectivity from Excitatory to Inhibitory neurons     |
network_type_ei | "Sparse" | `Sparse` or `Dense` connectivity from Inhibitory to Excitatory neurons     |
lambda_ee       | 20       | % of connections between neurons in Excitatory pool                        |
lambda_ei       | 40       | % of connections from Inhibitory to Excitatory neurons                     |
lambda_ie       | 100      | % of connections from Excitatory to Inhibitory neurons                     |
eta_stdp        | 0.004    | Hebbian Learning rate for connections between excitatory neurons           |
eta_inhib       | 0.001    | Hebbian Learning rate for connections from Inhibitory to Excitatory neurons|
eta_ip          | 0.01     | Intrinsic plasticity learning rate                                         |
te_max          | 1.0      | Maximum excitatory neuron threshold value                                  |
ti_max          | 0.5      | Maximum inhibitory neuron threshold value                                  |
ti_min          | 0.0      | Minimum inhibitory neuron threshold value                                  |
te_min          | 0.0      | Minimum excitatory neuron threshold value                                  |
mu_ip           | 0.1      | Target mean firing rate of excitatory neuron                               |
sigma_ip        | 0.0      | Standard deviation of firing rate of excitatory neuron                     |

Override the default hyperparameters using the `kwargs` as shown below,
```Python

# To resume the simulation, load the state_dict from previous simulation;
state_dict, E, I, R, C = Simulator.simulate_sorn(inputs = inputs, phase='plasticity',
                                                matrices=None, noise= True,
                                                time_steps=time_steps,
                                                ne = 200, nu=num_features)
```
### Training phase

```Python
from sorn import Trainer
inputs = np.random.rand(num_features,1)

# SORN network is frozen during training phase
state_dict, E, I, R, C = Trainer.train_sorn(inputs = inputs, phase='training',
                                            matrices=state_dict,
                                            nu=num_features, time_steps=1)
```
### Freeze plasticity
To turn off any plasticity mechanisms during `simulation` or `training` phase, use `freeze` argument.
For example to stop intrinsic plasticity during simulation phase,

```python
# Sample input
num_features = 10
time_steps = 200
inputs = np.random.rand(num_features,time_steps)

state_dict, E, I, R, C = Simulator.simulate_sorn(inputs = inputs, phase='plasticity',
                                                matrices=None, noise = True,
                                                time_steps=time_steps, ne = 200,
                                                nu=num_features, freeze=['ip'])
```

The other options are,

`stdp` - Spike Timing Dependent Plasticity

`ss` - Synaptic Scaling

`sp` - Structural Plasticity

`istdp` - Inhibitory Spike Timing Dependent Plasticity

Note: If you pass all above options to `freeze`, then the network will behave as Liquid State Machine(LSM)

### Network Output Descriptions
  `state_dict`  - Dictionary of connection weights ('Wee','Wei','Wie') , Excitatory network activity ('X'), Inhibitory network activities('Y'), Threshold values ('Te','Ti')

  `E` - Collection of Excitatory network activity of entire simulation period

  `I` - Collection of Inhibitory network activity of entire simulation period

  `R` - Collection of Recurrent network activity of entire simulation period

  `C` - List of number of active connections in the Excitatory pool at each time step

### Colaboratory Notebook
Sample simulation and training runs with few plotting functions are found at [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10TElAAE1dsgzuvaHO_NjgMAE5Pic3_jL#scrollTo=VDa0U4mf1Z75)

## Usage with OpenAI gym
### Cartpole balance problem

```python
from sorn import Simulator, Trainer
import gym

# Hyperparameters
NUM_EPISODES = 2e6
NUM_PLASTICITY_EPISODES = 20000

LEARNING_RATE = 0.0001 # Gradient ascent learning rate
GAMMA = 0.99 # Discounting factor for the Rewards

env = gym.make('CartPole-v0')

# Policy
def policy(state,w):
    "Implementation of softmax policy"
    z = state.dot(w)
    exp = np.exp(z)
    return exp/np.sum(exp)

for EPISODE in range(NUM_EPISODES):

    # Environment observation;
    # NOTE: Input to sorn should have time dimension. ie., input shape should be (input_features,time_steps)
    state = env.reset()[:, None] # (4,) --> (4,1)
    state = np.array(state)

    grads = [] # Episode log policy gradients
    rewards = [] # Episode rewards

    # Keep track of game score to print
    score = 0

    # Play the episode
    while True:

      # env.render() # Uncomment to see your model train in real time (slow down training progress)
      state = np.array(state[:,None])
      if EPISODE < NUM_PLASTICITY_EPISODE:

        # Plasticity phase
        state_dict, E, I, R, C = Simulator.simulate_sorn(inputs = state, phase ='plasticity',
                                                        matrices = None, time_steps = 1,
                                                        nu=4, noise=False)

      else:
        # Training phase with frozen reservoir connectivity
        state_dict, E, I, R, C = Trainer.train_sorn(inputs = state, phase = 'training',
                                                matrices = state_dict, time_steps = 1,
                                                nu=4, noise= False)

      # Feed E as input states to your RL algorithm, below goes for simple policy gradient algorithm
      # Sample policy w.r.t excitatory states and take action in the environment
      probs = policy(np.asarray(E),output_layer_weights))
      action = np.random.choice(action_space,probs)
      state,reward,done,_ = env.step(action)

      # COMPUTE GRADIENTS BASED ON YOUR OBJECTIVE FUNCTION;
      # Sample computation of simple policy gradient objective function
      dsoftmax = softmax_grad(probs)[action,:]
      dlog = dsoftmax / probs[0,action]
      grad = np.asarray(reservoir_states).T.dot(dlog[None,:])
      grads.append(grad)
      rewards.append(reward)
      score+=reward
      # Update old state to new state
      state = next_state
      if done:
          break

    # OPTIMIZE OUTPUT LAYER WEIGHTS `w` BASED ON YOUR OPTIMIZATION METHOD;
    # Below is a sample of weight update based on gradient ascent(maximize cumulative reward) method for temporal difference learning
    for i in range(len(grads)):

        # Loop through everything that happened in the episode and update towards the log policy gradient times future reward
        w += LEARNING_RATE * grads[i] * sum([ r * (GAMMA ** r) for t,r in enumerate(rewards[i:])])
    print('Episode %s  Score %s' %(str(e),str(score)))
```

There are several neural data analysis and visualization methods inbuilt with `sorn` package. Sample call for few plotting and statistical methods are shown below;

## Plotting functions

```python
from sorn import Plotter
# Plot weight distribution in the network
Wee = np.random.randn(200,200)
Wee=Wee/Wee.max() # This is for example. Wee returned by the SORN is already normalized
Plotter.weight_distribution(weights= Wee, bin_size = 5, savefig = True)
```
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/sorn/revision/imgs/weight_distribution.png" height="320" width="430"></a>

```python
# Plot Spike train of all neurons in the network
E = np.random.randint(2, size=(200,1000))
Plotter.scatter_plot(spike_train = E, savefig=True)
```
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/sorn/revision/imgs/ScatterSpikeTrain.png" height="320" width="430"></a>

```python
# Raster plot of activity of only first 10 neurons in the excitatory pool
Plotter.raster_plot(spike_train = E[:,0:10], savefig=True)
```
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/sorn/revision/imgs/RasterSpikeTrain.png" height="320" width="430"></a>

```python
# Histogram of number of presynaptic connections per neuron in the excitatory pool
Plotter.hist_incoming_conn(weights=state_dict['Wee], bin_size=10, histtype='bar', savefig=True)
```
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/sorn/revision/imgs/hist_incoming_conn.png" height="320" width="430"></a>

```python
# Distribution of firing rate of the network
Plotter.hist_firing_rate_network(E, bin_size=10, savefig=True)
```
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/sorn/revision/imgs/hist_firing_rate_network.png" height="320" width="430"></a>

```python
# Plot pearson correlation between neurons
from sorn import Statistics
avg_corr_coeff,_ = Statistics.avg_corr_coeff(E)
Plotter.correlation(avg_corr_coeff,savefig=True)
```
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/sorn/revision/imgs/correlation_between_neurons.png" height="320" width="430"></a>

```python
# Inter spike intervals with exponential curve fit for neuron 1 in the Excitatory pool
Plotter.isi_exponential_fit(E,neuron=1,bin_size=10, savefig=True)
```
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/sorn/revision/imgs/isi_exponential_fit.png" height="320" width="430"></a>

```python
# Distribution of connection weights in linear and lognormal scale
Plotter.linear_lognormal_fit(weights=state_dict['Wee'],num_points=100, savefig=True)
```
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/sorn/revision/imgs/LinearLognormalFit.png" height="320" width="430"></a>

```python
# Draw network connectivity using the pearson correlation function between neurons in the excitatory pool
Plotter.plot_network(avg_corr_coeff,corr_thres=0.01,fig_name='network.png')
```
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/sorn/revision/imgs/network.png" height="320" width="430"></a>

## Statistics and Analysis functions

```Python
from sorn import Statistics
#t-lagged auto correlation between neural activity
Statistics.autocorr(firing_rates = [1,1,5,6,3,7],t= 2)

# Fano factor: To verify poissonian process in spike generation of neuron 10
Statistics.fanofactor(spike_train= E,neuron = 10,window_size = 10)

# Measure the uncertainty about the origin of spike from the network using entropy
Statistics.spike_source_entropy(spike_train= E, num_neurons=200)
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


