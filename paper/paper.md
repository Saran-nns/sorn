---
title: `sorn`: ' A Python toolbox for Self Organizing Recurrent Neural Network'
tags:
    - Python
    - Spiking Neural Network
    - OpenAI Gym
    - Neuromorphic computing
    - Neuroscience
    - Self Organization
authors:
    - name: Saranraj Nambusubramaniyan
      affiliation: 1
      orcid: 0000-0002-4043-3420
affiliation:
    - name: Indian center for Robotics Innovation and Smart-intelligence(IRIS-i), India
      index: 1

bibliography: paper.bib

---

# Summary

Self Organizing Recurrent Neural(SORN) networks is a class of neuro-inspired artificial network build based on plasticity mechanisms in biological brain and mimic neocortical circuits ability of learning and adaptation through neuroplasticity mechanisms. Structurally, unlike other liquid
state models, SORN networks consists of pool of excitatory neurons and small population of inhibitory neurons. The network implements five fundamental plasticity
mechanisms found in neocortex, namely spike timing dependent plasticity, intrinsic plasticity, synaptic scaling, inhibitory spike timing dependent plasticity and structural plasticity [@zheng2013network] [@lazar2009sorn]. Using mathematical modelling, SORN network simplifies the underlying structural and functional connectivity mechanisms that are responsible for learning and memory encoded in neuro-synapses of neocortex region of mammalian brain.

`sorn` is a Python package designed for Self Organizing Recurrent Neural networks[@PyPiPackage]. While it is originally developed for SORN networks, it can also serve as an ideal research package for Liquid State Machines in general. The detailed documentation is provided at ([https://self-organizing-recurrent-neural-networks.readthedocs.io/en/latest/](self-organizing-recurrent-neural-networks.readthedocs.io)).Further to extend the applications of this network, a demonstrative example of neuro robotics experiment with OpenAI gym[@gym] is also provided in the documentation.

## Statement of the need:

Developing SORN network demands multidisciplinary knowledge in Python, Neurophysiology, Numerical Computation and Computational Neuroscience. To reduce the cognitive load of the researchers new to the field, it is necessary to have a package that encapsulates all features in a simple API. There is another open source code available[@papa2017criticality] for SORN network but it is intended for task/problem specific and not a general purpose software package. However, `sorn` allows researchers to develop the network of their interest with the combination of plasticity rules of their choice. Overall, the package provide a research enviroment for computational neuroscients to investigate the structural and functional properties of the brain networks by reverse engineering neuronal plasticity mechanisms.

## Library Overview:

`sorn` provide a reliable open source platform for the neuroscience researchers to develop, simulate and train the SORN network.  It is heavily depend on Numpy[@harris2020array] for numerical computations and analysis methods, seaborn and matplotlib[@barrett2005matplotlib] for visualization. The network is defined broadly in three classes; `SORN` object encapsulates all required functions that instantiate network variables like connection weights and thresholds. `Plasticity` inherits objects from `SORN` and has all plasticity functions implemented. `NetworkState` has functions that evaluates excitatory and inhibitory network at each timestep and finally the `MatrixCollection` objects acts as a memory cache. It collects the network states and keep track of the variables as the network evolves during simulation and training.

The network is then instantiated, simulated and trained using two classes `Simulator` and `Trainer` which inherit objects from `SORN`.

## SORN Network Model

Excitatory network state

$$𝑥_𝑖(𝑡+1)=𝛩\left (\sum_{j=1}^{N^E}𝑊_{𝑖𝑗}^{𝐸𝐸}(𝑡) 𝑥_𝑗(𝑡)−\sum_{j=1}^{N^I}𝑊_{𝑖𝑘}^{𝐸𝐼}(𝑡) 𝑦_𝑘(𝑡)+𝑢_𝑖(𝑡)−𝑇_𝑖𝐸(𝑡)+𝜉_𝐸(t)\right)$$

Inhibitory Network state
$$𝑦_𝑖(𝑡+1)=𝛩\left(\sum_{j=1}^{N_i}𝑊_{𝑖𝑗}^{𝐼𝐸}(𝑡) 𝑥_𝑗(𝑡)−𝑇_𝑖𝐼+ 𝜉𝐼(t)\right)$$
## Plasticity Rules
### Spike Timing Dependent Plasticity

Controls the efficacy of connection strength between Excitatory neurons
$$𝛥𝑊_{𝑖𝑗}^{𝐸𝐸}=𝜂𝑆𝑇𝐷𝑃(𝑥_𝑖(𝑡)𝑥_𝑗(𝑡−1)−𝑥_𝑖(𝑡−1)𝑥_𝑗(𝑡)$$
### Intrinsic Plasticity

Update the firing threshold of excitatory neurons
$𝑇_𝑖(𝑡+1)=𝑇_𝑖(𝑡)+𝜂_{𝐼𝑃}(𝑥_𝑖(𝑡)−𝐻_{𝐼𝑃})$

### Structural Plasticity

Add new synapses between excitatory neurons at a rate of 0.1
### Synaptic Scaling

Normalizes the incoming synaptic strenghts of a neuron
$$𝑊_{𝑖𝑗}^{𝐸𝐸}(𝑡)←𝑊_{𝑖𝑗}^{𝐸𝐸}(𝑡)/Σ𝑊_{𝑖𝑗}^{𝐸𝐸}(𝑡)$$
### Inhibitory Spike Timing Dependent Plasticity

Updates the efficac of synapses from Inhibitory to Excitatory network
$$𝛥𝑊_{𝑖𝑗}^{𝐸I}=𝜂i𝑆𝑇𝐷𝑃(y_j(𝑡-1)(1-x_i(t)(1+\frac{1}{\mu_{ip}})))$$
## Usage

For simulation, the `Simulator.simulate_sorn` has to be called as follows,
### Simulation
```python

from sorn import Simulator
import numpy as np

# Sample input
num_features = 10
time_steps = 200
inputs = np.random.rand(num_features,time_steps)

matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Simulator.simulate_sorn(inputs = inputs, phase='plasticity', matrices=None, noise = True, time_steps=time_steps, _ne = 200, _nu=num_features)

The other `kwargs` are,
{'_ne', '_nu', '_network_type_ee', '_network_type_ei', '_network_type_ie', '_lambda_ee','_lambda_ei', '_lambda_ie', '_eta_stdp','_eta_inhib', '_eta_ip', '_te_max', '_ti_max', '_ti_min', '_te_min', '_mu_ip','_sigma_ip'}
```
and to resume the simulation, load the matrices returned at the previous step as

```python

matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Simulator.simulate_sorn(inputs = inputs, phase='plasticity', matrices=matrices_dict, noise= True, time_steps=time_steps,_ne = 200, _nu=num_features)

```
The network can also be trained with and without plasticity mechanisms using the `Trainer` object as

### Training
```python
from sorn import Trainer
inputs = np.random.rand(num_features,1)

# Under all plasticity mechanisms
matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Trainer.train_sorn(inputs = inputs, phase='plasticity', matrices=matrices_dict,_nu=num_features, time_steps=1)

# Resume the training without any plasticity mechanisms

matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Trainer.train_sorn(inputs = inputs, phase='training', matrices=matrices_dict,_nu=num_features, time_steps=1)
```

To turn off any plasticity mechanisms during simulation or training phase, you can use freeze argument. For example to stop intrinsic plasticity during training phase,

```python

matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Simulator.simulate_sorn(inputs = inputs, phase='plasticity', matrices=None, noise = True, time_steps=time_steps, _ne = 200, _nu=num_features, freeze=['ip'])

```
The other options are,

`stdp` - Spike Timing Dependent Plasticity

`ss` - Synaptic Scaling

`sp` - Structural Plasticity

`istdp` - Inhibitory Spike Timing Dependent Plasticity

Note: If you pass all above options to freeze, then the network will behave as Liquid State Machine(LSM)

The `simulate_sorn` and `train_sorn` accepts the following keyword arguments

| kwargs             |                                          Description                                       |
|--------------------|:------------------------------------------------------------------------------------------:|
| inputs             |  External stimulus                                                                         |
| phase              |  `plasticity` or `training`                                                                |
| matrices           |  `matrices_dict` to resume simulation otherwise `None` to intialize new network            |
| time_steps         |  `simulaton` total time steps. For `training` should be 1                                  |
| noise              |  If `True`, Gaussian white noise will be added to excitatory field potentials              |
| freeze             |  To drop anu given plasticity mechanism(s) among [`'ip'`,`'stdp'`,`'istdp'`,`'ss'`, `'sp'`]|
| _ne                |  Number of Excitatory neurons in the network                                               |
| _nu                |  Number of input units among excitatory neurons                                            |
| _network_type_ee   |  `sparse` or  `dense` connection between excitatory neurons                                |
| _network_type_ei   |  `sparse` or  `dense` connection from inhibitory and excitatory neurons                    |
| _network_type_ie   |  `sparse` or  `dense` connection from excitatory and inhibitory neurons                    |
| _lambda_ee         |  Connection density between excitatory networks if network type is `sparse`                |
| _lambda_ei         |  Density of connections from inhibitory to excitatory networks if network type is `sparse` |
| _lambda_ie         |  Density of connections from inhibitory to excitatory networks if network type is `sparse` |
| _eta_stdp          |  Hebbian learning rate of excitatory synapses                                              |
| _eta_inhib         |  Hebbian learning rate synapses from inhibitory to excitatory                              |
| _eta_ip            |  Learning rate of excitatory neuron threshold                                              |
| _te_max            |  Maximum of excitatory neuron threshold range                                              |
| _ti_max            |  Maximum of inhibitory neuron threshold range                                              |
| _ti_min            |  Minimum of inhibitory neuron threshold range                                              |
| _te_min            |  Minimum of excitatory neuron threshold range                                              |
| _mu_ip             |  Target Mean firing rate of excitatory neuron                                              |
| _sigma_ip          |  Target Standard deviation of firing rate of excitatory neuron                             |


### Network Output Descriptions
`matrices_dict` - Dictionary of connection weights ('Wee','Wei','Wie') , Excitatory network activity ('X'), Inhibitory network activities('Y'), Threshold values ('Te','Ti')

`Exc_activity` - Collection of Excitatory network activity of entire simulation period

`Inh_activity` - Collection of Inhibitory network activity of entire simulation period

`Rec_activity` - Collection of Recurrent network activity of entire simulation period

`num_active_connections` - List of number of active connections in the Excitatory pool at each time step

## Statistical and Analysis methods

```python
from sorn import Statistics

# t-lagged auto correlation between neural activity
corr_coeff = Statistics.autocorr(firing_rates = [1,1,5,6,3,7],t= 2)

# Fano factor: To verify poissonian process in spike generation of neuron 10
mean_firing_rate, variance_firing_rate, fano_factor = Statistics.fanofactor(spike_train= np.asarray(Exc_activity),neuron = 10,window_size = 10)

# Measure the uncertainty about the origin of spike from the network using entropy
sse = Statistics.spike_source_entropy(spike_train= np.asarray(Exc_activity), num_neurons=200)

# Spike rate of specific neuron
time_period, bin_size, spike_rate = Statistics.firing_rate_neuron(spike_train: np.array, neuron: int, bin_size: int)

# Firing rate of the network
firing_rate = Statistics.firing_rate_network(spike_train: np.array)


# Average Pearson correlation coeffecient between neurons
corr_mat, corr_coeff = avg_corr_coeff(spike_train: np.array)

# Time instants at which neuron spikes
event_time = spike_times(spike_train: np.array)

# Inter spike intervals for each neuron
isi = spike_time_intervals(spike_train)

# Hamming distance between true netorks states and perturbed network states
hamming_distance(actual_spike_train: np.array, perturbed_spike_train: np.array)
```
## Plotter

# References