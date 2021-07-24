---
title: '`sorn`: A Python package for Self Organizing Recurrent Neural Network'
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
affiliations:
    - name: Indian center for Robotics Innovation and Smart-intelligence(IRIS-i), India
      index: 1
date: 24 July 2021
bibliography: paper.bib
---
# Summary

Self Organizing Recurrent Neural(SORN) network is a class of neuro-inspired artificial network build based on plasticity mechanisms in biological brain and mimic neocortical circuits ability of learning and adaptation through neuroplasticity mechanisms. Structurally, SORN networks consists of pool of excitatory neurons and small population of inhibitory neurons. The network implements five fundamental plasticity
mechanisms found in neocortex of brain, namely spike timing dependent plasticity, intrinsic plasticity, synaptic scaling, inhibitory spike timing dependent plasticity and structural plasticity [@zheng2013network; @lazar2009sorn]. Using mathematical modelling, SORN network simplifies the underlying structural and functional connectivity mechanisms that are responsible for learning and memory encoded in neuro-synapses of neocortex region of mammalian brain.

`sorn` is a Python package designed for Self Organizing Recurrent Neural networks. While it is originally developed for SORN networks, it can also serve as an ideal research package for Liquid State Machines in general. The detailed documentation is provided at [https://self-organizing-recurrent-neural-networks.readthedocs.io/en/latest/](self-organizing-recurrent-neural-networks.readthedocs.io).Further to extend the applications of this network, a demonstrative example of neuro robotics experiment with OpenAI gym[@gym] is also provided in the documentation.

## Statement of the need:

Reservoir computing models are neuro inspired artifical neural networks. RC networks has either sparse or densly connected units with fixed connection weights. Unlike other RC models, SORN has synaptic weights controlled by neuro inspired plasticity mechanisms. The network has two distinct pools of excitatory and inhibitory reservoirs competing with each other to remain in subcritical state suitable for learning. Sub critical regime is a state between chaos and order, otherwise `edge of chaos`. At this state, network has intrinsic dynamics with strong affinity towards order, yet sensitive to external perturbations. Under carefully designed plasticity mechansisms, the network has the ability to overcome the perturbations and return to their subcritical dynamics. That self-adaptive behavior is otherwise called Self Organization. Building such network from scratch is time consuming and require deeper understanding of neurophysiology and softcomputing. Therefore, to reduce the cognitive load of the theorists, experimentalist or researchers new to the field, it is necessary to have a realiable package that encapsulates all plasticity mechanisms to study self organization, adaptation, learning, memory and behavior of biological brain.

There is another open source code [see @papa2017criticality] for SORN network but it is intended for problem specific and not a general purpose software package. However, `sorn` is flexible package that allows researchers to develop the network of their interest with respect to the combination of plasticity rules of their choice. Overall, the `sorn` provide a research enviroment for computational neuroscients to investigate the structural and functional properties of the brain networks by reverse engineering neuronal plasticity mechanisms.

## Library Overview:

`sorn` package heavily depend on Numpy [ @harris2020array] for numerical computations and analysis methods, seaborn and matplotlib [ @barrett2005matplotlib] for visualization. The network is defined broadly in three classes; `SORN` object encapsulates all required functions that instantiate network variables like connection weights and thresholds. `Plasticity` inherits objects from `SORN` and implements plasticity rules using `stdp()`, `ip()`, `ss()`, `sp()`and `istdp()` methods . `NetworkState` has mthods that evaluates excitatory and inhibitory network states at each timestep and finally the `MatrixCollection` objects acts as a memory cache. It collects the network states and keep track of the variables like weights and thresholds as the network evolves during simulation and training.

The network can be instantiated, simulated and trained using two classes `Simulator` and `Trainer` which inherit objects from `SORN`.

## SORN Network Model

Excitatory network state

$$𝑥_𝑖(𝑡+1)=𝛩\left (\sum_{j=1}^{N^E}𝑊_{𝑖𝑗}^{𝐸𝐸}(𝑡) 𝑥_𝑗(𝑡)−\sum_{j=1}^{N^I}𝑊_{𝑖𝑘}^{𝐸𝐼}(𝑡) 𝑦_𝑘(𝑡)+𝑢_𝑖(𝑡)−𝑇_𝑖𝐸(𝑡)+𝜉_𝐸(t)\right)$$

Inhibitory Network state

$$𝑦_𝑖(𝑡+1)=𝛩\left(\sum_{j=1}^{N_i}𝑊_{𝑖𝑗}^{𝐼𝐸}(𝑡) 𝑥_𝑗(𝑡)−𝑇_𝑖𝐼+ 𝜉𝐼(t)\right)$$

## Plasticity Rules

### Spike Timing Dependent Plasticity

It changes the  synaptic efficacy between excitatory neurons  based on the spike- timing between pre $j$ and post synaptic neuron $i$.

$$𝛥𝑊_{𝑖𝑗}^{𝐸𝐸}=𝜂_{𝑆𝑇𝐷𝑃}(𝑥_𝑖(𝑡)𝑥_𝑗(𝑡−1)−𝑥_𝑖(𝑡−1)𝑥_𝑗(𝑡)$$

### Intrinsic Plasticity

IP update the firing threshold of excitatory neurons based on the state of the neuron at each time step. It increases the threshold if the neuron fires and decrease it otherwise.

$$𝑇_𝑖\(𝑡+1)=𝑇_𝑖(𝑡)+𝜂_{𝐼𝑃}(𝑥_𝑖(𝑡)−𝐻_{𝐼𝑃}\)$$

### Structural Plasticity

It is responsible for creating new synapses between excitatory neurons at a rate of 1 per every 10th time step.

### Synaptic Scaling

SS normalizes the incoming synaptic strenghts of a neuron and prevent the network activity from attenuation or exploding.

$$𝑊_{𝑖𝑗}^{𝐸𝐸}(𝑡)←𝑊_{𝑖𝑗}^{𝐸𝐸}(𝑡)/Σ𝑊_{𝑖𝑗}^{𝐸𝐸}(𝑡)$$

### Inhibitory Spike Timing Dependent Plasticity

iSTDP is responisble for controlling the synaptic strenghts from Inhibitory to Excitatory network.

$$𝛥𝑊_{𝑖𝑗}^{𝐸I}=𝜂_{i𝑆𝑇𝐷𝑃}(y_j(𝑡-1)(1-x_i(t)(1+\frac{1}{\mu_{ip}})))$$

## Sample Simulation methods
```python

from sorn import Simulator
import numpy as np

# Sample input
num_features = 10
time_steps = 200
inputs = np.random.rand(num_features,time_steps)

matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Simulator.simulate_sorn(inputs = inputs, phase='plasticity', matrices=None, noise = True, time_steps=time_steps, _ne = 200, _nu=num_features)
```
and to resume the simulation, load the matrices returned at the previous step as

```python

matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Simulator.simulate_sorn(inputs = inputs, phase='plasticity', matrices=matrices_dict, noise= True, time_steps=time_steps,_ne = 200, _nu=num_features)

```
The network can also be trained with and without plasticity mechanisms using the `Trainer` object as

## Sample Training methods
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

The `simulate_sorn` and `train_sorn` methods accepts the following keyword arguments

| kwargs             |                                          Description                                       |
|--------------------|:------------------------------------------------------------------------------------------:|
| inputs             |  External stimulus                                                                         |
| phase              |  `plasticity` or `training`                                                                |
| matrices           |  `matrices_dict` to resume simulation otherwise `None` to intialize new network            |
| time_steps         |  `simulaton` total time steps. For `training` should be 1                                  |
| noise              |  If `True`, Gaussian white noise will be added to excitatory field potentials              |
| freeze             |  To drop any given plasticity mechanism(s) among [`'ip'`,`'stdp'`,`'istdp'`,`'ss'`, `'sp'`]|
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

### Analysis functions

`sorn` package also includes necessary methods to investigate network properties. Few methods in `Statistics` are,


| methods                |                                          Description                                       |
|------------------------|:------------------------------------------------------------------------------------------:|
| autocorr()             |  t-lagged auto correlation between neural activity                                         |
| fanofactor()           |  To verify poissonian process in spike generation of neuron(s)                             |
| spike_source_entropy() |  Measure the uncertainty about the origin of spike from the network using entropy          |
| firing_rate_neuron()   |  Spike rate of specific neuron                                                             |
| firing_rate_network()  |  Spike rate of entire network                                                              |
| avg_corr_coeff()       |  Average Pearson correlation coeffecient between neurons                                   |
| spike_times()          |  Time instants at which neuron spikes                                                      |
| spike_time_intervals() |  Inter spike intervals for each neuron                                                     |
| hamming_distance()     |  Hamming distance between two network states                                               |

More details about the statistical and plotting tools in the package is found at ([https://self-organizing-recurrent-neural-networks.readthedocs.io/en/latest/](self-organizing-recurrent-neural-networks.readthedocs.io))

# References