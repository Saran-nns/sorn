---
title: '`sorn`: A Python package for Self Organizing Recurrent Neural Network'
tags:
    - Python
    - Spiking Neural Network
    - OpenAI Gym
    - Neuromorphic computing
    - Neuroscience
    - Self Organizing Networks
    - Hebbian Learning
    - Associative Networks


authors:
    - name: Saranraj Nambusubramaniyan
      affiliation: "1,2"
      orcid: 0000-0002-4043-3420
affiliations:
    - name: Indian center for Robotics Innovation and Smart-intelligence(IRIS-i), India
      index: 1
    - - name: Institute of Congitive Science, Universität Osnabrück, Germany
      index: 2
date: 24 July 2021
bibliography: paper.bib
---
# Summary


Self Organizing Recurrent Neural(SORN) network is a class of neuro-inspired artificial network. It is proven that these class of networks can mimic neocortical circuits ability of learning and adaptation through neuroplasticity mechanisms. Structurally, SORN networks consists of pool of excitatory neurons and small population of inhibitory neurons. The network uses five fundamental plasticity
mechanisms found in neocortex of brain, namely spike timing dependent plasticity, intrinsic plasticity, synaptic scaling, inhibitory spike timing dependent plasticity and structural plasticity [@zheng2013network; @lazar2009sorn; @papa2017criticality] for optimizing its parameters. With the help of mathematical tools, SORN network simplifies the underlying structural and functional connectivity mechanisms that are responsible for learning and memory in brain.

`sorn` is a Python package designed for Self Organizing Recurrent Neural networks. While it is originally developed for SORN networks, it can also serve as an ideal research package for Liquid State Machines in general. The detailed documentation is provided at [https://self-organizing-recurrent-neural-networks.readthedocs.io/en/latest/](self-organizing-recurrent-neural-networks.readthedocs.io). Further to extend the applications of this network, a demonstrative example of neuro robotics experiment with OpenAI gym [ @brockman2016openai] is also provided at [sorn package](https://github.com/Saran-nns/sorn/).

## Statement of the need:

Reservoir computing models are neuro inspired artifical neural networks. RC networks have either sparse or densly connected units with fixed connection weights. Unlike other RC models, SORN has synaptic weights controlled by neuro inspired plasticity mechanisms. The network has two distinct pools of excitatory and inhibitory reservoirs competing with each other to remain in subcritical state suitable for learning. Sub critical regime is a state between chaos and order, otherwise "edge of chaos". At this state, network has intrinsic dynamics with strong affinity towards order, yet sensitive to external perturbations. Under plasticity mechansisms, the network has the ability to overcome the perturbations and return to their subcritical dynamics. That self-adaptive behavior is otherwise called Self Organization. Building such network with synergestic combination of plasticity mechanisms from scratch  require deeper understanding of neurophysiology and softcomputing. Further it is a time consuming process. Therefore, to reduce the cognitive load of the theorists, experimentalists or researchers who are new to the field, there is a need for a realiable package that encapsulates all plasticity mechanisms with high degree of reliability and flexibility.

There are few other open source code [sorn v1](https://github.com/delpapa/SORN), [sorn v2](https://github.com/delpapa/SORN_V2), for SORN network but they are found to be publication specific and are not general purpose software packages. However, `sorn` is flexible package that allows researchers to develop the network of their interest provided them the freedom to choose the combination of plasticity rules of their choice. Further, due to its easy to explain and integrate with machine learning frameworks like PyTorch and reinforcement learning toolkits like OpenAI Gym. Overall, `sorn` provide a research enviroment for computational neuroscients to investigate self organization, adaptation, learning, memory and behavior of brain circuits by reverse engineering neuronal plasticity mechanisms.

## Library Overview:

`sorn` package heavily depend on Numpy [@harris2020array] for numerical computations and analysis methods, seaborn and matplotlib [@barrett2005matplotlib] for visualization. The network is developed broadly in 5 classes; `SORN` object encapsulates all required functions that instantiate network variables like connection weights and thresholds. `Plasticity` inherits objects from `SORN` and implements plasticity rules using `stdp()`, `ip()`, `ss()`, `sp()`and `istdp()` methods . `NetworkState` has methods that evaluates excitatory and inhibitory network states at each timestep and finally the `MatrixCollection` objects behave as a memory cache. It collects the network states and keep track of the variables like weights and thresholds as the network evolves during simulation and training.

The network can be instantiated, simulated and trained using two classes `Simulator` and `Trainer` which inherit objects from `SORN`.

## SORN Network Model

Excitatory network state


\begin{equation}
\label{es}
x_i(t+1) =  \Theta\left (\sum_{j=1}^{N^E} {W_{ij}^{EE}(t)} {x_{j}(t)} - \sum_{j=1}^{N^I}W_{ik}^{EI}(t) y_{k}(t)+u_{i}(t) - T_{i}^{E}(t)+\xi_{E}(t)\right)
\end{equation}

Inhibitory Network state

\begin{equation}
\label{is}
y_i(t+1)=\Theta\left(\sum_{j=1}^{N_i}W_{ij}^{IE}(t) x_j(t)-T_i^I+ \xi_{I}(t)\right)
\end{equation}

## Plasticity Rules

### Spike Timing Dependent Plasticity

It changes the  synaptic efficacy between excitatory neurons  based on the spike- timing between pre $j$ and post synaptic neuron $i$.

\begin{equation}
\label{stdp}
\Delta W_{ij}^{EE}=\eta_{STDP}(x_i(t)x_j(t-1)-x_i(t-1)x_j(t)
\end{equation}

where,

$W_{ij}^{EE}$ -  Connection strength between excitatory neurons

$\eta_{STDP}$ - STDP learning rate

$x_j(t-1)$ - Presynaptic neuron state at $t-1$

$x_i$ - Postsynaptic neuron state at $t$


### Intrinsic Plasticity

IP update the firing threshold of excitatory neurons based on the state of the neuron at each time step. It increases the threshold if the neuron fires and decrease it otherwise.

\begin{equation}
\label{ip}
T_i(t+1)=T_i(t)+\eta_{IP}{x_i(t)-H_{IP}}
\end{equation}

where,

$T_i(t)$ - Firing threshold of the neuron $i$ at time $t$

$\eta_{IP}$ - Intrinsic plasticity step size

$H_{IP}$ - Target firing rate of the neuron

### Structural Plasticity

It is responsible for creating new synapses between excitatory neurons at a rate of approximately 1 connection per every 10th time step.

### Synaptic Scaling

SS normalizes the incoming synaptic strenghts of a neuron and prevent the network activity from attenuation or exploding.

\begin{equation}
\label{ss}
W_{ij}^{EE}(t)←W_{ij}^{EE}(t)/\sum{𝑊_{ij}^{EE}(t)}
\end{equation}

### Inhibitory Spike Timing Dependent Plasticity

iSTDP is responisble for controlling the synaptic strenghts from Inhibitory to Excitatory network.

\begin{equation}
\label{istdp}
\Delta W_{ij}^{EI}=\eta_{istdp}(y_j(t-1)\left(1-x_i(t)(1+\frac{1}{\mu_{ip}}))\right)
\end{equation}


where,

$W_{ij}^{EI}$ - Synaptic strenght from Inhibitory to excitatory network

$\eta_{istdp}$ - Inhibitory STDP learning rate

$\mu_{ip}$ - Mean firing rate of the neuron

Note that, the connection strength from excitatory to inhibitory ($W_{ij}^{IE}$) remain fixed at the intial state.

## Sample Simulation methods
```python

# Sample input
num_features = 10
time_steps = 200
inputs = numpy.random.rand(num_features,time_steps)

state_dict,E,I,R,C=Simulator.simulate_sorn(inputs=inputs,phase='plasticity',

                                        matrices=None,noise=True,

                                        time_steps=time_steps,_ne=200,

                                        _nu=num_features)
```
`simulate_sorn` returns the dictionary of network state variables at last time steps, excitatory and inhibitory network activity of the entire simulation period and also the recurrent activity and the count of active connections at each time steps. To resume the simulation, load the matrices returned at the previous step as,

```python
state_dict,E,I,R,C=Simulator.simulate_sorn(inputs=inputs,phase='plasticity',

                                        matrices=state_dict, noise=True,

                                        time_steps=time_steps,

                                        _ne = 200,_nu=num_features)

```

### Network Output Descriptions

`state_dict` - Dictionary of connection weights ($W_{ij}^{EE}$,$W_{ij}^{EI}$,$W_{ij}^{IE}$) ,

               Excitatory network activity ('E'),

               Inhibitory network activities('I'),

               Threshold values $T^E,T^I$

`E` - Collection of Excitatory network activity of entire simulation period

`I` - Collection of Inhibitory network activity of entire simulation period

`R` - Collection of Recurrent network activity of entire simulation period

`C` - List of number of active connections in the Excitatory pool at each time step


## Sample Training methods
```python
from sorn import Trainer
inputs = np.random.rand(num_features,1)

# Under all plasticity mechanisms
state_dict,E,I,R,C=Trainer.train_sorn(inputs=inputs,phase='plasticity',

                                    matrices=state_dict,

                                    _nu=num_features,time_steps=1)



# Resume the training without any plasticity mechanisms

state_dict,E,I,R,C=Trainer.train_sorn(inputs=inputs,phase='training',

                                    matrices=state_dict,

                                    _nu=num_features,time_steps=1)
```

To turn off any plasticity mechanisms during simulation or training phase, you can use freeze argument. For example to stop intrinsic plasticity during training phase,

```python

state_dict,E,I,R,C=Trainer.train_sorn(inputs=inputs,phase='plasticity',

                                    matrices=None,noise=True,

                                    time_steps=1,_ne=200,

                                    _nu=num_features,freeze=['ip'])

```
The other options for `freeze` argument are,

`stdp` - Spike Timing Dependent Plasticity

`ss` - Synaptic Scaling

`sp` - Structural Plasticity

`istdp` - Inhibitory Spike Timing Dependent Plasticity

Note: If you pass all above options to freeze, then the network will behave as Liquid State Machine(LSM)

The `simulate_sorn` and `train_sorn` methods accepts the following keyword arguments

| kwargs                |                                          Description                                       |
|-----------------------|--------------------------------------------------------------------------------------------|
| `inputs`              |  External stimulus                                                                         |
| `phase`               |  `plasticity` or `training`                                                                |
| `matrices`            |  `state_dict` to resume simulation otherwise `None` to intialize new network               |
| `time_steps`          |  `simulaton` total time steps. For `training` should be 1                                  |
| `noise`               |  If `True`, Gaussian white noise will be added to excitatory field potentials              |
| `freeze`              |  To drop any given plasticity mechanism(s) among [`'ip'`,`'stdp'`,`'istdp'`,`'ss'`, `'sp'`]|
| `_ne`                 |  Number of Excitatory neurons in the network                                               |
| `_nu`                 |  Number of input units among excitatory neurons                                            |
| `_network_type_ee`    |  `sparse` or  `dense` connection between excitatory neurons                                |
| `_network_type_ei`    |  `sparse` or  `dense` connection from inhibitory and excitatory neurons                    |
| `_network_type_ie`    |  `sparse` or  `dense` connection from excitatory and inhibitory neurons                    |
| `_lambda_ee`          |  Connection density between excitatory networks if network type is `sparse`                |
| `_lambda_ei`          |  Density of connections from inhibitory to excitatory networks if network type is `sparse` |
| `_lambda_ie`          |  Density of connections from inhibitory to excitatory networks if network type is `sparse` |
| `_eta_stdp`           |  Hebbian learning rate of excitatory synapses                                              |
| `_eta_inhib`          |  Hebbian learning rate synapses from inhibitory to excitatory                              |
| `_eta_ip`             |  Learning rate of excitatory neuron threshold                                              |
| `_te_max`             |  Maximum of excitatory neuron threshold range                                              |
| `_ti_max`             |  Maximum of inhibitory neuron threshold range                                              |
| `_ti_min`             |  Minimum of inhibitory neuron threshold range                                              |
| `_te_min`             |  Minimum of excitatory neuron threshold range                                              |
| `_mu_ip`              |  Target Mean firing rate of excitatory neuron                                              |
| `_sigma_ip`           |  Target Standard deviation of firing rate of excitatory neuron                             |


### Analysis functions

`sorn` package also includes necessary methods to investigate network properties. Few methods in `Statistics` are,


| methods                       |                                      Description                                    |
|-------------------------------|-------------------------------------------------------------------------------------|
| `autocorr`                    |  t-lagged auto correlation between neural activity                                  |
| `fanofactor`                  |  To verify poissonian process in spike generation of neuron(s)                      |
| `spike_source_entropy`        |  Measure the uncertainty about the origin of spike from the network using entropy   |
| `firing_rate_neuron`          |  Spike rate of specific neuron                                                      |
| `firing_rate_network`         |  Spike rate of entire network                                                       |
| `avg_corr_coeff`              |  Average Pearson correlation coeffecient between neurons                            |
| `spike_times`                 |  Time instants at which neuron spikes                                               |
| `spike_time_intervals`        |  Inter spike intervals for each neuron                                              |
| `hamming_distance`            |  Hamming distance between two network states                                        |

More details about the statistical and plotting tools in the package is found at ([https://self-organizing-recurrent-neural-networks.readthedocs.io/en/latest/](self-organizing-recurrent-neural-networks.readthedocs.io))

# References