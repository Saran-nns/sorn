## Self-Organizing Recurrent Neural Networks 

SORN is a class of neuro-inspired artificial network build based on plasticity mechanisms in biological brain and mimic neocortical circuits ability of learning and adaptation through neuroplasticity mechanisms.

For ease of maintanance, example use cases and the API(under developement) are moved to https://github.com/Saran-nns/PySORN_0.1 

[![Build Status](https://travis-ci.org/Saran-nns/sorn.svg?branch=master)](https://travis-ci.org/Saran-nns/sorn)
[![codecov](https://codecov.io/gh/Saran-nns/sorn/branch/master/graph/badge.svg)](https://codecov.io/gh/Saran-nns/sorn)
[![Join the chat at https://gitter.im/Self-Organizing-Recurrent-Neural-Networks](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/Self-Organizing-Recurrent-Neural-Networks?utm_source=badge&utm_medium=badge&utm_content=badge)
[![PyPI version](https://badge.fury.io/py/sorn.svg)](https://badge.fury.io/py/sorn)
![PyPI - Downloads](https://img.shields.io/pypi/dw/sorn.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2593671.svg)](https://doi.org/10.5281/zenodo.2593671)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h4 align="Left">SORN Reservoir and the evolution of synaptic efficacies</h4> 
<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/PySORN_0.1/master/v0.1.0/doc/images/SORN1.png" height="320" width="430" ></a> <a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/PySORN_0.1/master/v0.1.0/doc/images/weights.png" height="375" width="425" ></a>	

<h4 align="center">Neural Connectome</h4> 
<p align="center">
<a href="url"><img src="https://github.com/Saran-nns/PySORN_0.1/blob/master/v0.1.0/doc/images/neuralcorrelationall.png" height="450" width="450" ></a>
</p>

#### To install the latest release:

```python
pip install sorn
```

The library is still in alpha stage, so you may also want to install the latest version from the development branch:

```python
pip install git+https://github.com/Saran-nns/sorn
```

#### Dependencies
SORN supports Python 3.5+ ONLY. For older Python versions please use the official Python client


#### Usage:

##### Update Network configurations

Navigate to home/conda/envs/ENVNAME/Lib/site-packages/sorn

or if you are unsure about the directory of sorn

Run

```python
import sorn

sorn.__file__
```
to find the location of the sorn package

Then, update/edit the configuration.ini


##### Plasticity Phase

```Python
# Import 
from sorn.sorn import RunSorn

# Sample input 
inputs = [0.]

# To simulate the network; 
matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = RunSorn(phase='Plasticity', matrices=None,
                                                                          time_steps=100).run_sorn(inputs)

# To resume the simulation, load the matrices_dict from previous simulation;
matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = RunSorn(phase='Plasticity', matrices=matrices_dict,
                                                                          time_steps=100).run_sorn(inputs)
```

##### Training phase:

```Python
matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = RunSorn(phase='Training', matrices=matrices_dict,
                                                                          time_steps=100).run_sorn(inputs)
```

#### Network Output Descriptions:
    matrices_dict  - Dictionary of connection weights ('Wee','Wei','Wie') , Excitatory network activity ('X'), Inhibitory network activities('Y'), Threshold values ('Te','Ti')

    Exc_activity - Collection of Excitatory network activity of entire simulation period

    Inh_activitsy - Collection of Inhibitory network activity of entire simulation period

    Rec_activity - Collection of Recurrent network activity of entire simulation period

    num_active_connections - List of number of active connections in the Excitatory pool at each time step 


#### Sample use with OpenAI gym :
##### Cartpole balance problem
```python
# Imports

import utils.InitHelper as initializer
from sorn.sorn import Sorn, Plasticity, TrainSorn, TrainSornPlasticity
import gym

# Load the simulated network matrices
# Note these matrices are obtained after the network achieved convergence under random inputs and noise

with open('simulation_matrices.pkl','rb') as f:  
    sim_matrices,excit_states,inhib_states,recur_states,num_reservoir_conn = pickle.load(f)


# Training parameters

NUM_EPISODES = 2e6
NUM_PLASTICITY_EPISODES = 20000

env = gym.make('CartPole-v0')

for EPISODE in range(NUM_EPISODES):
    
    # Environment observation
    state = env.reset()[None,:]
    
    # Play the episode
    
    while True:
      
      if EPISODE < NUM_PLASTICITY_EPISODE:
      
        # Plasticity phase
        sim_matrices,excit_states,inhib_states,recur_states,num_reservoir_conn = TrainSornPlasticity.train_sorn(phase = 'Plasticity',
                                                                                                            matrices = sim_matrices,
                                                                                                            inputs = state)

      else:
        # Training phase with frozen reservoir connectivity
        sim_matrices,excit_states,inhib_states,recur_states,num_reservoir_conn = TrainSorn.train_sorn(phase = 'Training',
                                                                                                            matrices = sim_matrices,
                                                                                                            inputs = state)
      
      # Feed excit_states as input states to your RL algorithm, below goes for simple policy gradient algorithm
      # Sample policy w.r.t excitatory states and take action in the environment
       
      probs = policy(np.asarray(excit_states),output_layer_weights))
      action = np.random.choice(action_space,probs)
      state,reward,done,_ = env.step(action) 
      
      if done:
        break
```


#### Sample Plotting functions 

```Python
from sorn.utils import Plotter
# Plot weight distribution in the network
Plotter.weight_distribution(weights= matrices_dict['Wee'], bin_size = 5, savefig = False)

# Plot Spike train of all neurons in the network
Plotter.scatter_plot(spike_train = np.asarray(Exc_activity), savefig=False)


Plotter.raster_plot(spike_train = np.asarray(Exc_activity), savefig=False)
```

#### Sample Statistical analysis functions

```Python
from sorn.utils import Statistics
#t-lagged auto correlation between neural activity
Statistics.autocorr(firing_rates = [1,1,5,6,3,7],t= 2)

# Fano factor: To verify poissonian process in spike generation of neuron 10
Statistics.fanofactor(spike_train= np.asarray(Exc_activity),neuron = 10,window_size = 10)
```

#### The network is inspired by folowing articles:

Lazar, A. (2009). SORN: a Self-organizing Recurrent Neural Network. Frontiers in Computational Neuroscience, 3. https://doi.org/10.3389/neuro.10.023.2009

Hartmann, C., Lazar, A., Nessler, B., & Triesch, J. (2015). Where’s the Noise? Key Features of Spontaneous Activity and Neural Variability Arise through Learning in a Deterministic Network. PLoS Computational Biology, 11(12). https://doi.org/10.1371/journal.pcbi.1004640 

Del Papa, B., Priesemann, V., & Triesch, J. (2017). Criticality meets learning: Criticality signatures in a self-organizing recurrent neural network. PLoS ONE, 12(5). https://doi.org/10.1371/journal.pone.0178683 

Zheng, P., Dimitrakakis, C., & Triesch, J. (2013). Network Self-Organization Explains the Statistics and Dynamics of Synaptic Connection Strengths in Cortex. PLoS Computational Biology, 9(1). https://doi.org/10.1371/journal.pcbi.1002848  

