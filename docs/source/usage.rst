Usage
=====

Update network configuration
----------------------------

Configuration file
^^^^^^^^^^^^^^^^^^
Navigate to `home/conda/envs/ENVNAME/Lib/site-packages/sorn` or if you are unsure about the directory of `sorn`:

Run::

    import sorn
    sorn.__file__

to find the location of the sorn package:

Then, update/edit arguments in `configuration.ini`

Keyword arguments 
^^^^^^^^^^^^^^^^^
Pass the arguments with valid names (listed below). This will override the default values at `configuration.ini`: 

The allowed `kwargs` are::

    kwargs_ = ['ne', 'nu', 'network_type_ee', 'network_type_ei', 'network_type_ie', 
               'lambda_ee','lambda_ei', 'lambda_ie', 'eta_stdp','eta_inhib', 'eta_ip', 
               'te_max', 'ti_max', 'ti_min', 'te_min', 'mu_ip','sigma_ip']

Simulation
----------
Plasticity Phase
^^^^^^^^^^^^^^^^
The default ne, nu values are overriden by passing them as kwargs insidesimulate_sorn method::

    # Import 
    from sorn import Simulator
    import numpy as np

    # Sample input 
    num_features = 10
    time_steps = 200
    inputs = np.random.rand(num_features,time_steps)

    # To simulate the network
    matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Simulator.simulate_sorn(inputs = inputs, phase='plasticity', matrices=None, noise = True, time_steps=time_steps, ne = 200, nu=num_features)

    # To resume the simulation, load the matrices_dict from previous simulation
    matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Simulator.simulate_sorn(inputs = inputs, phase='plasticity', matrices=matrices_dict, noise= True, time_steps=time_steps,ne = 200, nu=num_features)

Training Phase
^^^^^^^^^^^^^^
Train the network using the matrices obtained from plasticity phase::

    # Import
    from sorn import Trainer
    # Sample input
    inputs = np.random.rand(num_features,1) 

    # SORN network is frozen during training phase
    matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = Trainer.train_sorn(inputs = inputs, phase='Training', matrices=matrices_dict,nu=num_features, time_steps=1)

Network Output Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
During plasticity and training phase, SORN returns a dictionary of below matrices for analysis:

    `matrices_dict` - Dictionary of connection weights ('Wee','Wei','Wie') , Excitatory network activity ('X'), Inhibitory network activities('Y'), Threshold values ('Te','Ti')

    `Exc_activity` - Collection of Excitatory network activity of entire simulation period

    `Inh_activity` - Collection of Inhibitory network activity of entire simulation period

    `Rec_activity` - Collection of Recurrent network activity of entire simulation period

    `num_active_connections` - List of number of active connections in the Excitatory pool at each time step

Sample use with OpenAI gym
^^^^^^^^^^^^^^^^^^^^^^^^^^
`Cartpole balance problem without changing the default network parameters`::

    # Imports

    from sorn import Simulator, Trainer
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
            sim_matrices,excit_states,inhib_states,recur_states,num_reservoir_conn = Simulator.simulate_sorn(inputs = state, phase ='plasticity', matrices = sim_matrices, noise=False)

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