# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import os
from sorn.utils import Initializer
from configparser import ConfigParser
import random
import tqdm
import pickle


parser = ConfigParser()
cwd = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(cwd, 'configuration.ini')
parser.read(config_file)

class Sorn(object):

    # SORN network Initialization

    def __init__(self):
        pass

    """Get network variables from configuration file as class variables of SORN"""
    
    nu = int(parser.get('Network_Config', 'Nu'))  # Number of input units
    ne = int(parser.get('Network_Config', 'Ne'))  # Number of excitatory units
    ni = int(0.2 * ne)  # Number of inhibitory units in the network
    
    eta_stdp = float(parser.get('Network_Config', 'eta_stdp'))
    eta_inhib = float(parser.get('Network_Config', 'eta_inhib'))
    eta_ip = float(parser.get('Network_Config', 'eta_ip'))
    te_max = float(parser.get('Network_Config', 'te_max'))
    ti_max = float(parser.get('Network_Config', 'ti_max'))
    ti_min = float(parser.get('Network_Config', 'ti_min'))
    te_min = float(parser.get('Network_Config', 'te_min'))
    mu_ip = float(parser.get('Network_Config', 'mu_ip'))
    sigma_ip = float(parser.get('Network_Config', 'sigma_ip'))  # Standard deviation, variance == 0

    network_type_ee = str(parser.get('Network_Config','network_type_ee'))
    network_type_ei = str(parser.get('Network_Config','network_type_ei'))
    network_type_ie = str(parser.get('Network_Config','network_type_ie'))

    lambda_ee = int(parser.get('Network_Config','lambda_ee'))
    lambda_ei = int(parser.get('Network_Config','lambda_ei'))
    lambda_ie = int(parser.get('Network_Config','lambda_ie'))

    # Initialize weight matrices

    @staticmethod
    def initialize_weight_matrix(network_type, synaptic_connection, self_connection, lambd_w):

        """
        Args:

        network_type(str) - Spare or Dense
        synaptic_connection(str) - EE,EI,IE: Note that Spare connection is defined only for EE connections
        self_connection(str) - True or False: i-->i ; Network is tested only using j-->i
        lambd_w(int) - Average number of incoming and outgoing connections per neuron

        Returns:
        weight_matrix(array) -  Array of connection strengths
        """

        if (network_type == "Sparse") and (self_connection == "False"):

            """ Generate weight matrix for E-E/ E-I connections with mean lamda incoming and 
               out-going connections per neuron """

            weight_matrix = Initializer.generate_lambd_connections(synaptic_connection, Sorn.ne, Sorn.ni, lambd_w, lambd_std=1)

        # Dense matrix for W_ie

        elif (network_type == 'Dense') and (self_connection == 'False'):

            # Gaussian distribution of weights
            # weight_matrix = np.random.randn(Sorn.ne, Sorn.ni) + 2 # Small random values from gaussian distribution
            # Centered around 1
            # weight_matrix.reshape(Sorn.ne, Sorn.ni)
            # weight_matrix *= 0.01 # Setting spectral radius

            # Uniform distribution of weights
            weight_matrix = np.random.uniform(0.0, 0.1, (Sorn.ne, Sorn.ni))
            weight_matrix.reshape((Sorn.ne, Sorn.ni))

        return weight_matrix

    @staticmethod
    def initialize_threshold_matrix(te_min, te_max, ti_min, ti_max):

        # Initialize the threshold for excitatory and inhibitory neurons

        """Args:
            te_min(float) -- Min threshold value for excitatory units
            ti_min(float) -- Min threshold value for inhibitory units
            te_max(float) -- Max threshold value for excitatory units
            ti_max(float) -- Max threshold value for inhibitory units
        Returns:
            te(vector) -- Threshold values for excitatory units
            ti(vector) -- Threshold values for inhibitory units"""

        te = np.random.uniform(te_min, te_max, (Sorn.ne, 1))
        ti = np.random.uniform(ti_min, ti_max, (Sorn.ni, 1))

        return te, ti

    @staticmethod
    def initialize_activity_vector(ne, ni):

        # Initialize the activity vectors X and Y for excitatory and inhibitory neurons

        """Args:
            ne(int) -- Number of excitatory neurons
            ni(int) -- Number of inhibitory neurons
        Returns:
             x(array) -- Array of activity vectors of excitatory population
             y(array) -- Array of activity vectors of inhibitory population"""

        x = np.zeros((ne, 2))
        y = np.zeros((ni, 2))

        return x, y


class Plasticity(Sorn):
    """
    Instance of class Sorn. Inherits the variables and functions defined in class Sorn
    Encapsulates all plasticity mechanisms mentioned in the article """

    # Initialize the global variables for the class //Class attributes

    def __init__(self):

        super().__init__()
        self.nu = Sorn.nu  # Number of input units
        self.ne = Sorn.ne  # Number of excitatory units
        self.eta_stdp = Sorn.eta_stdp  # STDP plasticity Learning rate constant; SORN1 and SORN2
        self.eta_ip = Sorn.eta_ip  # Intrinsic plasticity learning rate constant; SORN1 and SORN2
        self.eta_inhib = Sorn.eta_inhib  # Intrinsic plasticity learning rate constant; SORN2 only
        self.h_ip = 2 * Sorn.nu / Sorn.ne  # Target firing rate
        self.mu_ip = Sorn.mu_ip  # Mean target firing rate
        self.ni = int(0.2 * Sorn.ne)  # Number of inhibitory units in the network
        self.time_steps = Sorn.time_steps  # Total time steps of simulation
        self.te_min = Sorn.te_min  # Excitatory minimum Threshold
        self.te_max = Sorn.te_max  # Excitatory maximum Threshold

    def stdp(self, wee, x, cutoff_weights):

        """ Apply STDP rule : Regulates synaptic strength between the pre(Xj) and post(Xi) synaptic neurons"""

        x = np.asarray(x)
        xt_1 = x[:, 0]
        xt = x[:, 1]
        wee_t = wee.copy()

        # STDP applies only on the neurons which are connected.

        for i in range(len(wee_t[0])):  # Each neuron i, Post-synaptic neuron

            for j in range(len(wee_t[0:])):  # Incoming connection from jth pre-synaptic neuron to ith neuron

                if wee_t[j][i] != 0.:  # Check connectivity

                    # Get the change in weight
                    delta_wee_t = self.eta_stdp * (xt[i] * xt_1[j] - xt_1[i] * xt[j])

                    # Update the weight between jth neuron to i ""Different from notation in article

                    wee_t[j][i] = wee[j][i] + delta_wee_t

        """ Prune the smallest weights induced by plasticity mechanisms; Apply lower cutoff weight"""
        wee_t = Initializer.prune_small_weights(wee_t, cutoff_weights[0])

        """Check and set all weights < upper cutoff weight """
        wee_t = Initializer.set_max_cutoff_weight(wee_t, cutoff_weights[1])

        return wee_t

    def ip(self, te, x):

        # IP rule: Active unit increases its threshold and inactive decreases its threshold.

        xt = x[:, 1]

        te_update = te + self.eta_ip * (xt.reshape(self.ne, 1) - self.h_ip)

        """ Check whether all te are in range [0.0,1.0] and update acordingly"""

        # Update te < 0.0 ---> 0.0
        # te_update = prune_small_weights(te_update,self.te_min)

        # Set all te > 1.0 --> 1.0
        # te_update = set_max_cutoff_weight(te_update,self.te_max)

        return te_update

    def ss(self, wee_t):

        """Synaptic Scaling or Synaptic Normalization"""

        wee_t = wee_t / np.sum(wee_t, axis=0)

        return wee_t

    def istdp(self, wei, x, y, cutoff_weights):

        #  Apply iSTDP rule : Regulates synaptic strength between the pre(Yj) and post(Xi) synaptic neurons

        # Excitatory network activity
        x = np.asarray(x)  # Array sanity check
        xt_1 = x[:, 0]
        xt = x[:, 1]

        # Inhibitory network activity
        y = np.asarray(y)

        yt_1 = y[:, 0]
        yt = y[:, 1]

        # iSTDP applies only on the neurons which are connected.
        wei_t = wei.copy()

        for i in range(len(wei_t[0])):  # Each neuron i, Post-synaptic neuron: means for each column;

            for j in range(len(wei_t[0:])):  # Incoming connection from j, pre-synaptic neuron to ith neuron

                if wei_t[j][i] != 0.:  # Check connectivity

                    # Get the change in weight
                    delta_wei_t = - self.eta_inhib * yt_1[j] * (1 - xt[i] * (1 + 1 / self.mu_ip))

                    # Update the weight between jth neuron to i ""Different from notation in article

                    wei_t[j][i] = wei[j][i] + delta_wei_t

        """ Prune the smallest weights induced by plasticity mechanisms; Apply lower cutoff weight"""
        wei_t = Initializer.prune_small_weights(wei_t, cutoff_weights[0])

        """Check and set all weights < upper cutoff weight """
        wei_t = Initializer.set_max_cutoff_weight(wei_t, cutoff_weights[1])

        return wei_t

    @staticmethod
    def structural_plasticity(wee):

        """ Add new connection value to the smallest weight between excitatory units randomly"""

        p_c = np.random.randint(0, 10, 1)

        if p_c == 0:  # p_c= 0.1

            """ Do structural plasticity """

            # Choose the smallest weights randomly from the weight matrix wee

            indexes = Initializer.get_unconnected_indexes(wee)

            # Choose any idx randomly
            idx_rand = random.choice(indexes)

            if idx_rand[0] == idx_rand[1]:
                idx_rand = random.choice(indexes)

            wee[idx_rand[0]][idx_rand[1]] = 0.001

        return wee


    @staticmethod
    def initialize_plasticity():

        """NOTE: DO NOT TRANSPOSE THE WEIGHT MATRIX WEI FOR SORN 2 MODEL"""

        # Create and initialize sorn object and variables

        sorn_init = Sorn()
        WEE_init = sorn_init.initialize_weight_matrix(network_type=Sorn.network_type_ee, synaptic_connection='EE',
                                                      self_connection='False',
                                                      lambd_w=Sorn.lambda_ee)
        WEI_init = sorn_init.initialize_weight_matrix(network_type=Sorn.network_type_ei, synaptic_connection='EI',
                                                      self_connection='False',
                                                      lambd_w=Sorn.lambda_ei)
        WIE_init = sorn_init.initialize_weight_matrix(network_type=Sorn.network_type_ie, synaptic_connection='IE',
                                                      self_connection='False',
                                                      lambd_w=Sorn.lambda_ie)

        Wee_init = Initializer.zero_sum_incoming_check(WEE_init)
        # Wei_init = initializer.zero_sum_incoming_check(WEI_init.T)  # For SORN 1
        Wei_init = Initializer.zero_sum_incoming_check(WEI_init)
        Wie_init = Initializer.zero_sum_incoming_check(WIE_init)

        c = np.count_nonzero(Wee_init)
        v = np.count_nonzero(Wei_init)
        b = np.count_nonzero(Wie_init)

        print('Network Initialized')
        print('Number of connections in Wee %s , Wei %s, Wie %s' %(c, v, b))
        print('Shapes Wee %s Wei %s Wie %s' % (Wee_init.shape, Wei_init.shape, Wie_init.shape))

        # Normalize the incoming weights

        normalized_wee = Initializer.normalize_weight_matrix(Wee_init)
        normalized_wei = Initializer.normalize_weight_matrix(Wei_init)
        normalized_wie = Initializer.normalize_weight_matrix(Wie_init)

        te_init, ti_init = sorn_init.initialize_threshold_matrix(Sorn.te_min, Sorn.te_max, Sorn.ti_min, Sorn.ti_max)
        x_init, y_init = sorn_init.initialize_activity_vector(Sorn.ne, Sorn.ni)

        # Initializing variables from sorn_initialize.py

        wee = normalized_wee.copy()
        wei = normalized_wei.copy()
        wie = normalized_wie.copy()
        te = te_init.copy()
        ti = ti_init.copy()
        x = x_init.copy()
        y = y_init.copy()

        return wee, wei, wie, te, ti, x, y

    @staticmethod
    def reorganize_network():

        return NotImplementedError


class MatrixCollection(Sorn):
    def __init__(self, phase, matrices=None):
        super().__init__()

        self.phase = phase
        self.matrices = matrices
        if self.phase == 'Plasticity' and self.matrices == None:

            self.time_steps = Sorn.time_steps + 1  # Total training steps
            self.Wee, self.Wei, self.Wie, self.Te, self.Ti, self.X, self.Y = [0] * self.time_steps, [
                0] * self.time_steps, \
                                                                             [0] * self.time_steps, [
                                                                                 0] * self.time_steps, \
                                                                             [0] * self.time_steps, [
                                                                          0] * self.time_steps, \
                                                                             [0] * self.time_steps
            wee, wei, wie, te, ti, x, y = Plasticity.initialize_plasticity()

            # Assign initial matrix to the master matrices
            self.Wee[0] = wee
            self.Wei[0] = wei
            self.Wie[0] = wie
            self.Te[0] = te
            self.Ti[0] = ti
            self.X[0] = x
            self.Y[0] = y

        elif self.phase == 'Plasticity' and self.matrices != None:

            self.time_steps = Sorn.time_steps + 1  # Total training steps
            self.Wee, self.Wei, self.Wie, self.Te, self.Ti, self.X, self.Y = [0] * self.time_steps, [
                0] * self.time_steps, \
                                                                             [0] * self.time_steps, [
                                                                                 0] * self.time_steps, \
                                                                             [0] * self.time_steps, [
                                                                                 0] * self.time_steps, \
                                                                             [0] * self.time_steps
            # Assign matrices from plasticity phase to the new master matrices for training phase
            self.Wee[0] = matrices['Wee']
            self.Wei[0] = matrices['Wei']
            self.Wie[0] = matrices['Wie']
            self.Te[0] = matrices['Te']
            self.Ti[0] = matrices['Ti']
            self.X[0] = matrices['X']
            self.Y[0] = matrices['Y']

        elif self.phase == 'Training':

            """NOTE:
            time_steps here is diferent for plasticity and training phase"""
            self.time_steps = Sorn.time_steps + 1  # Total training steps
            self.Wee, self.Wei, self.Wie, self.Te, self.Ti, self.X, self.Y = [0] * self.time_steps, [
                0] * self.time_steps, \
                                                                             [0] * self.time_steps, [
                                                                                 0] * self.time_steps, \
                                                                             [0] * self.time_steps, [
                                                                                 0] * self.time_steps, \
                                                                             [0] * self.time_steps
            # Assign matrices from plasticity phase to new respective matrices for training phase
            self.Wee[0] = matrices['Wee']
            self.Wei[0] = matrices['Wei']
            self.Wie[0] = matrices['Wie']
            self.Te[0] = matrices['Te']
            self.Ti[0] = matrices['Ti']
            self.X[0] = matrices['X']
            self.Y[0] = matrices['Y']

    # @staticmethod
    def weight_matrix(self, wee, wei, wie, i):
        # Get delta_weight from Plasticity.stdp

        # i - training step
        self.Wee[i + 1] = wee
        self.Wei[i + 1] = wei
        self.Wie[i + 1] = wie

        return self.Wee, self.Wei, self.Wie

    # @staticmethod
    def threshold_matrix(self, te, ti, i):
        self.Te[i + 1] = te
        self.Ti[i + 1] = ti
        return self.Te, self.Ti

    # @staticmethod
    def network_activity_t(self, excitatory_net, inhibitory_net, i):
        self.X[i + 1] = excitatory_net
        self.Y[i + 1] = inhibitory_net

        return self.X, self.Y

    # @staticmethod
    def network_activity_t_1(self, x, y, i):
        x_1, y_1 = [0] * self.time_steps, [0] * self.time_steps
        x_1[i] = x
        y_1[i] = y

        return x_1, y_1


class NetworkState(Plasticity):
    """The evolution of network states"""

    def __init__(self, v_t):
        super().__init__()
        self.v_t = v_t

    def incoming_drive(self, weights, activity_vector):

        # Broadcasting weight*acivity vectors

        incoming = weights * activity_vector
        incoming = np.array(incoming.sum(axis=0))
        return incoming

    def excitatory_network_state(self, wee, wei, te, x, y, white_noise_e):

        """ Activity of Excitatory neurons in the network"""

        xt = x[:, 1]
        xt = xt.reshape(self.ne, 1)
        yt = y[:, 1]
        yt = yt.reshape(self.ni, 1)

        incoming_drive_e = np.expand_dims(self.incoming_drive(weights=wee, activity_vector=xt), 1)
        incoming_drive_i = np.expand_dims(self.incoming_drive(weights=wei, activity_vector=yt), 1)

        tot_incoming_drive = incoming_drive_e - incoming_drive_i + white_noise_e + np.asarray(self.v_t) - te

        """Heaviside step function"""
        heaviside_step = np.expand_dims([0.] * len(tot_incoming_drive),1)
        heaviside_step[tot_incoming_drive > 0] = 1.
        
        xt_next = np.asarray(heaviside_step.copy())  # Additional Memory cost just for the sake of variable name

        return xt_next

    def inhibitory_network_state(self, wie, ti, x, white_noise_i):

        # Activity of inhibitory neurons

        wie = np.asarray(wie)
        xt = x[:, 1]
        xt = xt.reshape(Sorn.ne, 1)

        incoming_drive_e = np.expand_dims(self.incoming_drive(weights=wie, activity_vector=xt), 1)

        tot_incoming_drive = incoming_drive_e + white_noise_i - ti

        heaviside_step = np.expand_dims([0.] * len(tot_incoming_drive),1)
        heaviside_step[tot_incoming_drive > 0] = 1.

        yt_next = np.asarray(heaviside_step.copy())  # Additional Memory cost just for the sake of variable name

        return yt_next

    def recurrent_drive(self, wee, wei, te, x, y, white_noise_e):

        """Network state due to recurrent drive received by the each unit at time t+1"""

        xt = x[:, 1]
        xt = xt.reshape(self.ne, 1)
        yt = y[:, 1]
        yt = yt.reshape(self.ni, 1)

        incoming_drive_e = np.expand_dims(self.incoming_drive(weights=wee, activity_vector=xt), 1)
        incoming_drive_i = np.expand_dims(self.incoming_drive(weights=wei, activity_vector=yt), 1)

        tot_incoming_drive = incoming_drive_e - incoming_drive_i + white_noise_e - te

        heaviside_step = np.expand_dims([0.] * len(tot_incoming_drive),1)
        heaviside_step[tot_incoming_drive > 0] = 1.

        xt_next = np.asarray(heaviside_step.copy())  # Additional Memory cost just for the sake of variable name

        return xt_next


# Simulate / Train SORN
class RunSorn(Sorn):

    def __init__(self, phase, matrices, time_steps):
        super().__init__()
        self.time_steps = time_steps
        Sorn.time_steps = time_steps
        self.phase = phase
        self.matrices = matrices

    def run_sorn(self, inp):
        # Initialize/Get the weight, threshold matrices and activity vectors
        matrix_collection = MatrixCollection(phase=self.phase, matrices=self.matrices)

        # Collect the network activity at all time steps

        X_all = [0] * self.time_steps
        Y_all = [0] * self.time_steps
        R_all = [0] * self.time_steps

        frac_pos_active_conn = []

        # To get the last activation status of Exc and Inh neurons

        for i in tqdm.tqdm(range(self.time_steps)):
            """ Generate white noise"""
            white_noise_e = Initializer.white_gaussian_noise(mu=0., sigma=0.04, t=Sorn.ne)
            white_noise_i = Initializer.white_gaussian_noise(mu=0., sigma=0.04, t=Sorn.ni)

            # Generate inputs
            # inp_ = np.expand_dims(Initializer.generate_normal_inp(10), 1)

            network_state = NetworkState(inp)  # Feed input and initialize network state

            # Buffers to get the resulting x and y vectors at the current time step and update the master matrix

            x_buffer, y_buffer = np.zeros((Sorn.ne, 2)), np.zeros((Sorn.ni, 2))

            # TODO: Return te,ti values in next version # UNUSED 
            te_buffer, ti_buffer = np.zeros((Sorn.ne, 1)), np.zeros((Sorn.ni, 1))

            # Get the matrices and rename them for ease of reading

            Wee, Wei, Wie = matrix_collection.Wee, matrix_collection.Wei, matrix_collection.Wie
            Te, Ti = matrix_collection.Te, matrix_collection.Ti
            X, Y = matrix_collection.X, matrix_collection.Y

            """ Fraction of active connections between E-E network"""
            frac_pos_active_conn.append((Wee[i] > 0.0).sum())

            """ Recurrent drive"""

            r = network_state.recurrent_drive(Wee[i], Wei[i], Te[i], X[i], Y[i], white_noise_e)

            """Get excitatory states and inhibitory states given the weights and thresholds"""

            # x(t+1), y(t+1)
            excitatory_state_xt_buffer = network_state.excitatory_network_state(Wee[i], Wei[i], Te[i], X[i], Y[i],
                                                                                white_noise_e)

            inhibitory_state_yt_buffer = network_state.inhibitory_network_state(Wie[i], Ti[i], X[i], white_noise_i)

            """ Update X and Y """
            x_buffer[:, 0] = X[i][:, 1]  # xt -->(becomes) xt_1
            x_buffer[:, 1] = excitatory_state_xt_buffer.T  # New_activation; x_buffer --> xt

            y_buffer[:, 0] = Y[i][:, 1]
            y_buffer[:, 1] = inhibitory_state_yt_buffer.T

            """Plasticity phase"""

            plasticity = Plasticity()

            # TODO
            # Can be initialised outside loop--> Plasticity will receive dynamic args in future version

            # STDP
            Wee_t = plasticity.stdp(Wee[i], x_buffer, cutoff_weights=(0.0, 1.0))

            # Intrinsic plasticity
            Te_t = plasticity.ip(Te[i], x_buffer)

            # Structural plasticity
            Wee_t = plasticity.structural_plasticity(Wee_t)

            # iSTDP
            Wei_t = plasticity.istdp(Wei[i], x_buffer, y_buffer, cutoff_weights=(0.0, 1.0))

            # Synaptic scaling Wee
            Wee_t = Plasticity().ss(Wee_t)

            # Synaptic scaling Wei
            Wei_t = Plasticity().ss(Wei_t)

            """Assign the matrices to the matrix collections"""
            matrix_collection.weight_matrix(Wee_t, Wei_t, Wie[i], i)
            matrix_collection.threshold_matrix(Te_t, Ti[i], i)
            matrix_collection.network_activity_t(x_buffer, y_buffer, i)

            X_all[i] = x_buffer[:, 1]
            Y_all[i] = y_buffer[:, 1]
            R_all[i] = r

        plastic_matrices = {'Wee': matrix_collection.Wee[-1],
                            'Wei': matrix_collection.Wei[-1],
                            'Wie': matrix_collection.Wie[-1],
                            'Te': matrix_collection.Te[-1], 'Ti': matrix_collection.Ti[-1],
                            'X': X[-1], 'Y': Y[-1]}

        return plastic_matrices, X_all, Y_all, R_all, frac_pos_active_conn

class Generator(object):

    def __init__(self):
        pass

    def get_initial_matrices(self):
        
        return Plasticity.initialize_plasticity()

# SAMPLE USAGE

"""
# Start the Simulation step 

# Used only during linear output layer optimization: During simulation, use input generator from utils

_inputs = None  # Can also simulate the network without inputs else pass the input values here

#  During first batch of training; Pass matrices as None:
# SORN will initialize the matrices based on the configuration settings

plastic_matrices, X_all, Y_all, R_all, frac_pos_active_conn = RunSorn(phase='Plasticity', matrices=None,
                                                                          time_steps=10000).run_sorn(_inputs)

# Pickle the simulaion matrices for reuse

with open('stdp2013_10k.pkl', 'wb') as f:
    pickle.dump([plastic_matrices, X_all, Y_all, R_all, frac_pos_active_conn], f)

# While re simulate the network using any already simulated/ acquired matrices

with open('stdp2013_10k.pkl', 'rb') as f:
    plastic_matrices, X_all, Y_all, R_all, frac_pos_active_conn = pickle.load(f)


plastic_matrices1, X_all1, Y_all1, R_all1, frac_pos_active_conn1 = RunSorn(phase='Plasticity',
                                                                               matrices=plastic_matrices,
                                                                               time_steps=20000).run_sorn(inp=None)
"""

