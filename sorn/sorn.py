# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import os
from configparser import ConfigParser
import random

try:
    from sorn.utils import Initializer
except:
    from utils import Initializer

parser = ConfigParser()
config_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configuration.ini"
)
parser.read(config_file)


class Sorn(object):

    """ This class wraps initialization of the network and its parameters

    Args:
        nu (int): Number of input units. Defaults to 10

        ne (int): Sorn._ne  # Number of excitatory units. Defaults to 200

        eta_stdp (float): STDP plasticity Learning rate constant; SORN1 and SORN2. Defaults to 0.004

        eta_ip (float): Intrinsic plasticity learning rate constant; SORN1 and SORN2. Defaults to 0.001

        eta_inhib (float): Intrinsic plasticity learning rate constant; SORN2 only. Defaults to 0.01

        h_ip (float): Target firing rate. Defaults to 2 * Sorn._nu / Sorn._ne

        mu_ip (float): Mean target firing rate. Defaults to 0.1

        sigma_ip (float): Variance of target firing rate. Defaults to 0.0

        ni (int): Number of inhibitory units in the network. Defaults to int(0.2 * Sorn._ne)

        time_steps (float): Total time steps of simulation

        te_min (float): Excitatory minimum Threshold. Defaults to 0.0

        te_max (float): Excitatory maximum Threshold. Defaults to 1.0

        ti_min (float): Inhibitory minimum Threshold. Defaults to 0.0

        ti_max (float): Inhibitory maximum Threshold. Defaults to 0.5

        network_type_ee (str): Dense or Sparse. Defaults to Sparse

        network_type_ei (str): Dense or Sparse. Defaults to Sparse

        network_type_ie (str): Dense or Sparse. Defaults to Dense

        lambda_ee (int):  Number of connections to and from a single excitatory unit to another at initialization. Defaults to 20

        lambda_ei (int): Number of connections to and from a single inhibitory unit to exitatory unit at initialization. Defaults to 40

        lambda_ie (int):  Number of connections to and from a single excitatory unit to inhibitory unit at initialization. Defaults to 100"""

    def __init__(self):
        pass

    _nu = int(parser.get("Network_Config", "Nu"))
    _ne = int(parser.get("Network_Config", "Ne"))
    _ni = int(0.2 * _ne)
    _eta_stdp = float(parser.get("Network_Config", "eta_stdp"))
    _eta_inhib = float(parser.get("Network_Config", "eta_inhib"))
    _eta_ip = float(parser.get("Network_Config", "eta_ip"))
    _te_max = float(parser.get("Network_Config", "te_max"))
    _ti_max = float(parser.get("Network_Config", "ti_max"))
    _ti_min = float(parser.get("Network_Config", "ti_min"))
    _te_min = float(parser.get("Network_Config", "te_min"))
    _mu_ip = float(parser.get("Network_Config", "mu_ip"))
    _sigma_ip = float(parser.get("Network_Config", "sigma_ip"))
    _network_type_ee = str(parser.get("Network_Config", "network_type_ee"))
    _network_type_ei = str(parser.get("Network_Config", "network_type_ei"))
    _network_type_ie = str(parser.get("Network_Config", "network_type_ie"))
    _lambda_ee = int(parser.get("Network_Config", "lambda_ee"))
    _lambda_ei = int(parser.get("Network_Config", "lambda_ei"))
    _lambda_ie = int(parser.get("Network_Config", "lambda_ie"))

    @staticmethod
    def initialize_weight_matrix(
        network_type: str, synaptic_connection: str, self_connection: str, lambd_w: int
    ):
        """Wrapper for initializing the weight matrices for SORN

        Args:
            network_type (str): Spare or Dense
            synaptic_connection (str): EE,EI,IE. Note that Spare connection is defined only for EE connections
            self_connection (str): True or False: Synaptic delay or time delay
            lambd_w (int): Average number of incoming and outgoing connections per neuron

        Returns:
            weight_matrix (array): Array of connection strengths
        """

        if (network_type == "Sparse") and (self_connection == "False"):

            # Generate weight matrix for E-E/ E-I connections with mean lamda incoming and out-going connections per neuron
            assert (lambd_w <= Sorn._ne), "Number of connections per unit (lambda) should be less than number of units(Ne) in the pool and also Ne should be greater than 25"
            weight_matrix = Initializer.generate_lambd_connections(
                synaptic_connection, Sorn._ne, Sorn._ni, lambd_w, lambd_std=1
            )

        # Dense matrix for W_ie
        elif (network_type == "Dense") and (self_connection == "False"):

            # Uniform distribution of weights
            weight_matrix = np.random.uniform(0.0, 0.1, (Sorn._ne, Sorn._ni))
            weight_matrix.reshape((Sorn._ne, Sorn._ni))

        return weight_matrix

    @staticmethod
    def initialize_threshold_matrix(
        te_min: float, te_max: float, ti_min: float, ti_max: float
    ):
        """Initialize the threshold for excitatory and inhibitory neurons

        Args:
            te_min (float): Min threshold value for excitatory units
            te_max (float): Min threshold value for inhibitory units
            ti_min (float): Max threshold value for excitatory units
            ti_max (float): Max threshold value for inhibitory units

        Returns:
            te (array): Threshold values for excitatory units
            ti (array): Threshold values for inhibitory units
        """

        te = np.random.uniform(te_min, te_max, (Sorn._ne, 1))
        ti = np.random.uniform(ti_min, ti_max, (Sorn._ni, 1))

        return te, ti

    @staticmethod
    def initialize_activity_vector(ne: int, ni: int):
        """Initialize the activity vectors X and Y for excitatory and inhibitory neurons

        Args:
            ne (int): Number of excitatory neurons

            ni (int): Number of inhibitory neurons

        Returns:
            x (array): Array of activity vectors of excitatory population

            y (array): Array of activity vectors of inhibitory population"""

        x = np.zeros((ne, 2))
        y = np.zeros((ni, 2))

        return x, y

class Plasticity(Sorn):
    """Instance of class Sorn. Inherits the variables and functions defined in class Sorn.
    It encapsulates all plasticity mechanisms mentioned in the article. Inherits all attributed from parent class Sorn

    Args:
        nu (int): Number of input units. Defaults to 10

        ne (int): Sorn._ne  # Number of excitatory units. Defaults to 200

        eta_stdp (float): STDP plasticity Learning rate constant; SORN1 and SORN2. Defaults to 0.004

        eta_ip (float): Intrinsic plasticity learning rate constant; SORN1 and SORN2. Defaults to 0.001

        eta_inhib (float): Intrinsic plasticity learning rate constant; SORN2 only. Defaults to 0.01

        h_ip (float): Target firing rate. Defaults to 2 * Sorn._nu / Sorn._ne

        mu_ip (float): Mean target firing rate. Defaults to 0.1

        sigma_ip (float): Variance of target firing rate. Defaults to 0.0

        ni (int): Number of inhibitory units in the network. Defaults to int(0.2 * Sorn._ne)

        time_steps (float): Total time steps of simulation

        te_min (float): Excitatory minimum Threshold. Defaults to 0.0

        te_max (float): Excitatory maximum Threshold. Defaults to 1.0

        ti_min (float): Inhibitory minimum Threshold. Defaults to 0.0

        ti_max (float): Inhibitory maximum Threshold. Defaults to 0.5
    """

    def __init__(self):

        super().__init__()
        self.nu = Sorn._nu  # Number of input units
        self.ne = Sorn._ne  # Number of excitatory units
        self.eta_stdp = (
            Sorn._eta_stdp
        )  # STDP plasticity Learning rate constant; SORN1 and SORN2
        self.eta_ip = (
            Sorn._eta_ip
        )  # Intrinsic plasticity learning rate constant; SORN1 and SORN2
        self.eta_inhib = (
            Sorn._eta_inhib
        )  # Intrinsic plasticity learning rate constant; SORN2 only
        self.h_ip = 2 * Sorn._nu / Sorn._ne  # Target firing rate
        self.mu_ip = Sorn._mu_ip  # Mean target firing rate
        # Number of inhibitory units in the network
        self.ni = int(0.2 * Sorn._ne)
        self.time_steps = Sorn._time_steps  # Total time steps of simulation
        self.te_min = Sorn._te_min  # Excitatory minimum Threshold
        self.te_max = Sorn._te_max  # Excitatory maximum Threshold

    def stdp(self, wee: np.array, x: np.array, cutoff_weights: list):
        """Apply STDP rule : Regulates synaptic strength between the pre(Xj) and post(Xi) synaptic neurons

        Args:
            wee (array):  Weight matrix

            x (array): Excitatory network activity

            cutoff_weights (list): Maximum and minimum weight ranges

        Returns:
            wee (array):  Weight matrix
        """

        x = np.asarray(x)
        xt_1 = x[:, 0]
        xt = x[:, 1]
        wee_t = wee.copy()

        # STDP applies only on the neurons which are connected.

        for i in range(len(wee_t[0])):  # Each neuron i, Post-synaptic neuron

            for j in range(
                len(wee_t[0:])
            ):  # Incoming connection from jth pre-synaptic neuron to ith neuron

                if wee_t[j][i] != 0.0:  # Check connectivity

                    # Get the change in weight
                    delta_wee_t = self.eta_stdp * \
                        (xt[i] * xt_1[j] - xt_1[i] * xt[j])

                    # Update the weight between jth neuron to i ""Different from notation in article

                    wee_t[j][i] = wee[j][i] + delta_wee_t

        # Prune the smallest weights induced by plasticity mechanisms; Apply lower cutoff weight
        wee_t = Initializer.prune_small_weights(wee_t, cutoff_weights[0])

        # Check and set all weights < upper cutoff weight
        wee_t = Initializer.set_max_cutoff_weight(wee_t, cutoff_weights[1])

        return wee_t

    def ip(self, te: np.array, x: np.array):
        """Intrinsic Plasiticity mechanism

        Args:
            te (array): Threshold vector of excitatory units

            x (array): Excitatory network activity

        Returns:
            te (array): Threshold vector of excitatory units
        """

        # IP rule: Active unit increases its threshold and inactive decreases its threshold.
        xt = x[:, 1]

        te_update = te + self.eta_ip * (xt.reshape(self.ne, 1) - self.h_ip)

        # Check whether all te are in range [0.0,1.0] and update acordingly

        # Update te < 0.0 ---> 0.0
        # te_update = prune_small_weights(te_update,self.te_min)

        # Set all te > 1.0 --> 1.0
        # te_update = set_max_cutoff_weight(te_update,self.te_max)

        return te_update

    def ss(self, wee: np.array):
        """Synaptic Scaling or Synaptic Normalization

        Args:
            wee (array):  Weight matrix

        Returns:
            wee (array):  Scaled Weight matrix
        """
        wee = wee / np.sum(wee, axis=0)
        return wee

    def istdp(self, wei: np.array, x: np.array, y: np.array, cutoff_weights: list):
        """Apply iSTDP rule, which regulates synaptic strength between the pre inhibitory(Xj) and post Excitatory(Xi) synaptic neurons

        Args:
            wei (array): Synaptic strengths from inhibitory to excitatory

            x (array): Excitatory network activity

            y (array): Inhibitory network activity

            cutoff_weights (list): Maximum and minimum weight ranges

        Returns:
            wei (array): Synaptic strengths from inhibitory to excitatory"""

        # Excitatory network activity
        xt = np.asarray(x)[:, 1]

        # Inhibitory network activity
        yt_1 = np.asarray(y)[:, 0]

        # iSTDP applies only on the neurons which are connected.
        wei_t = wei.copy()

        for i in range(
            len(wei_t[0])
        ):  # Each neuron i, Post-synaptic neuron: means for each column;

            for j in range(
                len(wei_t[0:])
            ):  # Incoming connection from j, pre-synaptic neuron to ith neuron

                if wei_t[j][i] != 0.0:  # Check connectivity

                    # Get the change in weight
                    delta_wei_t = (
                        -self.eta_inhib * yt_1[j] *
                        (1 - xt[i] * (1 + 1 / self.mu_ip))
                    )

                    # Update the weight between jth neuron to i ""Different from notation in article

                    wei_t[j][i] = wei[j][i] + delta_wei_t

        # Prune the smallest weights induced by plasticity mechanisms; Apply lower cutoff weight
        wei_t = Initializer.prune_small_weights(wei_t, cutoff_weights[0])

        # Check and set all weights < upper cutoff weight
        wei_t = Initializer.set_max_cutoff_weight(wei_t, cutoff_weights[1])

        return wei_t

    @staticmethod
    def structural_plasticity(wee: np.array):
        """Add new connection value to the smallest weight between excitatory units randomly

        Args:
            wee (array): Weight matrix

        Returns:
            wee (array):  Weight matrix"""

        p_c = np.random.randint(0, 10, 1)

        if p_c == 0:  # p_c= 0.1

            # Do structural plasticity
            # Choose the smallest weights randomly from the weight matrix wee
            indexes = Initializer.get_unconnected_indexes(wee)

            # Choose any idx randomly such that i!=j
            while True:
                idx_rand = random.choice(indexes)
                if idx_rand[0] != idx_rand[1]:
                    break

            wee[idx_rand[0]][idx_rand[1]] = 0.001

        return wee

    @staticmethod
    def initialize_plasticity():
        """Initialize weight matrices for plasticity phase based on network configuration

        Args:
            kwargs (self.__dict__): All arguments are inherited from Sorn attributes

        Returns:
            tuple(array): Weight matrices WEI, WEE, WIE and threshold matrices Te, Ti and Initial state vectors X,Y """

        sorn_init = Sorn()
        WEE_init = sorn_init.initialize_weight_matrix(
            network_type=Sorn._network_type_ee,
            synaptic_connection="EE",
            self_connection="False",
            lambd_w=Sorn._lambda_ee,
        )
        WEI_init = sorn_init.initialize_weight_matrix(
            network_type=Sorn._network_type_ei,
            synaptic_connection="EI",
            self_connection="False",
            lambd_w=Sorn._lambda_ei,
        )
        WIE_init = sorn_init.initialize_weight_matrix(
            network_type=Sorn._network_type_ie,
            synaptic_connection="IE",
            self_connection="False",
            lambd_w=Sorn._lambda_ie,
        )

        Wee_init = Initializer.zero_sum_incoming_check(WEE_init)
        # Wei_init = initializer.zero_sum_incoming_check(WEI_init.T)  # For SORN 1
        Wei_init = Initializer.zero_sum_incoming_check(WEI_init)
        Wie_init = Initializer.zero_sum_incoming_check(WIE_init)

        c = np.count_nonzero(Wee_init)
        v = np.count_nonzero(Wei_init)
        b = np.count_nonzero(Wie_init)

        print("Network Initialized")
        print("Number of connections in Wee %s , Wei %s, Wie %s" % (c, v, b))
        print(
            "Shapes Wee %s Wei %s Wie %s"
            % (Wee_init.shape, Wei_init.shape, Wie_init.shape)
        )

        # Normalize the incoming weights

        normalized_wee = Initializer.normalize_weight_matrix(Wee_init)
        normalized_wei = Initializer.normalize_weight_matrix(Wei_init)
        normalized_wie = Initializer.normalize_weight_matrix(Wie_init)

        te_init, ti_init = sorn_init.initialize_threshold_matrix(
            Sorn._te_min, Sorn._te_max, Sorn._ti_min, Sorn._ti_max
        )
        x_init, y_init = sorn_init.initialize_activity_vector(
            Sorn._ne, Sorn._ni)

        # Initializing variables from sorn_initialize.py

        wee = normalized_wee.copy()
        wei = normalized_wei.copy()
        wie = normalized_wie.copy()
        te = te_init.copy()
        ti = ti_init.copy()
        x = x_init.copy()
        y = y_init.copy()

        return wee, wei, wie, te, ti, x, y


class MatrixCollection(Sorn):
    """Collect all matrices initialized and updated during simulation (plasiticity and training phases)

    Args:
        phase (str): Training or Plasticity phase

        matrices (dict): Network activity, threshold and connection matrices

    Returns:
        MatrixCollection instance"""

    def __init__(self, phase: str, matrices: dict = None):
        super().__init__()

        self.phase = phase
        self.matrices = matrices
        if self.phase == "plasticity" and self.matrices == None:

            self.time_steps = Sorn._time_steps + 1  # Total training steps
            self.Wee, self.Wei, self.Wie, self.Te, self.Ti, self.X, self.Y = (
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
            )
            wee, wei, wie, te, ti, x, y = Plasticity.initialize_plasticity()

            # Assign initial matrix to the master matrices
            self.Wee[0] = wee
            self.Wei[0] = wei
            self.Wie[0] = wie
            self.Te[0] = te
            self.Ti[0] = ti
            self.X[0] = x
            self.Y[0] = y

        elif self.phase == "plasticity" and self.matrices != None:

            self.time_steps = Sorn._time_steps + 1  # Total training steps
            self.Wee, self.Wei, self.Wie, self.Te, self.Ti, self.X, self.Y = (
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
            )
            # Assign matrices from plasticity phase to the new master matrices for training phase
            self.Wee[0] = matrices["Wee"]
            self.Wei[0] = matrices["Wei"]
            self.Wie[0] = matrices["Wie"]
            self.Te[0] = matrices["Te"]
            self.Ti[0] = matrices["Ti"]
            self.X[0] = matrices["X"]
            self.Y[0] = matrices["Y"]

        elif self.phase == "training":

            # NOTE:time_steps here is diferent for plasticity and training phase
            self.time_steps = Sorn._time_steps + 1  # Total training steps
            self.Wee, self.Wei, self.Wie, self.Te, self.Ti, self.X, self.Y = (
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
                [0] * self.time_steps,
            )
            # Assign matrices from plasticity phase to new respective matrices for training phase
            self.Wee[0] = matrices["Wee"]
            self.Wei[0] = matrices["Wei"]
            self.Wie[0] = matrices["Wie"]
            self.Te[0] = matrices["Te"]
            self.Ti[0] = matrices["Ti"]
            self.X[0] = matrices["X"]
            self.Y[0] = matrices["Y"]

    def weight_matrix(self, wee: np.array, wei: np.array, wie: np.array, i: int):
        """Update weight matrices

        Args:
            wee (array): Excitatory-Excitatory weight matrix

            wei (array): Inhibitory-Excitatory weight matrix

            wie (array): Excitatory-Inhibitory weight matrix

            i (int): Time step

        Returns:
            tuple (array): Weight Matrices Wee, Wei, Wie"""

        self.Wee[i + 1] = wee
        self.Wei[i + 1] = wei
        self.Wie[i + 1] = wie

        return self.Wee, self.Wei, self.Wie

    def threshold_matrix(self, te: np.array, ti: np.array, i: int):
        """Update threshold matrices

        Args:
            te (array): Excitatory threshold

            ti (array): Inhibitory threshold

            i (int): Time step
        Returns:
            tuple (array): Threshold Matrices Te and Ti"""

        self.Te[i + 1] = te
        self.Ti[i + 1] = ti
        return self.Te, self.Ti

    def network_activity_t(
        self, excitatory_net: np.array, inhibitory_net: np.array, i: int
    ):
        """Network state at current time step

        Args:
            excitatory_net (array): Excitatory network activity

            inhibitory_net (array): Inhibitory network activity

            i (int): Time step

        Returns:
            tuple (array): Updated Excitatory and Inhibitory states
        """

        self.X[i + 1] = excitatory_net
        self.Y[i + 1] = inhibitory_net

        return self.X, self.Y

    def network_activity_t_1(self, x: np.array, y: np.array, i: int):
        """Network activity at previous time step

        Args:
            x (array): Excitatory network activity

            y (array): Inhibitory network activity

            i (int): Time step

        Returns:
            tuple (array): Previous Excitatory and Inhibitory states
        """
        x_1, y_1 = [0] * self.time_steps, [0] * self.time_steps
        x_1[i] = x
        y_1[i] = y

        return x_1, y_1


class NetworkState(Plasticity):

    """The evolution of network states

    Args:
        v_t (array): External input/stimuli

    Returns:
        instance (object): NetworkState instance"""

    def __init__(self, v_t: np.array):
        super().__init__()
        self.v_t = v_t
        assert Sorn._nu == len(
            self.v_t
        ), "Input units and input size mismatch: {} != {}".format(
            Sorn._nu, len(self.v_t)
        )
        if Sorn._nu != Sorn._ne:
            self.v_t = list(self.v_t) + [0.0] * (Sorn._ne - Sorn._nu)
        self.v_t = np.expand_dims(self.v_t, 1)

    def incoming_drive(self, weights: np.array, activity_vector: np.array):
        """Excitatory Post synaptic potential towards neurons in the reservoir in the absence of external input

        Args:
            weights (array): Synaptic strengths

            activity_vector (list): Acitivity of inhibitory or Excitatory neurons

        Returns:
            incoming (array): Excitatory Post synaptic potential towards neurons
        """
        incoming = weights * activity_vector
        incoming = np.array(incoming.sum(axis=0))
        return incoming

    def excitatory_network_state(
        self,
        wee: np.array,
        wei: np.array,
        te: np.array,
        x: np.array,
        y: np.array,
        white_noise_e: np.array,
    ):
        """Activity of Excitatory neurons in the network

        Args:
            wee (array): Excitatory-Excitatory weight matrix

            wei (array): Inhibitory-Excitatory weight matrix

            te (array): Excitatory threshold

            x (array): Excitatory network activity

            y (array): Inhibitory network activity

            white_noise_e (array): Gaussian noise

        Returns:
            x (array): Current Excitatory network activity
        """
        xt = x[:, 1]
        xt = xt.reshape(self.ne, 1)
        yt = y[:, 1]
        yt = yt.reshape(self.ni, 1)

        incoming_drive_e = np.expand_dims(
            self.incoming_drive(weights=wee, activity_vector=xt), 1
        )
        incoming_drive_i = np.expand_dims(
            self.incoming_drive(weights=wei, activity_vector=yt), 1
        )
        tot_incoming_drive = (
            incoming_drive_e
            - incoming_drive_i
            + white_noise_e
            + np.asarray(self.v_t)
            - te
        )

        # Heaviside step function
        heaviside_step = np.expand_dims([0.0] * len(tot_incoming_drive), 1)
        heaviside_step[tot_incoming_drive > 0] = 1.0
        return heaviside_step

    def inhibitory_network_state(
        self, wie: np.array, ti: np.array, y: np.array, white_noise_i: np.array
    ):
        """Activity of Excitatory neurons in the network

        Args:
            wee (array): Excitatory-Excitatory weight matrix

            wie (array): Excitatory-Inhibitory weight matrix

            ti (array): Inhibitory threshold

            y (array): Inhibitory network activity

            white_noise_i (array): Gaussian noise

        Returns:
            y (array): Current Inhibitory network activity"""

        wie = np.asarray(wie)
        yt = y[:, 1]
        yt = yt.reshape(Sorn._ne, 1)

        incoming_drive_e = np.expand_dims(
            self.incoming_drive(weights=wie, activity_vector=yt), 1
        )

        tot_incoming_drive = incoming_drive_e + white_noise_i - ti
        heaviside_step = np.expand_dims([0.0] * len(tot_incoming_drive), 1)
        heaviside_step[tot_incoming_drive > 0] = 1.0

        return heaviside_step

    def recurrent_drive(
        self,
        wee: np.array,
        wei: np.array,
        te: np.array,
        x: np.array,
        y: np.array,
        white_noise_e: np.array,
    ):
        """Network state due to recurrent drive received by the each unit at time t+1. Activity of Excitatory neurons without external stimuli

        Args:
            wee (array): Excitatory-Excitatory weight matrix

            wei (array): Inhibitory-Excitatory weight matrix

            te (array): Excitatory threshold

            x (array): Excitatory network activity

            y (array): Inhibitory network activity

            white_noise_e (array): Gaussian noise

        Returns:
            xt (array): Recurrent network state
        """
        xt = x[:, 1]
        xt = xt.reshape(self.ne, 1)
        yt = y[:, 1]
        yt = yt.reshape(self.ni, 1)

        incoming_drive_e = np.expand_dims(
            self.incoming_drive(weights=wee, activity_vector=xt), 1
        )
        incoming_drive_i = np.expand_dims(
            self.incoming_drive(weights=wei, activity_vector=yt), 1
        )

        tot_incoming_drive = incoming_drive_e - incoming_drive_i + white_noise_e - te

        heaviside_step = np.expand_dims([0.0] * len(tot_incoming_drive), 1)
        heaviside_step[tot_incoming_drive > 0] = 1.0

        return heaviside_step


# Simulate / Train SORN
class Simulator_(Sorn):

    """Simulate SORN using external input/noise using the fresh or pretrained matrices

    Args:
        inputs (np.array, optional): External stimuli. Defaults to None.

        phase (str, optional): Plasticity phase. Defaults to "plasticity".

        matrices (dict, optional): Network states, connections and threshold matrices. Defaults to None.

        time_steps (int, optional): Total number of time steps to simulate the network. Defaults to 1.

        noise (bool, optional): If True, noise will be added. Defaults to True.

    Returns:
        plastic_matrices (dict): Network states, connections and threshold matrices

        X_all (array): Excitatory network activity collected during entire simulation steps

        Y_all (array): Inhibitory network activity collected during entire simulation steps

        R_all (array): Recurrent network activity collected during entire simulation steps

        frac_pos_active_conn (list): Number of positive connection strengths in the network at each time step during simulation"""

    def __init__(self):
        super().__init__()
        pass

    def simulate_sorn(
        self,
        inputs: np.array = None,
        phase: str = "plasticity",
        matrices: dict = None,
        time_steps: int = None,
        noise: bool = True,
        freeze: list = None,
        **kwargs
    ):
        """Simulation/Plasticity phase

        Args:
            inputs (np.array, optional): External stimuli. Defaults to None.

            phase (str, optional): Plasticity phase. Defaults to "plasticity"

            matrices (dict, optional): Network states, connections and threshold matrices. Defaults to None.

            time_steps (int, optional): Total number of time steps to simulate the network. Defaults to 1.

            noise (bool, optional): If True, noise will be added. Defaults to True.

            freeze (list, optional): List of synaptic plasticity mechanisms which will be turned off during simulation. Defaults to None.

        Returns:
            plastic_matrices (dict): Network states, connections and threshold matrices

            X_all (array): Excitatory network activity collected during entire simulation steps

            Y_all (array): Inhibitory network activity collected during entire simulation steps

            R_all (array): Recurrent network activity collected during entire simulation steps

            frac_pos_active_conn (list): Number of positive connection strengths in the network at each time step during simulation"""

        assert (
            phase == "plasticity" or "training"
        ), "Phase can be either 'plasticity' or 'training'"

        self.time_steps = time_steps
        Sorn._time_steps = time_steps
        self.phase = phase
        self.matrices = matrices
        self.freeze = freeze

        kwargs_ = [
            "_ne",
            "_nu",
            "_network_type_ee",
            "_network_type_ei",
            "_network_type_ie",
            "_lambda_ee",
            "_lambda_ei",
            "_lambda_ie",
            "_eta_stdp",
            "_eta_inhib",
            "_eta_ip",
            "_te_max",
            "_ti_max",
            "_ti_min",
            "_te_min",
            "_mu_ip",
            "_sigma_ip",
        ]
        for key, value in kwargs.items():
            if key in kwargs_:
                setattr(Sorn, key, value)
        # assert Sorn._nu == len(inputs[:,0]),"Size mismatch: Input != Nu "
        Sorn._ni = int(0.2 * Sorn._ne)

        # Initialize/Get the weight, threshold matrices and activity vectors
        matrix_collection = MatrixCollection(
            phase=self.phase, matrices=self.matrices)

        # Collect the network activity at all time steps

        X_all = [0] * self.time_steps
        Y_all = [0] * self.time_steps
        R_all = [0] * self.time_steps

        frac_pos_active_conn = []

        # To get the last activation status of Exc and Inh neurons
        for i in range(self.time_steps):

            if noise:
                white_noise_e = Initializer.white_gaussian_noise(
                    mu=0.0, sigma=0.04, t=Sorn._ne
                )
                white_noise_i = Initializer.white_gaussian_noise(
                    mu=0.0, sigma=0.04, t=Sorn._ni
                )
            else:
                white_noise_e, white_noise_i = 0.0, 0.0

            network_state = NetworkState(
                inputs[:, i]
            )

            # Buffers to get the resulting x and y vectors at the current time step and update the master matrix
            x_buffer, y_buffer = np.zeros(
                (Sorn._ne, 2)), np.zeros((Sorn._ni, 2))

            te_buffer, ti_buffer = np.zeros(
                (Sorn._ne, 1)), np.zeros((Sorn._ni, 1))

            # Get the matrices and rename them for ease of reading
            Wee, Wei, Wie = (
                matrix_collection.Wee,
                matrix_collection.Wei,
                matrix_collection.Wie,
            )
            Te, Ti = matrix_collection.Te, matrix_collection.Ti
            X, Y = matrix_collection.X, matrix_collection.Y

            # Fraction of active connections between E-E network
            frac_pos_active_conn.append((Wee[i] > 0.0).sum())

            # Recurrent drive
            r = network_state.recurrent_drive(
                Wee[i], Wei[i], Te[i], X[i], Y[i], white_noise_e
            )

            # Get excitatory states and inhibitory states given the weights and thresholds
            # x(t+1), y(t+1)
            excitatory_state_xt_buffer = network_state.excitatory_network_state(
                Wee[i], Wei[i], Te[i], X[i], Y[i], white_noise_e
            )
            inhibitory_state_yt_buffer = network_state.inhibitory_network_state(
                Wie[i], Ti[i], X[i], white_noise_i
            )

            # Update X and Y
            x_buffer[:, 0] = X[i][:, 1]  # xt -->(becomes) xt_1
            x_buffer[
                :, 1
            ] = excitatory_state_xt_buffer.T  # New_activation; x_buffer --> xt

            y_buffer[:, 0] = Y[i][:, 1]
            y_buffer[:, 1] = inhibitory_state_yt_buffer.T

            # Plasticity phase
            plasticity = Plasticity()

            # STDP
            if 'stdp' not in self.freeze:
                Wee_t = plasticity.stdp(
                    Wee[i], x_buffer, cutoff_weights=(0.0, 1.0))

            # Intrinsic plasticity
            if 'ip' not in self.freeze:
                Te_t = plasticity.ip(Te[i], x_buffer)

            # Structural plasticity
            if 'sp' not in freeze:
                Wee_t = plasticity.structural_plasticity(Wee_t)

            # iSTDP
            if 'istdp' not in freeze:
                Wei_t = plasticity.istdp(
                    Wei[i], x_buffer, y_buffer, cutoff_weights=(0.0, 1.0)
                )

            # Synaptic scaling Wee
            if 'ss' not in self.freeze:
                Wee_t = Plasticity().ss(Wee_t)
                Wei_t = Plasticity().ss(Wei_t)

            # Assign the matrices to the matrix collections
            matrix_collection.weight_matrix(Wee_t, Wei_t, Wie[i], i)
            matrix_collection.threshold_matrix(Te_t, Ti[i], i)
            matrix_collection.network_activity_t(x_buffer, y_buffer, i)

            X_all[i] = x_buffer[:, 1]
            Y_all[i] = y_buffer[:, 1]
            R_all[i] = r

        plastic_matrices = {
            "Wee": matrix_collection.Wee[-1],
            "Wei": matrix_collection.Wei[-1],
            "Wie": matrix_collection.Wie[-1],
            "Te": matrix_collection.Te[-1],
            "Ti": matrix_collection.Ti[-1],
            "X": X[-1],
            "Y": Y[-1],
        }

        return plastic_matrices, X_all, Y_all, R_all, frac_pos_active_conn


class Trainer_(Sorn):
    """Train the network with the fresh or pretrained network matrices and external stimuli

    Args:
        inputs (np.array, optional): External stimuli. Defaults to None.

        phase (str, optional): Training phase. Defaults to "training".

        matrices (dict, optional): Network states, connections and threshold matrices. Defaults to None.

        time_steps (int, optional): Total number of time steps to simulate the network. Defaults to 1.

        noise (bool, optional): If True, noise will be added. Defaults to True.

        freeze (list, optional): List of synaptic plasticity mechanisms which will be turned off during simulation. Defaults to None.

    Returns:
        plastic_matrices (dict): Network states, connections and threshold matrices

        X_all (array): Excitatory network activity collected during entire simulation steps

        Y_all (array): Inhibitory network activity collected during entire simulation steps

        R_all (array): Recurrent network activity collected during entire simulation steps

        frac_pos_active_conn (list): Number of positive connection strengths in the network at each time step during simulation"""

    def __init__(self):
        super().__init__()
        pass

    def train_sorn(
        self,
        inputs: np.array = None,
        phase: str = "training",
        matrices: dict = None,
        time_steps: int = None,
        noise: bool = True,
        freeze: list = None,
        **kwargs
    ):
        """Train the network with the fresh or pretrained network matrices and external stimuli

        Args:
            inputs (np.array, optional): External stimuli. Defaults to None.

            phase (str, optional): Training phase. Defaults to "training".

            matrices (dict, optional): Network states, connections and threshold matrices. Defaults to None.

            time_steps (int, optional): Total number of time steps to simulate the network. Defaults to 1.

            noise (bool, optional): If True, noise will be added. Defaults to True.

        Returns:
            plastic_matrices (dict): Network states, connections and threshold matrices

            X_all (array): Excitatory network activity collected during entire simulation steps

            Y_all (array): Inhibitory network activity collected during entire simulation steps

            R_all (array): Recurrent network activity collected during entire simulation steps

            frac_pos_active_conn (list): Number of positive connection strengths in the network at each time step during simulation"""

        assert (
            phase == "plasticity" or "training"
        ), "Phase can be either 'plasticity' or 'training'"

        kwargs_ = [
            "_ne",
            "_nu",
            "_network_type_ee",
            "_network_type_ei",
            "_network_type_ie",
            "_lambda_ee",
            "_lambda_ei",
            "_lambda_ie",
            "_eta_stdp",
            "_eta_inhib",
            "_eta_ip",
            "_te_max",
            "_ti_max",
            "_ti_min",
            "_te_min",
            "_mu_ip",
            "_sigma_ip",
        ]
        for key, value in kwargs.items():
            if key in kwargs_:
                setattr(Sorn, key, value)
        Sorn._ni = int(0.2 * Sorn._ne)
        # assert Sorn._nu == len(inputs[:,0]),"Size mismatch: Input != Nu "

        self.phase = phase
        self.matrices = matrices
        self.time_steps = time_steps
        Sorn._time_steps = time_steps
        self.inputs = np.asarray(inputs)
        self.freeze = freeze
        # Collect the network activity at all time steps
        X_all = [0] * self.time_steps
        Y_all = [0] * self.time_steps
        R_all = [0] * self.time_steps

        frac_pos_active_conn = []

        matrix_collection = MatrixCollection(
            phase=self.phase, matrices=self.matrices)

        for i in range(self.time_steps):

            if noise:
                white_noise_e = Initializer.white_gaussian_noise(
                    mu=0.0, sigma=0.04, t=Sorn._ne
                )
                white_noise_i = Initializer.white_gaussian_noise(
                    mu=0.0, sigma=0.04, t=Sorn._ni
                )
            else:
                white_noise_e = 0.0
                white_noise_i = 0.0

            network_state = NetworkState(
                self.inputs[:, i]
            )  # Feed Input as an argument to the class

            # Buffers to get the resulting x and y vectors at the current time step and update the master matrix
            x_buffer, y_buffer = np.zeros(
                (Sorn._ne, 2)), np.zeros((Sorn._ni, 2))
            te_buffer, ti_buffer = np.zeros(
                (Sorn._ne, 1)), np.zeros((Sorn._ni, 1))

            # Get the matrices and rename them for ease of reading
            Wee, Wei, Wie = (
                matrix_collection.Wee,
                matrix_collection.Wei,
                matrix_collection.Wie,
            )
            Te, Ti = matrix_collection.Te, matrix_collection.Ti
            X, Y = matrix_collection.X, matrix_collection.Y

            # Fraction of active connections between E-E network
            frac_pos_active_conn.append((Wee[i] > 0.0).sum())

            # Recurrent drive at t+1 used to predict the next external stimuli
            r = network_state.recurrent_drive(
                Wee[i], Wei[i], Te[i], X[i], Y[i], white_noise_e=white_noise_e
            )

            # Get excitatory states and inhibitory states given the weights and thresholds
            # x(t+1), y(t+1)
            excitatory_state_xt_buffer = network_state.excitatory_network_state(
                Wee[i], Wei[i], Te[i], X[i], Y[i], white_noise_e=white_noise_e
            )
            inhibitory_state_yt_buffer = network_state.inhibitory_network_state(
                Wie[i], Ti[i], X[i], white_noise_i=white_noise_i
            )

            # Update X and Y
            x_buffer[:, 0] = X[i][:, 1]  # xt -->xt_1
            x_buffer[:, 1] = excitatory_state_xt_buffer.T  # x_buffer --> xt
            y_buffer[:, 0] = Y[i][:, 1]
            y_buffer[:, 1] = inhibitory_state_yt_buffer.T

            if self.phase == "plasticity":
                plasticity = Plasticity()

                # STDP
                if 'stdp' not in self.freeze:
                    Wee_t = plasticity.stdp(
                        Wee[i], x_buffer, cutoff_weights=(0.0, 1.0))

                # Intrinsic plasticity
                if 'ip' not in self.freeze:
                    Te_t = plasticity.ip(Te[i], x_buffer)

                # Structural plasticity
                if 'sp' not in self.freeze:
                    Wee_t = plasticity.structural_plasticity(Wee_t)

                # iSTDP
                if 'istdp' not in self.freeze:
                    Wei_t = plasticity.istdp(
                        Wei[i], x_buffer, y_buffer, cutoff_weights=(0.0, 1.0)
                    )

                # Synaptic scaling Wee
                if 'sc' not in self.freeze:
                    Wee_t = Plasticity().ss(Wee_t)

                    # Synaptic scaling Wei
                    Wei_t = Plasticity().ss(Wei_t)
            else:
                Wee_t, Wei_t, Te_t = Wee[i], Wei[i], Te[i]

            # Assign the matrices to the matrix collections
            matrix_collection.weight_matrix(Wee_t, Wei_t, Wie[i], i)
            matrix_collection.threshold_matrix(Te_t, Ti[i], i)
            matrix_collection.network_activity_t(x_buffer, y_buffer, i)

            X_all[i] = x_buffer[:, 1]
            Y_all[i] = y_buffer[:, 1]
            R_all[i] = r

        plastic_matrices = {
            "Wee": matrix_collection.Wee[-1],
            "Wei": matrix_collection.Wei[-1],
            "Wie": matrix_collection.Wie[-1],
            "Te": matrix_collection.Te[-1],
            "Ti": matrix_collection.Ti[-1],
            "X": X[-1],
            "Y": Y[-1],
        }

        return plastic_matrices, X_all, Y_all, R_all, frac_pos_active_conn


Trainer = Trainer_()
Simulator = Simulator_()
if __name__ == "__main__":
    pass
