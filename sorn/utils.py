# MIT License

# Copyright (c) 2019 Saranraj Nambusubramaniyan

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import division
import numpy as np
from scipy.stats import norm
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
import networkx as nx
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


class Initializer(object):
    """
    Helper class to initialize the matrices for the SORN
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_strong_inp(length: int, reservoir_size: int):
        """Generate strong one-hot vector of input. Random neurons in the reservoir acts as inputs

        Args:
            length (int): Number of input neurons

        Returns:
            inp (array): Input vector of length equals the number of neurons in the reservoir
                  with randomly chosen neuron set active

            idx (list): List of chosen input neurons"""

        inp = [0] * reservoir_size
        x = [0] * length
        idx = np.random.choice(length, np.random.randint(reservoir_size))

        for i in idx:
            x[i] = 1.0e4

        inp[: len(x)] = x

        return inp, idx

    # Generate multi-node one-hot strong inputs

    @staticmethod
    def multi_one_hot_inp(ne: int, inputs: list, n_nodes_per_inp: int):
        """Generate multi(n_nodes_per_inp) one hot vector for each input.
        For each input, set n_nodes_per_inp equals one and the rest of
        neurons in the pool recieves no external stimuli

        Args:
          ne (int): Number of excitatory units in sorn

          inputs (list): input labels

          n_nodes_per_inp(int): Number of target units in pool that receives single input

        Returns:
          one_hot_vector for each label with length equals ne

        """

        one_hot = np.zeros((ne, len(inputs)))

        idxs = []

        for _ in range(n_nodes_per_inp):
            idxs.append(random.sample(range(0, ne), len(inputs)))

        idxs = list(zip(*idxs))

        j = 0  # Max(j) = len(inputs)
        for idx_list in idxs:
            for i in idx_list:
                one_hot[i][j] = 1
            j += 1

        return one_hot, idxs

    @staticmethod
    def generate_gaussian_inputs(length: int, reservoir_size: int):

        """Generate external stimuli sampled from Gaussian distribution.
        Randomly neurons in the reservoir receives this input at each timestep

        Args:
            length (int): Number of input neurons

        Returns:
            out (array): Input vector of length equals the number of neurons in the reservoir
                  with randomly chosen neuron set active

            idx (int): List of chosen input neurons
        """

        out = [0] * reservoir_size
        x = [0] * length
        idx = np.random.choice(length, np.random.randint(reservoir_size))
        inp = np.random.normal(length)

        for i in idx:
            x[i] = inp[i]

        out[: len(x)] = x

        return out, idx

    @staticmethod
    def normalize_weight_matrix(weight_matrix: np.array):

        # Applied only while initializing the weight. During simulation, Synaptic scaling applied on weight matrices

        """Normalize the weights in the matrix such that incoming connections to a neuron sum up to 1

        Args:
            weight_matrix (array): Incoming Weights from W_ee or W_ei or W_ie

        Returns:
            weight_matrix (array): Normalized weight matrix"""

        normalized_weight_matrix = weight_matrix / np.sum(weight_matrix, axis=0)

        return normalized_weight_matrix

    @staticmethod
    def generate_lambd_connections(
        synaptic_connection: str, ne: int, ni: int, lambd_w: int, lambd_std: int
    ):

        """Generate lambda incoming connections for Excitatory neurons and outgoing connections per Inhibitory neuron

        Args:
            synaptic_connection (str):  Type of sysnpatic connection (EE,EI or IE)

            ne (int): Number of excitatory units

            ni (int): Number of inhibitory units

            lambd_w (int): Average number of incoming connections

            lambd_std (int): Standard deviation of average number of connections per neuron

        Returns:
            connection_weights (array) - Weight matrix

        """

        if synaptic_connection == "EE":

            """Choose random lamda connections per neuron"""

            # Draw normally distributed ne integers with mean lambd_w

            lambdas_incoming = norm.ppf(
                np.random.random(ne), loc=lambd_w, scale=lambd_std
            ).astype(int)

            # lambdas_outgoing = norm.ppf(np.random.random(ne), loc=lambd_w, scale=lambd_std).astype(int)

            # List of neurons

            list_neurons = list(range(ne))

            # Connection weights

            connection_weights = np.zeros((ne, ne))

            # For each lambd value in the above list,
            # generate weights for incoming and outgoing connections

            # -------------Gaussian Distribution of weights --------------

            # weight_matrix = np.random.randn(Sorn.ne, Sorn.ni) + 2 # Small random values from gaussian distribution
            # Centered around 2 to make all values positive

            # ------------Uniform Distribution --------------------------
            global_incoming_weights = np.random.uniform(0.0, 0.1, sum(lambdas_incoming))

            # Index Counter
            global_incoming_weights_idx = 0

            # Choose the neurons in order [0 to 199]

            for neuron in list_neurons:

                # Choose ramdom unique (lambdas[neuron]) neurons from  list_neurons
                possible_connections = list_neurons.copy()

                possible_connections.remove(
                    neuron
                )  # Remove the selected neuron from possible connections i!=j

                # Choose random presynaptic neurons
                possible_incoming_connections = random.sample(
                    possible_connections, lambdas_incoming[neuron]
                )

                incoming_weights_neuron = global_incoming_weights[
                    global_incoming_weights_idx : global_incoming_weights_idx
                    + lambdas_incoming[neuron]
                ]

                # ---------- Update the connection weight matrix ------------

                # Update incoming connection weights for selected 'neuron'

                for incoming_idx, incoming_weight in enumerate(incoming_weights_neuron):
                    connection_weights[possible_incoming_connections[incoming_idx]][
                        neuron
                    ] = incoming_weight

                global_incoming_weights_idx += lambdas_incoming[neuron]

            return connection_weights

        if synaptic_connection == "EI":

            """Choose random lamda connections per neuron"""

            # Draw normally distributed ni integers with mean lambd_w
            lambdas = norm.ppf(
                np.random.random(ni), loc=lambd_w, scale=lambd_std
            ).astype(int)

            # List of neurons

            list_neurons = list(range(ni))  # Each i can connect with random ne neurons

            # Initializing connection weights variable

            connection_weights = np.zeros((ni, ne))

            # ------------Uniform Distribution -----------------------------
            global_outgoing_weights = np.random.uniform(0.0, 0.1, sum(lambdas))

            # Index Counter
            global_outgoing_weights_idx = 0

            # Choose the neurons in order [0 to 40]

            for neuron in list_neurons:

                # Choose random unique (lambdas[neuron]) neurons from  list_neurons
                possible_connections = list(range(ne))

                possible_outgoing_connections = random.sample(
                    possible_connections, lambdas[neuron]
                )  # possible_outgoing connections to the neuron

                # Update weights
                outgoing_weights = global_outgoing_weights[
                    global_outgoing_weights_idx : global_outgoing_weights_idx
                    + lambdas[neuron]
                ]

                # ---------- Update the connection weight matrix ------------

                # Update outgoing connections for the neuron

                for outgoing_idx, outgoing_weight in enumerate(
                    outgoing_weights
                ):  # Update the columns in the connection matrix
                    connection_weights[neuron][
                        possible_outgoing_connections[outgoing_idx]
                    ] = outgoing_weight

                # Update the global weight values index
                global_outgoing_weights_idx += lambdas[neuron]

            return connection_weights

    @staticmethod
    def get_incoming_connection_dict(weights: np.array):
        """Get the non-zero entries in columns is the incoming connections for the neurons

        Args:
            weights (np.array): Connection/Synaptic weights

        Returns:
            dict : Dictionary of incoming connections to each neuron
        """

        # Indices of nonzero entries in the columns
        connection_dict = dict.fromkeys(range(1, len(weights) + 1), 0)

        for i in range(len(weights[0])):  # For each neuron
            connection_dict[i] = list(np.nonzero(weights[:, i])[0])

        return connection_dict

    @staticmethod
    def get_outgoing_connection_dict(weights: np.array):
        """Get the non-zero entries in rows is the outgoing connections for the neurons

        Args:
            weights (np.array): Connection/Synaptic weights

        Returns:
            dict : Dictionary of outgoing connections from each neuron
        """

        # Indices of nonzero entries in the rows
        connection_dict = dict.fromkeys(range(1, len(weights) + 1), 1)

        for i in range(len(weights[0])):  # For each neuron
            connection_dict[i] = list(np.nonzero(weights[i, :])[0])

        return connection_dict

    @staticmethod
    def reset_min(z: np.array, cutoff_val: float):
        """Prune the connections/thresholds with negative connection strength. The weights less than cutoff_weight set to 0

        Args:
            z (np.array): Synaptic strengths or neuron threshold values

            cutoff_val (float): Lower threshold

        Returns:
            array: Connections weights or unit thresholds with values less than cutoff_val set to 0
        """

        z[z <= cutoff_val] = cutoff_val

        return z

    @staticmethod
    def reset_max(z: np.array, cutoff_val: float):
        """Set cutoff limit for the values in given array

        Args:
            z (np.array): Synaptic strengths or neuron threshold values

            cutoff_val (float): Higher threshold

        Returns:
            array: Connections weights or unit thresolds with values greater than cutoff_val set to 1
        """

        z[z > cutoff_val] = cutoff_val

        return z

    @staticmethod
    def inactive_synapses(wee: np.array):
        """Helper function for Structural plasticity to randomly select the unconnected units

        Args:
            wee (array):  Weight matrix

        Returns:
            list (indices): (row_idx,col_idx)"""

        i, j = np.where(wee <= 0.0)
        indices = list(zip(i, j))

        self_conn_removed = []
        for i, idxs in enumerate(indices):

            if idxs[0] != idxs[1]:
                self_conn_removed.append(indices[i])

        return self_conn_removed

    @staticmethod
    def white_gaussian_noise(mu: float, sigma: float, t: int):

        """Generates white gaussian noise with mean mu, standard deviation sigma and
        the noise length equals t

        Args:
            mu (float): Mean value of Gaussian noise

            sigma (float): Standard deviation of Gaussian noise

            t (int): Length of noise vector

        Returns:
            array: White gaussian noise of length t
        """

        noise = np.random.normal(mu, sigma, t)

        return np.expand_dims(noise, 1)

    @staticmethod
    def zero_sum_incoming_check(weights: np.array):
        """Make sure, each neuron in the pool has atleast 1 incoming connection

        Args:
            weights (array): Synaptic strengths

        Returns:
            array: Synaptic weights of neurons with atleast one positive (non-zero) incoming connection strength
        """
        zero_sum_incomings = np.where(np.sum(weights, axis=0) == 0.0)
        if len(zero_sum_incomings[-1]) == 0:
            return weights
        else:
            for zero_sum_incoming in zero_sum_incomings[-1]:

                rand_indices = np.random.randint(int(weights.shape[0] * 0.2), size=2)
                rand_values = np.random.uniform(0.0, 0.1, 2)

                for i, idx in enumerate(rand_indices):
                    weights[:, zero_sum_incoming][idx] = rand_values[i]

        return weights


class Plotter(object):
    """Wrapper class to call plotting methods"""

    def __init__(self):
        pass

    @staticmethod
    def hist_incoming_conn(
        weights: np.array, bin_size: int, histtype: str, savefig: bool
    ):
        """Plot the histogram of number of presynaptic connections per neuron

        Args:
            weights (array): Connection weights

            bin_size (int): Histogram bin size

            histtype (str): Same as histtype matplotlib

            savefig (bool): If True plot will be saved as png file in the cwd

        Returns:
            plot (matplotlib.pyplot): plot object
        """
        num_incoming_weights = np.sum(np.array(weights) > 0, axis=0)

        plt.figure(figsize=(12, 5))
        plt.xlabel("Number of connections")
        plt.ylabel("Probability")

        # Fit a normal distribution to the data
        mu, std = norm.fit(num_incoming_weights)
        plt.hist(
            num_incoming_weights, bins=bin_size, density=True, alpha=0.6, color="b"
        )

        # PDF
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, max(num_incoming_weights))
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, "k", linewidth=2)
        title = "Distribution of presynaptic connections: mu = %.2f,  std = %.2f" % (
            mu,
            std,
        )
        plt.title(title)

        if savefig:
            plt.savefig("hist_incoming_conn")

        return plt.show()

    @staticmethod
    def hist_outgoing_conn(
        weights: np.array, bin_size: int, histtype: str, savefig: bool
    ):
        """Plot the histogram of number of incoming connections per neuron

        Args:
            weights (array): Connection weights

            bin_size (int): Histogram bin size

            histtype (str): Same as histtype matplotlib

            savefig (bool): If True plot will be saved as png file in the cwd

        Returns:
            plot object"""

        # Plot the histogram of distribution of number of incoming connections in the network

        num_outgoing_weights = np.sum(np.array(weights) > 0, axis=1)

        plt.figure(figsize=(12, 5))
        plt.xlabel("Number of connections")
        plt.ylabel("Probability")

        # Fit a normal distribution to the data
        mu, std = norm.fit(num_outgoing_weights)
        plt.hist(
            num_outgoing_weights, bins=bin_size, density=True, alpha=0.6, color="b"
        )

        # PDF
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, max(num_outgoing_weights))
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, "k", linewidth=2)
        title = "Distribution of post synaptic connections: mu = %.2f,  std = %.2f" % (
            mu,
            std,
        )
        plt.title(title)

        if savefig:
            plt.savefig("hist_outgoing_conn")

        return plt.show()

    @staticmethod
    def network_connection_dynamics(connection_counts: np.array, savefig: bool):
        """Plot number of positive connection in the excitatory pool

        Args:
            connection_counts (array) - 1D Array of number of connections in the network per time step

            savefig (bool) - If True plot will be saved as png file in the cwd

        Returns:
            plot object
        """

        # Plot graph for entire simulation time period
        _, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(connection_counts, label="Connection dynamics")
        plt.margins(x=0)
        ax1.set_xticks(ax1.get_xticks()[::2])

        ax1.set_title("Network connection dynamics")
        plt.ylabel("Number of active connections")
        plt.xlabel("Time step")
        plt.legend(loc="upper right")
        plt.tight_layout()

        if savefig:
            plt.savefig("connection_dynamics")

        return plt.show()

    @staticmethod
    def hist_firing_rate_network(spike_train: np.array, bin_size: int, savefig: bool):

        """Plot the histogram of firing rate (total number of neurons spike at each time step)

        Args:
            spike_train (array): Array of spike trains

            bin_size (int): Histogram bin size

            savefig (bool): If True, plot will be saved in the cwd

        Returns:
            plot object"""

        fr = np.count_nonzero(spike_train.tolist(), 1)

        # Filter zero entries in firing rate list above
        fr = list(filter(lambda a: a != 0, fr))
        plt.title("Distribution of population activity without inactive time steps")
        plt.xlabel("Spikes/time step")
        plt.ylabel("Count")

        plt.hist(fr, bin_size)

        if savefig:
            plt.savefig("hist_firing_rate_network.png")

        return plt.show()

    @staticmethod
    def scatter_plot(spike_train: np.array, savefig: bool):

        """Scatter plot of spike trains

        Args:
            spike_train (list): Array of spike trains

            with_firing_rates (bool): If True, firing rate of the network will be plotted

            savefig (bool): If True, plot will be saved in the cwd

        Returns:
            plot object"""

        # Conver the list of spike train into array
        spike_train = np.asarray(spike_train)
        # Get the indices where spike_train is 1
        x, y = np.argwhere(spike_train.T == 1).T

        plt.figure(figsize=(8, 5))

        firing_rates = Statistics.firing_rate_network(spike_train).tolist()
        plt.plot(firing_rates, label="Firing rate")
        plt.legend(loc="upper left")

        plt.scatter(y, x, s=0.1, color="black")
        plt.title("Spike Trains")
        plt.xlabel("Time step")
        plt.ylabel("Neuron")
        plt.legend(loc="upper left")

        if savefig:
            plt.savefig("ScatterSpikeTrain.png")
        return plt.show()

    @staticmethod
    def raster_plot(spike_train: np.array, savefig: bool):

        """Raster plot of spike trains

        Args:
            spike_train (array): Array of spike trains

            with_firing_rates (bool): If True, firing rate of the network will be plotted

            savefig (bool): If True, plot will be saved in the cwd

        Returns:
            plot object"""

        # Conver the list of spike train into array
        spike_train = np.asarray(spike_train)

        plt.figure(figsize=(11, 6))

        firing_rates = Statistics.firing_rate_network(spike_train).tolist()
        plt.plot(firing_rates, label="Firing rate")
        plt.legend(loc="upper left")
        plt.title("Spike Trains")
        # Get the indices where spike_train is 1
        x, y = np.argwhere(spike_train.T == 1).T

        plt.plot(y, x, "|r")
        plt.xlabel("Time step")
        plt.ylabel("Neuron")

        if savefig:
            plt.savefig("RasterSpikeTrain.png")
        return plt.show()

    @staticmethod
    def correlation(corr: np.array, savefig: bool):

        """Plot correlation between neurons

        Args:
            corr (array): Correlation matrix

            savefig (bool): If true will save the plot at the current working directory

        Returns:
            matplotlib.pyplot: Neuron Correlation plot
        """

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(11, 9))

        # Custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            xticklabels=5,
            yticklabels=5,
            vmax=0.1,
            center=0,
            square=False,
            linewidths=0.0,
            cbar_kws={"shrink": 0.9},
        )

        if savefig:
            plt.savefig("Correlation between neurons")
        return None

    @staticmethod
    def isi_exponential_fit(
        spike_train: np.array, neuron: int, bin_size: int, savefig: bool
    ):

        """Plot Exponential fit on the inter-spike intervals during training or simulation phase

        Args:
            spike_train (array): Array of spike trains

            neuron (int): Target neuron

            bin_size (int): Spike train will be splitted into bins of size bin_size

            savefig (bool): If True, plot will be saved in the cwd

        Returns:
            plot object"""

        isi = Statistics.spike_time_intervals(spike_train[:, neuron])

        y, x = np.histogram(sorted(isi), bins=bin_size)

        x = [int(i) for i in x]
        y = [float(i) for i in y]

        def exponential_func(y, a, b, c):
            return a * np.exp(-b * np.array(y)) - c

        # Curve fit
        popt, _ = curve_fit(exponential_func, x[1:bin_size], y[1:bin_size])

        plt.plot(
            x[1:bin_size],
            exponential_func(x[1:bin_size], *popt),
            label="Exponential fit",
        )
        plt.title("Distribution of Inter Spike Intervals and Exponential Curve Fit")
        plt.scatter(x[1:bin_size], y[1:bin_size], s=2.0, color="black", label="ISI")
        plt.xlabel("ISI")
        plt.ylabel("Frequency")
        plt.legend()

        if savefig:
            plt.savefig("isi_exponential_fit")
        return plt.show()

    @staticmethod
    def weight_distribution(weights: np.array, bin_size: int, savefig: bool):

        """Plot the distribution of synaptic weights

        Args:
            weights (array): Connection weights

            bin_size (int): Spike train will be splited into bins of size bin_size

            savefig (bool): If True, plot will be saved in the cwd

        Returns:
            plot object"""

        weights = weights[
            weights >= 0.01
        ]  # Remove the weight values less than 0.01 # As reported in article SORN 2013
        y, x = np.histogram(weights, bins=bin_size)  # Create histogram with bin_size
        plt.title("Synaptic weight distribution")
        plt.scatter(x[:-1], y, s=2.0, c="black")
        plt.xlabel("Connection strength")
        plt.ylabel("Frequency")

        if savefig:
            plt.savefig("weight_distribution")

        return plt.show()

    @staticmethod
    def linear_lognormal_fit(weights: np.array, num_points: int, savefig: bool):

        """Lognormal curve fit on connection weight distribution

        Args:
            weights (array): Connection weights

            num_points(int): Number of points to be plotted in the x axis

            savefig(bool): If True, plot will be saved in the cwd

        Returns:
            plot object"""

        weights = np.array(weights.tolist())
        weights = weights[weights >= 0.01]

        M = float(np.mean(weights))  # Geometric mean
        s = float(np.std(weights))  # Geometric standard deviation

        # Lognormal distribution parameters

        mu = float(np.mean(np.log(weights)))  # Mean of log(X)
        sigma = float(np.std(np.log(weights)))  # Standard deviation of log(X)
        shape = sigma  # Scipy's shape parameter
        scale = np.exp(mu)  # Scipy's scale parameter
        median = np.exp(mu)

        mode = np.exp(mu - sigma ** 2)  # Note that mode depends on both M and s
        mean = np.exp(mu + (sigma ** 2 / 2))  # Note that mean depends on both M and s
        x = np.linspace(np.min(weights), np.max(weights), num=num_points)

        pdf = stats.lognorm.pdf(x, shape, loc=0, scale=scale)

        plt.figure(figsize=(12, 4.5))
        plt.title("Curve fit on connection weight distribution")
        # Figure on linear scale
        plt.subplot(121)
        plt.plot(x, pdf)

        plt.vlines(mode, 0, pdf.max(), linestyle=":", label="Mode")
        plt.vlines(
            mean,
            0,
            stats.lognorm.pdf(mean, shape, loc=0, scale=scale),
            linestyle="--",
            color="green",
            label="Mean",
        )
        plt.vlines(
            median,
            0,
            stats.lognorm.pdf(median, shape, loc=0, scale=scale),
            color="blue",
            label="Median",
        )
        plt.ylim(ymin=0)
        plt.xlabel("Weight")
        plt.title("Linear scale")
        plt.legend()

        # Figure on logarithmic scale
        plt.subplot(122)
        plt.semilogx(x, pdf)

        plt.vlines(mode, 0, pdf.max(), linestyle=":", label="Mode")
        plt.vlines(
            mean,
            0,
            stats.lognorm.pdf(mean, shape, loc=0, scale=scale),
            linestyle="--",
            color="green",
            label="Mean",
        )
        plt.vlines(
            median,
            0,
            stats.lognorm.pdf(median, shape, loc=0, scale=scale),
            color="blue",
            label="Median",
        )
        plt.ylim(ymin=0)
        plt.xlabel("Weight")
        plt.title("Logarithmic scale")
        plt.legend()

        if savefig:
            plt.savefig("LinearLognormalFit")

        return plt.show()

    @staticmethod
    def plot_network(corr: np.array, corr_thres: float, fig_name: str = None):

        """Network x graphical visualization of the network using the correlation matrix

        Args:
            corr (array): Correlation between neurons

            corr_thres (array): Threshold to prune the connection. Smaller the threshold,
                                higher the density of connections

            fig_name (array, optional): Name of the figure. Defaults to None.

        Returns:
            matplotlib.pyplot: Plot instance
        """

        df = pd.DataFrame(corr)

        links = df.stack().reset_index()
        links.columns = ["var1", "var2", "value"]
        links_filtered = links.loc[
            (links["value"] > corr_thres) & (links["var1"] != links["var2"])
        ]

        G = nx.from_pandas_edgelist(links_filtered, "var1", "var2")

        plt.figure(figsize=(50, 50))
        nx.draw(
            G,
            with_labels=True,
            node_color="orange",
            node_size=50,
            linewidths=5,
            font_size=10,
        )
        plt.text(0.1, 0.9, "%s" % corr_thres)
        plt.savefig("%s" % fig_name)
        plt.show()

    @staticmethod
    def hamming_distance(hamming_dist: list, savefig: bool):
        """Hamming distance between true netorks states and perturbed network states

        Args:
            hamming_dist (list): Hamming distance values

            savefig (bool): If True, save the fig at current working directory

        Returns:
            matplotlib.pyplot: Hamming distance between true and perturbed network states
        """

        plt.figure(figsize=(15, 6))
        plt.title("Hamming distance between actual and perturbed states")
        plt.xlabel("Time steps")
        plt.ylabel("Hamming distance")
        plt.plot(hamming_dist)

        if savefig:
            plt.savefig("HammingDistance")

        return plt.show()


class Statistics(object):
    """Wrapper class for statistical analysis methods"""

    def __init__(self):
        pass

    @staticmethod
    def firing_rate_neuron(spike_train: np.array, neuron: int, bin_size: int):

        """Measure spike rate of given neuron during given time window

        Args:
            spike_train (array): Array of spike trains

            neuron (int): Target neuron in the reservoir

            bin_size (int): Divide the spike trains into bins of size bin_size

        Returns:
            int: firing_rate"""

        time_period = len(spike_train[:, 0])

        neuron_spike_train = spike_train[:, neuron]

        # Split the list(neuron_spike_train) into sub lists of length time_step
        samples_spike_train = [
            neuron_spike_train[i : i + bin_size]
            for i in range(0, len(neuron_spike_train), bin_size)
        ]

        spike_rate = 0.0

        for _, spike_train in enumerate(samples_spike_train):
            spike_rate += list(spike_train).count(1.0)

        spike_rate = spike_rate * bin_size / time_period

        return time_period, bin_size, spike_rate

    @staticmethod
    def firing_rate_network(spike_train: np.array):

        """Calculate number of neurons spikes at each time step.Firing rate of the network

        Args:
            spike_train (array): Array of spike trains

        Returns:
            int: firing_rate"""

        firing_rate = np.count_nonzero(spike_train.tolist(), 1)

        return firing_rate

    @staticmethod
    def scale_dependent_smoothness_measure(firing_rates: list):

        """Smoothem the firing rate depend on its scale. Smaller values corresponds to smoother series

        Args:
            firing_rates (list): List of number of active neurons per time step

        Returns:
            sd_diff (list): Float value signifies the smoothness of the semantic changes in firing rates
        """

        diff = np.diff(firing_rates)
        sd_diff = np.std(diff)

        return sd_diff

    @staticmethod
    def scale_independent_smoothness_measure(firing_rates: list):

        """Smoothem the firing rate independent of its scale. Smaller values corresponds to smoother series

        Args:
            firing_rates (list): List of number of active neurons per time step

        Returns:
            coeff_var (list):Float value signifies the smoothness of the semantic changes in firing rates"""

        diff = np.diff(firing_rates)
        mean_diff = np.mean(diff)
        sd_diff = np.std(diff)

        coeff_var = sd_diff / abs(mean_diff)

        return coeff_var

    @staticmethod
    def autocorr(firing_rates: list, t: int = 2):
        """
        Score interpretation
        - scores near 1 imply a smoothly varying series
        - scores near 0 imply that there's no overall linear relationship between a data point and the following one (that is, plot(x[-length(x)],x[-1]) won't give a scatter plot with any apparent linearity)

        - scores near -1 suggest that the series is jagged in a particular way: if one point is above the mean, the next is likely to be below the mean by about the same amount, and vice versa.

        Args:
            firing_rates (list): Firing rates of the network

            t (int, optional): Window size. Defaults to 2.

        Returns:
            array: Autocorrelation between neurons given their firing rates
        """

        return np.corrcoef(
            np.array(
                [
                    firing_rates[0 : len(firing_rates) - t],
                    firing_rates[t : len(firing_rates)],
                ]
            )
        )

    @staticmethod
    def avg_corr_coeff(spike_train: np.array):

        """Measure Average Pearson correlation coeffecient between neurons

        Args:
            spike_train (array): Neural activity

        Returns:
            array: Average correlation coeffecient"""

        corr_mat = np.corrcoef(np.asarray(spike_train).T)
        avg_corr = np.sum(corr_mat, axis=1) / 200
        corr_coeff = (
            avg_corr.sum() / 200
        )  # 2D to 1D and either upper  or lower half of correlation matrix.

        return corr_mat, corr_coeff

    @staticmethod
    def spike_times(spike_train: np.array):

        """Get the time instants at which neuron spikes

        Args:
            spike_train (array): Spike trains of neurons

        Returns:
            (array): Spike time of each neurons in the pool"""

        times = np.where(spike_train == 1.0)
        return times

    @staticmethod
    def spike_time_intervals(spike_train):

        """Generate spike time intervals spike_trains

        Args:
            spike_train (array): Network activity

        Returns:
            list: Inter spike intervals for each neuron in the reservoir
        """

        spike_times = Statistics.spike_times(spike_train)
        isi = np.diff(spike_times[-1])
        return isi

    @staticmethod
    def hamming_distance(actual_spike_train: np.array, perturbed_spike_train: np.array):
        """Hamming distance between true netorks states and perturbed network states

        Args:
            actual_spike_train (np.array): True network's states

            perturbed_spike_train (np.array): Perturbated network's states

        Returns:
            float: Hamming distance between true and perturbed network states
        """
        hd = [
            np.count_nonzero(actual_spike_train[i] != perturbed_spike_train[i])
            for i in range(len(actual_spike_train))
        ]
        return hd

    @staticmethod
    def fanofactor(spike_train: np.array, neuron: int, window_size: int):

        """Investigate whether neuronal spike generation is a poisson process

        Args:
            spike_train (np.array): Spike train of neurons in the reservoir

            neuron (int): Target neuron in the pool

            window_size (int): Sliding window size for time step ranges to be considered for measuring the fanofactor

        Returns:
            float : Fano factor of the neuron spike train
        """

        # Choose activity of random neuron
        neuron_act = spike_train[:, neuron]

        # Divide total observations into 'tws' time windows of size 'ws' for a neuron 60

        tws = np.split(neuron_act, window_size)
        fr = []
        for i in range(len(tws)):
            fr.append(np.count_nonzero(tws[i]))

        # print('Firing rate of the neuron during each time window of size %s is %s' %(ws,fr))

        mean_firing_rate = np.mean(fr)
        variance_firing_rate = np.var(fr)

        fano_factor = variance_firing_rate / mean_firing_rate

        return mean_firing_rate, variance_firing_rate, fano_factor

    @staticmethod
    def spike_source_entropy(spike_train: np.array, num_neurons: int):

        """Measure the uncertainty about the origin of spike from the network using entropy

        Args:
            spike_train (np.array): Spike train of neurons

            num_neurons (int): Number of neurons in the reservoir

        Returns:
            int : Spike source entropy of the network
        """
        # Number of spikes from each neuron during the interval
        n_spikes = np.count_nonzero(spike_train, axis=0)
        p = n_spikes / np.count_nonzero(
            spike_train
        )  # Probability of each neuron that can generate spike in next step
        # print(p)  # Note: pi shouldn't be zero
        sse = np.sum([pi * np.log(pi) for pi in p]) / np.log(
            1 / num_neurons
        )  # Spike source entropy

        return sse
