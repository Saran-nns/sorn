# -*- coding: utf-8 -*-

"""### IMPORT REQUIRED LIBRARIES"""

from __future__ import division

import numpy as np
from scipy.stats import norm
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy import stats

from matplotlib import pylab
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# Random seeds

# random.seed(110)
# np.random.seed(1101)

""" UTILS"""


class Initializer(object):

    def __init__(self):
        pass

    # INPUT GENERATOR
    # Generate strong one-hot vector of input

    @staticmethod
    def generate_strong_inp(length, reservoir_size):

        # Random neurons in the reservoir acts as inputs

        """
        Args:
            length - Number of input neurons
        Returns:
            out - Input vector of length equals the number of neurons in the reservoir
                  with randomly chosen neuron set active
            idx - List of chosen input neurons """

        inp = [0] * reservoir_size
        x = [0] * length
        idx = np.random.choice(length, np.random.randint(reservoir_size))

        for i in idx:
            x[i] = 1.0e4

        inp[:len(x)] = x

        return inp, idx

    # Generate multi-node one-hot strong inputs

    @staticmethod
    def multi_one_hot_inp(ne, inputs, n_nodes_per_inp):
        """Args:

          ne - Number of excitatory units in sorn
          inputs - input labels
          n_nodes_per_inp - Number of target units in pool that receives single input

        Returns:

          one_hot_vector for each label with length equals ne"""

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

    # one_hot_inp_identity, input_neurons = multi_one_hot_inp(200, inputs, 1)
    # """Edit: ROWS Equals number of neurons, hence each input has to be transposed"""
    #
    #
    # # print('Shape of one hot inputs',list(one_hot_inp_identity[:,1]),input_neurons)
    #
    # # # np.shape(list(one_hot_inp_identity[:,1]))
    # # c = np.expand_dims(np.asarray(one_hot_inp_identity[:,1]),1)
    # # c.shape

    # NOTE: Gaussian input is passed directly inside the class RunSORN:
    # TODO: generate_gaussian_inputs will be removed from RunSORN in future versions

    @staticmethod
    def generate_gaussian_inputs(length, reservoir_size):

        # Randomly neurons in the reservoir acts as inputs

        """
        Args:
            length - Number of input neurons
        Returns:
            out - Input vector of length equals the number of neurons in the reservoir
                  with randomly chosen neuron set active
            idx - List of chosen input neurons """

        out = [0] * reservoir_size
        x = [0] * length
        idx = np.random.choice(length, np.random.randint(reservoir_size))
        inp = np.random.normal(length)

        for i in idx:
            x[i] = inp[i]

        out[:len(x)] = x

        return out, idx

    @staticmethod
    def normalize_weight_matrix(weight_matrix):

        # Applied only while initializing the weight. During simulation, Synaptic scaling applied on weight matrices

        """ Normalize the weights in the matrix such that incoming connections to a neuron sum up to 1

        Args:
            weight_matrix(array) -- Incoming Weights from W_ee or W_ei or W_ie

        Returns:
            weight_matrix(array) -- Normalized weight matrix"""

        normalized_weight_matrix = weight_matrix / np.sum(weight_matrix, axis=0)

        return normalized_weight_matrix

    """Connection Generator:
     lambda incoming connections for Excitatory neurons and outgoing connections per Inhibitory neuron"""

    @staticmethod
    def generate_lambd_connections(synaptic_connection, ne, ni, lambd_w, lambd_std):

        """
        Args:
        synaptic_connection -  Type of sysnpatic connection (EE,EI or IE)
        ne - Number of excitatory units
        ni - Number of inhibitory units
        lambd_w - Average number of incoming connections
        lambd_std - Standard deviation of average number of connections per neuron

        Returns:

        connection_weights - Weight matrix

        """

        if synaptic_connection == 'EE':

            """Choose random lamda connections per neuron"""

            # Draw normally distributed ne integers with mean lambd_w

            lambdas_incoming = norm.ppf(np.random.random(ne), loc=lambd_w, scale=lambd_std).astype(int)

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

                possible_connections.remove(neuron)  # Remove the selected neuron from possible connections i!=j

                # Choose random presynaptic neurons
                possible_incoming_connections = random.sample(possible_connections, lambdas_incoming[neuron])

                incoming_weights_neuron = global_incoming_weights[
                                          global_incoming_weights_idx:global_incoming_weights_idx + lambdas_incoming[
                                              neuron]]

                # ---------- Update the connection weight matrix ------------

                # Update incoming connection weights for selected 'neuron'

                for incoming_idx, incoming_weight in enumerate(incoming_weights_neuron):
                    connection_weights[possible_incoming_connections[incoming_idx]][neuron] = incoming_weight

                global_incoming_weights_idx += lambdas_incoming[neuron]

            return connection_weights

        if synaptic_connection == 'EI':

            """Choose random lamda connections per neuron"""

            # Draw normally distributed ni integers with mean lambd_w
            lambdas = norm.ppf(np.random.random(ni), loc=lambd_w, scale=lambd_std).astype(int)

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

                possible_outgoing_connections = random.sample(possible_connections, lambdas[
                    neuron])  # possible_outgoing connections to the neuron

                # Update weights
                outgoing_weights = global_outgoing_weights[
                                   global_outgoing_weights_idx:global_outgoing_weights_idx + lambdas[neuron]]

                # ---------- Update the connection weight matrix ------------

                # Update outgoing connections for the neuron

                for outgoing_idx, outgoing_weight in enumerate(
                        outgoing_weights):  # Update the columns in the connection matrix
                    connection_weights[neuron][possible_outgoing_connections[outgoing_idx]] = outgoing_weight

                # Update the global weight values index
                global_outgoing_weights_idx += lambdas[neuron]

            return connection_weights

    """ More Util functions"""

    @staticmethod
    def get_incoming_connection_dict(weights):

        """ Get the non-zero entries in columns is the incoming connections for the neurons"""

        # Indices of nonzero entries in the columns
        connection_dict = dict.fromkeys(range(1, len(weights) + 1), 0)

        for i in range(len(weights[0])):  # For each neuron
            connection_dict[i] = list(np.nonzero(weights[:, i])[0])

        return connection_dict

    @staticmethod
    def get_outgoing_connection_dict(weights):

        """Get the non-zero entries in rows is the outgoing connections for the neurons"""

        # Indices of nonzero entries in the rows
        connection_dict = dict.fromkeys(range(1, len(weights) + 1), 1)

        for i in range(len(weights[0])):  # For each neuron
            connection_dict[i] = list(np.nonzero(weights[i, :])[0])

        return connection_dict

    @staticmethod
    def prune_small_weights(weights, cutoff_weight):

        """ Prune the connections with negative connection strength"""

        weights[weights <= cutoff_weight] = cutoff_weight

        return weights

    @staticmethod
    def set_max_cutoff_weight(weights, cutoff_weight):
        """ Set cutoff limit for the values in given array"""

        weights[weights > cutoff_weight] = cutoff_weight

        return weights

    @staticmethod
    def get_unconnected_indexes(wee):
        """
        Helper function for Structural plasticity to randomly select the unconnected units

        Args:
        wee -  Weight matrix

        Returns:
        list (indices) // indices = (row_idx,col_idx)"""

        i, j = np.where(wee <= 0.)
        indices = list(zip(i, j))

        self_conn_removed = []
        for i, idxs in enumerate(indices):

            if idxs[0] != idxs[1]:
                self_conn_removed.append(indices[i])

        return self_conn_removed

    @staticmethod
    def white_gaussian_noise(mu, sigma, t):
        """Generates white gaussian noise with mean mu, standard deviation sigma and
        the noise length equals t """

        noise = np.random.normal(mu, sigma, t)

        return np.expand_dims(noise, 1)

    # SANITY CHECK EACH WEIGHTS
    # Note this function has no influence in weight matrix, will be deprecated in next version

    @staticmethod
    def zero_sum_incoming_check(weights):
        zero_sum_incomings = np.where(np.sum(weights, axis=0) == 0.)

        if len(zero_sum_incomings[-1]) == 0:
            return weights
        else:
            for zero_sum_incoming in zero_sum_incomings[-1]:

                rand_indices = np.random.randint(40, size=2)  # 40 in sense that size of E = 200
                # given the probability of connections 0.2
                rand_values = np.random.uniform(0.0, 0.1, 2)

                for i, idx in enumerate(rand_indices):
                    weights[:, zero_sum_incoming][idx] = rand_values[i]

        return weights


# ANALYSIS PLOT HELPER CLASS

class Plotter(object):

    def __init__(self):
        pass

    @staticmethod
    def hist_incoming_conn(weights, bin_size, histtype, savefig):

        """Args:
        :param weights(array) - Connection weights
        :param bin_size(int) - Histogram bin size
        :param histtype(str) - Same as histtype matplotlib
        :param savefig(bool) - If True plot will be saved as png file in the cwd

        Returns:
        plot object """

        # Plot the histogram of distribution of number of incoming connections in the network

        num_incoming_weights = np.sum(np.array(weights) > 0, axis=0)

        plt.figure(figsize=(12, 5))

        plt.title('Number of incoming connections')
        plt.xlabel('Number of connections')
        plt.ylabel('Count')
        plt.hist(num_incoming_weights, bins=bin_size, histtype=histtype)

        # Empirical average and variance are computed
        avg = np.mean(num_incoming_weights)
        var = np.var(num_incoming_weights)
        # From hist plot above, it is clear that connection count follow gaussian distribution
        pdf_x = np.linspace(np.min(num_incoming_weights), np.max(num_incoming_weights), 100)
        pdf_y = 1.0 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (pdf_x - avg) ** 2 / var)

        plt.plot(pdf_x, pdf_y, 'k--', label='Gaussian fit')
        plt.axvline(x=avg, color='r', linestyle='--', label='Mean')
        plt.legend()

        if savefig:
            plt.savefig('hist_incoming_conn')

        return plt.show()

    @staticmethod
    def hist_outgoing_conn(weights, bin_size, histtype, savefig):

        """Args:
                :param weights(array) - Connection weights
                :param bin_size(int) - Histogram bin size
                :param histtype(str) - Same as histtype matplotlib
                :param savefig(bool) - If True plot will be saved as png file in the cwd

                Returns:
                plot object """

        # Plot the histogram of distribution of number of incoming connections in the network

        num_outgoing_weights = np.sum(np.array(weights) > 0, axis=1)

        plt.figure(figsize=(12, 5))

        plt.hist(num_outgoing_weights, bins=bin_size, histtype=histtype)
        plt.title('Number of Outgoing connections')
        plt.xlabel('Number of connections')
        plt.ylabel('Count')
        # Empirical average and variance are computed
        avg = np.mean(num_outgoing_weights)
        var = np.var(num_outgoing_weights)
        # From hist plot above, it is clear that connection count follow gaussian distribution
        pdf_x = np.linspace(np.min(num_outgoing_weights), np.max(num_outgoing_weights), 100)
        pdf_y = 1.0 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (pdf_x - avg) ** 2 / var)

        plt.plot(pdf_x, pdf_y, 'k--', label='Gaussian fit')
        plt.axvline(x=avg, color='r', linestyle='--', label='Mean')
        plt.legend()

        if savefig:
            plt.savefig('hist_outgoing_conn')

        return plt.show()

    @staticmethod
    def network_connection_dynamics(connection_counts, initial_steps, final_steps,savefig):

        """Args:
        :param connection_counts(array) - 1D Array of number of connections in the network per time step
        :param initial_steps(int) - Plot for initial steps
        :param final_steps(int) - Plot for final steps
        :param savefig(bool) - If True plot will be saved as png file in the cwd
        Returns:
        plot object"""

        # Plot graph for entire simulation time period
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(connection_counts, label='Connection dynamics')
        plt.margins(x=0)
        ax1.set_xticks(ax1.get_xticks()[::2])

        ax1.set_title("Network connection dynamics")
        plt.ylabel('Number of active connections')
        plt.xlabel('Time step')
        plt.legend(loc='upper right')
        plt.tight_layout()

        # Inset plot for initial simulation steps

        ax2 = plt.axes([0, 0, 1, 1])

        # Set the position and relative size of the inset axes within ax1
        ip = InsetPosition(ax1, [0.25, 0.4, 0.3, 0.3])
        ax2.set_axes_locator(ip)
        ax2.plot(connection_counts[0:initial_steps])
        plt.margins(x=0)
        ax2.set_title('Initial %s time steps of Decay Phase'%initial_steps)
        ax2.set_xticks(ax2.get_xticks()[::2])

        # End Inset plot
        ax3 = plt.axes([0, 0, 0, 0])

        # Set the position and relative size of the inset axes within ax1
        ip1 = InsetPosition(ax1, [0.6, 0.4, 0.3, 0.3])
        ax3.set_axes_locator(ip1)
        # Plot the last 10000 time steps
        ax3.plot(connection_counts[-final_steps:])
        plt.margins(x=0)
        ax3.set_title('Final %s time steps of Stable Phase'%final_steps)
        ax3.set_xticks(ax3.get_xticks()[::1])

        # Uncomment to show decay and stable phase in colors
        # ax1.axvspan(0, 200000, alpha=0.1, color='red')
        # ax2.axvspan(0, 10000, alpha=0.1, color='red')
        # ax1.axvspan(200000, 1000000, alpha=0.1, color='green')

        if savefig:
            plt.savefig('connection_dynamics')

        return plt.show()

    @staticmethod
    def hist_firing_rate_network(spike_train, bin_size, savefig):

        """Args:
        :param spike_train(array) - Array of spike trains
        :param bin_size(int) - Histogram bin size
        :param savefig(bool) - If True, plot will be saved in the cwd

        Returns: plot object """

        fr = np.count_nonzero(spike_train.tolist(), 1)

        # Filter zero entries in firing rate list above
        fr = list(filter(lambda a: a != 0, fr))
        plt.title('Distribution of population activity without inactive time steps')
        plt.xlabel('Spikes/time step')
        plt.ylabel('Count')

        plt.hist(fr, bin_size)

        if savefig:
            plt.savefig('hist_firing_rate_network.png')

        return plt.show()

    @staticmethod
    def scatter_plot(spike_train, savefig):

        """Args:
            :param spike_train (list) - Array of spike trains
            :param with_firing_rates(bool) - If True, firing rate of the network will be plotted
            :param savefig(bool) - If True, plot will be saved in the cwd

           Returns:
            plot object"""

        # Conver the list of spike train into array
        spike_train = np.asarray(spike_train)
        # Get the indices where spike_train is 1
        x, y = np.argwhere(spike_train.T == 1).T

        plt.figure(figsize=(8, 5))

        firing_rates = Statistics.firing_rate_network(spike_train).tolist()
        plt.plot(firing_rates, label='Firing rate')
        plt.legend(loc='upper left')

        plt.scatter(y, x, s=0.1, color='black')
        # plt.plot(y,x,'|b')
        # plt.gca().invert_yaxis()

        plt.xlabel('Time(ms)')
        plt.ylabel('Neuron #')
        plt.legend(loc='upper left')

        if savefig:
            plt.savefig('ScatterSpikeTrain.png')
        return plt.show()

    @staticmethod
    def raster_plot(spike_train, savefig):

        """Args:
                :param spike_train (array) - Array of spike trains
                :param with_firing_rates(bool) - If True, firing rate of the network will be plotted
                :param savefig(bool) - If True, plot will be saved in the cwd

            Returns:
                plot object"""

        # Conver the list of spike train into array
        spike_train = np.asarray(spike_train)

        plt.figure(figsize=(11, 6))


        firing_rates = Statistics.firing_rate_network(spike_train).tolist()
        plt.plot(firing_rates, label='Firing rate')
        plt.legend(loc='upper left')

        # Get the indices where spike_train is 1
        x, y = np.argwhere(spike_train.T == 1).T

        plt.plot(y, x, '|r')

        # plt.gca().invert_yaxis()
        plt.xlabel('Time(ms)')
        plt.ylabel('Neuron #')

        if savefig:
            plt.savefig('RasterSpikeTrain.png')
        return plt.show()

    @staticmethod
    def correlation(corr, savefig):

        """ Plot correlation between neurons"""

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(11, 9))

        # Custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio

        sns.heatmap(corr, mask=mask, cmap=cmap, xticklabels=5, yticklabels=5, vmax=.1, center=0,
                    square=False, linewidths=0.0, cbar_kws={"shrink": .9})
        if savefig:
            plt.savefig('Correlation between neurons')
        return None

    @staticmethod
    def isi_exponential_fit(spike_train, neuron, bin_size, savefig):

        """Args:
            :param spike_train (array) - Array of spike trains
            :param neuron(int) - If True, firing rate of the network will be plotted
            :param bin_size(int) - Spike train will be splitted into bins of size bin_size
            :param savefig(bool) - If True, plot will be saved in the cwd

            Returns:
            plot object"""

        spike_time = Statistics.	spike_times(spike_train.T[neuron])  # Locate the spike time of the target neuron

        isi = Statistics.spike_time_intervals(spike_time)  # ISI intervals of neuron

        y, x = np.histogram(sorted(isi), bins=bin_size)

        x = [int(i) for i in x]
        y = [float(i) for i in y]

        def exponential_func(y, a, b, c):
            return a * np.exp(-b * np.array(y)) - c

        # Curve fit
        popt, pcov = curve_fit(exponential_func, x[1:bin_size], y[1:bin_size])

        # Plot
        plt.plot(x[1:bin_size], exponential_func(x[1:bin_size], *popt), label='Exponential fit')
        plt.scatter(x[1:bin_size], y[1:bin_size], s=2.0, color='black', label='ISI')
        plt.xlabel('ISI(time step)')
        plt.ylabel('Frequency')
        plt.legend()

        if savefig:
            plt.savefig('isi_exponential_fit')
        return plt.show()

    @staticmethod
    def weight_distribution(weights, bin_size, savefig):
        """Args:
            :param weights (array) - Connection weights
            :param bin_size(int) - Spike train will be splited into bins of size bin_size
            :param savefig(bool) - If True, plot will be saved in the cwd

            Returns:
            plot object"""

        weights = weights[weights >= 0.01]  # Remove the weight values less than 0.01 # As reported in article SORN 2013
        y, x = np.histogram(weights, bins=bin_size)  # Create histogram with bin_size

        plt.scatter(x[:-1], y, s=2.0, c='black')
        plt.xlabel('Weight')
        plt.ylabel('Frequency')

        if savefig:
            plt.savefig('weight distribution')

        return plt.show()

    @staticmethod
    def linear_lognormal_fit(weights,num_points, savefig):

        """Args:
            :param weights (array) - Connection weights
            :param num_points(int) - Number of points to be plotted in the x axis
            :param savefig(bool) - If True, plot will be saved in the cwd

            Returns:
            plot object"""

        X = weights.copy()

        M = float(np.mean(X))  # Geometric mean
        s = float(np.std(X))  # Geometric standard deviation

        # Lognormal distribution parameters

        mu = float(np.mean(np.log(X)))  # Mean of log(X)
        sigma = float(np.std(np.log(X)))  # Standard deviation of log(X)
        shape = sigma  # Scipy's shape parameter
        scale = np.exp(mu)  # Scipy's scale parameter
        median = np.exp(mu)

        mode = np.exp(mu - sigma ** 2)  # Note that mode depends on both M and s
        mean = np.exp(mu + (sigma ** 2 / 2))  # Note that mean depends on both M and s
        x = np.linspace(np.min(weights), np.max(weights), num=num_points)  # values for x-axis

        pdf = stats.lognorm.pdf(x, shape, loc=0, scale=scale)  # probability distribution

        plt.figure(figsize=(12, 4.5))

        # Figure on linear scale
        plt.subplot(121)
        plt.plot(x, pdf)

        plt.vlines(mode, 0, pdf.max(), linestyle=':', label='Mode')
        plt.vlines(mean, 0, stats.lognorm.pdf(mean, shape, loc=0, scale=scale), linestyle='--', color='green',
                   label='Mean')
        plt.vlines(median, 0, stats.lognorm.pdf(median, shape, loc=0, scale=scale), color='blue', label='Median')
        plt.ylim(ymin=0)
        plt.xlabel('Weight')
        plt.title('Linear scale')
        plt.legend()

        # Figure on logarithmic scale
        plt.subplot(122)
        plt.semilogx(x, pdf)

        plt.vlines(mode, 0, pdf.max(), linestyle=':', label='Mode')
        plt.vlines(mean, 0, stats.lognorm.pdf(mean, shape, loc=0, scale=scale), linestyle='--', color='green',
                   label='Mean')
        plt.vlines(median, 0, stats.lognorm.pdf(median, shape, loc=0, scale=scale), color='blue', label='Median')
        plt.ylim(ymin=0)
        plt.xlabel('Weight')
        plt.title('Logarithmic scale')
        plt.legend()

        if savefig:
            plt.savefig('LinearLognormalFit')

        return plt.show()


    @staticmethod
    def hamming_distance(hamming_dist, savefig):

        plt.figure(figsize=(15, 6))
        plt.title("Hamming distance between actual and perturbed states")
        plt.xlabel("Time steps")
        plt.ylabel("Hamming distance")
        plt.plot(hamming_dist)

        if savefig:
            plt.savefig('HammingDistance')

        return plt.show()


class Statistics(object):

    def __init__(self):
        pass

    @staticmethod
    def firing_rate_neuron(spike_train, neuron, bin_size):

        # Measure spike rate of given neuron during given time window

        """Args:
                :param spike_train(array) - Array of spike trains
                :param neuron(int) - Target neuron in the reservoir
                :param bin_size(int) - Divide the spike trains into bins of size bin_size

                Returns: firing_rate """

        time_period = len(spike_train[:, 0])

        neuron_spike_train = spike_train[:, neuron]

        # Split the list(neuron_spike_train) into sub lists of length time_step
        samples_spike_train = [neuron_spike_train[i:i + bin_size] for i in
                               range(0, len(neuron_spike_train), bin_size)]

        spike_rate = 0.

        for idx, spike_train in enumerate(samples_spike_train):
            spike_rate += list(spike_train).count(1.)

        spike_rate = spike_rate * bin_size / time_period

        # print('Firing rate of neuron %s in %s time steps is %s' %(neuron,time_step,spike_rate/time_step))

        return time_period, bin_size, spike_rate

    @staticmethod
    def firing_rate_network(spike_train):

        """Args:
        :param spike_train(array) - Array of spike trains

        Returns: firing_rate """

        firing_rate = np.count_nonzero(spike_train.tolist(), 1)

        return firing_rate

    @staticmethod
    def scale_dependent_smoothness_measure(firing_rates):

        # Smaller values corresponds to smoother series
        """
        Args:

        firing_rates - List of number of active neurons per time step

        Returns:

        sd_diff - Float value signifies the smoothness of the semantic changes in firing rates
        """

        diff = np.diff(firing_rates)
        sd_diff = np.std(diff)

        return sd_diff

    @staticmethod
    def scale_independent_smoothness_measure(firing_rates):

        # Smaller values corresponds to smoother series
        """
        Args:
        firing_rates - List of number of active neurons per time step

        Returns:
        coeff_var - Float value signifies the smoothness of the semantic changes in firing rates """

        diff = np.diff(firing_rates)
        mean_diff = np.mean(diff)
        sd_diff = np.std(diff)

        coeff_var = sd_diff / abs(mean_diff)

        return coeff_var

    # Using one-lag auto-correlation measure

    @staticmethod
    def autocorr(firing_rates, t=2):

        """
        Score interpretation:

        scores near 1   imply a smoothly varying series
        scores near 0   imply that there's no overall linear relationship between a data point and the following one
                        (that is, plot(x[-length(x)],x[-1]) won't give a scatter plot with any apparent linearity)
        scores near -1  suggest that the series is jagged in a particular way: if one point is above the mean, the next
                        is likely to be below the mean by about the same amount, and vice versa."""

        return np.corrcoef(np.array([firing_rates[0:len(firing_rates) - t], firing_rates[t:len(firing_rates)]]))

    @staticmethod
    def avg_corr_coeff(spike_train):

        # Measure Average Pearson correlation coeffecient between neurons

        corr_mat = np.corrcoef(np.asarray(spike_train).T)
        avg_corr = np.sum(corr_mat, axis=1) / 200
        corr_coeff = avg_corr.sum() / 200  # 2D to 1D and either upper  or lower half of correlation matrix.

        return corr_mat, corr_coeff

    @staticmethod
    def spike_times(spike_train):

        """ Get the time instants at which neuron spikes"""

        times = np.where(spike_train == 1.)
        return times

    @staticmethod
    def spike_time_intervals(spike_train):

        """ Generate spike time intervals|spike_trains"""

        spike_times = Statistics.spike_times(spike_train)
        # isi = sorted(np.diff(spike_times)[-1])
        isi = np.diff(spike_times)[-1]
        return isi

    @staticmethod
    def hamming_distance(actual_spike_train, perturbed_spike_train):

        """ Hamming distance between  """
        hd = [np.count_nonzero(actual_spike_train[i] != perturbed_spike_train[i]) for i in range(len(actual_spike_train))]
        return hd

    # Fano Factor

    @staticmethod
    def fanofactor(spike_train,neuron,window_size):

        """Investigate whether neuronal spike generation is a poisson process"""

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

    # Spike Source Entropy

    @staticmethod
    def spike_source_entropy(spike_train, neurons_in_reservoir):

        # Uncertainty about the origin of spike from the network

        # TODO: Remove neurons_in_reservoir in future versions

        # Number of spikes from each neuron during the interval

        n_spikes = np.count_nonzero(spike_train, axis=0)
        p = n_spikes / np.count_nonzero(spike_train)  # Probability of each neuron that can generate spike in next step
        # print(p)  # Note: pi shouldn't be zero
        sse = np.sum([pi * np.log(pi) for pi in p]) / np.log(1 / neurons_in_reservoir)  # Spike source entropy

        return sse

    @staticmethod
    def mca(spike_train):
        pass
    	
