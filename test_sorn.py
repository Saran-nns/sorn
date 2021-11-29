import unittest
import pickle
import numpy as np
from sorn.sorn import Trainer, Simulator
from sorn.utils import Plotter, Statistics, Initializer

# Getting back the pickled matrices:
with open("sample_matrices.pkl", "rb") as f:
    (
        state_dict,
        Exc_activity,
        Inh_activity,
        Rec_activity,
        num_active_connections,
    ) = pickle.load(f)

# Default Test inputs
simulation_inputs = np.random.rand(10, 2)
gym_input = np.random.rand(10, 1)
sequence_input = np.random.rand(10, 1)

# Overriding defaults: Sample input
num_features = 10
time_steps = 1000
inputs = np.random.rand(num_features, time_steps)


class TestSorn(unittest.TestCase):
    def test_runsorn(self):
        """Test SORN initialization, simulation and training methods with different
        initialization schemes and plasticity rules
        """

        # Initialize and simulate SORN with the default hyperparameters
        self.assertRaises(
            Exception,
            Simulator.simulate_sorn(
                inputs=simulation_inputs,
                phase="plasticity",
                matrices=None,
                time_steps=2,
                noise=True,
                nu=num_features,
            ),
        )
        # Initilize and resume the simulation of SORN using the state dictionary, state_dict
        self.assertRaises(
            Exception,
            Simulator.simulate_sorn(
                inputs=simulation_inputs,
                phase="plasticity",
                matrices=state_dict,
                time_steps=2,
                noise=False,
            ),
        )
        # Freeze a particular plasticity during simulation
        self.assertRaises(
            Exception,
            Simulator.simulate_sorn(
                inputs=simulation_inputs,
                phase="plasticity",
                matrices=state_dict,
                time_steps=2,
                noise=False,
                freeze=["ip"],
            ),
        )

        # Freeze multiple plasticity mechanisms during simulation
        self.assertRaises(
            Exception,
            Simulator.simulate_sorn(
                inputs=simulation_inputs,
                phase="plasticity",
                matrices=state_dict,
                time_steps=2,
                noise=False,
                freeze=["stdp", "istdp", "ss", "sp"],
            ),
        )

        # Train SORN with all plasticity mechanisms active
        self.assertRaises(
            Exception,
            Trainer.train_sorn(
                inputs=gym_input,
                phase="plasticity",
                matrices=state_dict,
                time_steps=1,
                noise=True,
            ),
        )
        # Freeze multiple plasticity mechanisms during training
        self.assertRaises(
            Exception,
            Trainer.train_sorn(
                inputs=sequence_input,
                phase="training",
                matrices=state_dict,
                time_steps=1,
                noise=True,
                freeze=["stdp", "istdp", "ss", "sp"],
            ),
        )

        # Override the default hyperparameters, initialize SORN and simulate under all plasticity mechanisms
        self.assertRaises(
            Exception,
            Simulator.simulate_sorn(
                inputs=inputs,
                phase="plasticity",
                matrices=None,
                noise=True,
                time_steps=time_steps,
                ne=26,
                lambda_ee=4,
                lambda_ei=4,
                nu=num_features,
            ),
        )

        # Override the default hyperparameters, initialize SORN and train under all plasticity mechanisms
        self.assertRaises(
            Exception,
            Trainer.train_sorn(
                inputs=inputs,
                phase="plasticity",
                matrices=None,
                noise=True,
                time_steps=time_steps,
                ne=40,
                lambda_ee=5,
                lambda_ei=5,
                nu=num_features,
            ),
        )

    def test_plotter(self):
        """Test the Plotter class methods in utils module"""

        # Histogram of number of postsynaptic connections per neuron in the excitatory pool
        self.assertRaises(
            Exception,
            Plotter.hist_outgoing_conn(
                weights=state_dict["Wee"], bin_size=5, histtype="bar", savefig=False
            ),
        )
        # Histogram of number of presynaptic connections per neuron in the excitatory pool
        self.assertRaises(
            Exception,
            Plotter.hist_incoming_conn(
                weights=state_dict["Wee"], bin_size=5, histtype="bar", savefig=False
            ),
        )
        # Plot number of positive connection strengths (weights>0) in the network at each time step
        self.assertRaises(
            Exception,
            Plotter.network_connection_dynamics(
                connection_counts=num_active_connections,
                savefig=False,
            ),
        )
        # Histogram of firing rate of the network
        self.assertRaises(
            Exception,
            Plotter.hist_firing_rate_network(
                spike_train=np.asarray(Exc_activity), bin_size=5, savefig=False
            ),
        )
        # Plot Spike train of all neurons in the network
        self.assertRaises(
            Exception,
            Plotter.scatter_plot(spike_train=np.asarray(Exc_activity), savefig=False),
        )
        # Raster plot of activity of neurons in the excitatory pool
        self.assertRaises(
            Exception,
            Plotter.raster_plot(spike_train=np.asarray(Exc_activity), savefig=False),
        )
        # Inter spike intervals with exponential curve fit for neuron 10 in the Excitatory pool
        self.assertRaises(
            Exception,
            Plotter.isi_exponential_fit(
                spike_train=np.asarray(Exc_activity),
                neuron=10,
                bin_size=5,
                savefig=False,
            ),
        )

        # Plot weight distribution in the network
        self.assertRaises(
            Exception,
            Plotter.weight_distribution(
                weights=state_dict["Wee"], bin_size=5, savefig=False
            ),
        )
        # Distribution of connection weights in linear and lognormal scale
        self.assertRaises(
            Exception,
            Plotter.linear_lognormal_fit(
                weights=state_dict["Wee"], num_points=10, savefig=False
            ),
        )

        self.assertRaises(
            Exception,
            Plotter.hamming_distance(
                hamming_dist=[0, 0, 0, 1, 1, 1, 1, 1, 1], savefig=False
            ),
        )

    def test_statistics(self):
        """Test the functions in Statistics class"""
        # Firing rate of a neuron
        self.assertRaises(
            Exception,
            Statistics.firing_rate_neuron(
                spike_train=np.asarray(Exc_activity), neuron=10, bin_size=5
            ),
        )

        # Firing rate of the network
        self.assertRaises(
            Exception,
            Statistics.firing_rate_network(spike_train=np.asarray(Exc_activity)),
        )
        # Smoothness of the firing rate curve
        self.assertRaises(
            Exception,
            Statistics.scale_dependent_smoothness_measure(
                firing_rates=[1, 1, 5, 6, 3, 7]
            ),
        )
        # t lagged auto correlation between neurons given their firing rates
        self.assertRaises(
            Exception, Statistics.autocorr(firing_rates=[1, 1, 5, 6, 3, 7], t=2)
        )
        # Average of pearson correlation between neurons
        self.assertRaises(
            Exception, Statistics.avg_corr_coeff(spike_train=np.asarray(Exc_activity))
        )
        # Return the spike event times of each neuron in the pool
        self.assertRaises(
            Exception, Statistics.spike_times(spike_train=np.asarray(Exc_activity))
        )
        # Hamming distance measure for stability analysis
        self.assertRaises(
            Exception,
            Statistics.hamming_distance(
                actual_spike_train=np.asarray(Exc_activity),
                perturbed_spike_train=np.asarray(Exc_activity),
            ),
        )
        #  Inter spike interval of neurons
        self.assertRaises(
            Exception,
            Statistics.spike_time_intervals(spike_train=np.asarray(Exc_activity)),
        )
        # Verify whether the neural spiking obeys poisson
        self.assertRaises(
            Exception,
            Statistics.fanofactor(
                spike_train=np.asarray(Exc_activity), neuron=10, window_size=10
            ),
        )
        # Degree of uncertainty in the origin of spiking
        self.assertRaises(
            Exception,
            Statistics.spike_source_entropy(
                spike_train=np.asarray(Exc_activity), num_neurons=200
            ),
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
