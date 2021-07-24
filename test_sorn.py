import unittest
import pickle
import numpy as np
from sorn.sorn import Trainer, Simulator
from sorn.utils import Plotter, Statistics, Initializer

# Getting back the pickled matrices:
with open("sample_matrices.pkl", "rb") as f:
    (
        matrices_dict,
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
num_features = 4
time_steps = 1000
inputs = np.random.rand(num_features, time_steps)


class TestSorn(unittest.TestCase):
    def test_runsorn(self):
        self.assertRaises(
            Exception,
            Simulator.simulate_sorn(
                inputs=simulation_inputs,
                phase="plasticity",
                matrices=matrices_dict,
                time_steps=2,
                noise=True,
            ),
        )
        self.assertRaises(
            Exception,
            Trainer.train_sorn(
                inputs=gym_input,
                phase="plasticity",
                matrices=matrices_dict,
                time_steps=1,
                noise=True,
            ),
        )
        self.assertRaises(
            Exception,
            Trainer.train_sorn(
                inputs=sequence_input,
                phase="plasticity",
                matrices=matrices_dict,
                time_steps=1,
                noise=True,
            ),
        )
        self.assertRaises(
            Exception,
            Trainer.train_sorn(
                inputs=sequence_input,
                phase="training",
                matrices=matrices_dict,
                time_steps=1,
                noise=True,
            ),
        )
        self.assertRaises(
            Exception,
            Simulator.simulate_sorn(
                inputs=inputs,
                phase="plasticity",
                matrices=None,
                noise=True,
                time_steps=time_steps,
                _ne=26,
                _lambda_ee=4,
                _lambda_ei=4,
                _nu=num_features,
            ),
        )

        self.assertRaises(
            Exception,
            Trainer.train_sorn(
                inputs=inputs,
                phase="plasticity",
                matrices=None,
                noise=True,
                time_steps=time_steps,
                _ne=40,
                _lambda_ee=5,
                _lambda_ei=5,
                _nu=num_features,
            ),
        )

    def test_plotter(self):
        self.assertRaises(
            Exception,
            Plotter.hist_outgoing_conn(
                weights=matrices_dict["Wee"], bin_size=5, histtype="bar", savefig=False
            ),
        )
        self.assertRaises(
            Exception,
            Plotter.hist_incoming_conn(
                weights=matrices_dict["Wee"], bin_size=5, histtype="bar", savefig=False
            ),
        )
        self.assertRaises(
            Exception,
            Plotter.network_connection_dynamics(
                connection_counts=num_active_connections,
                initial_steps=10,
                final_steps=10,
                savefig=False,
            ),
        )
        self.assertRaises(
            Exception,
            Plotter.hist_firing_rate_network(
                spike_train=np.asarray(Exc_activity), bin_size=5, savefig=False
            ),
        )
        self.assertRaises(
            Exception,
            Plotter.scatter_plot(spike_train=np.asarray(Exc_activity), savefig=False),
        )
        self.assertRaises(
            Exception,
            Plotter.raster_plot(spike_train=np.asarray(Exc_activity), savefig=False),
        )
        self.assertRaises(
            Exception,
            Plotter.isi_exponential_fit(
                spike_train=np.asarray(Exc_activity),
                neuron=10,
                bin_size=5,
                savefig=False,
            ),
        )
        self.assertRaises(
            Exception,
            Plotter.weight_distribution(
                weights=matrices_dict["Wee"], bin_size=5, savefig=False
            ),
        )
        self.assertRaises(
            Exception,
            Plotter.linear_lognormal_fit(
                weights=matrices_dict["Wee"], num_points=10, savefig=False
            ),
        )
        self.assertRaises(
            Exception,
            Plotter.hamming_distance(
                hamming_dist=[0, 0, 0, 1, 1, 1, 1, 1, 1], savefig=False
            ),
        )

    def test_statistics(self):
        self.assertRaises(
            Exception,
            Statistics.firing_rate_neuron(
                spike_train=np.asarray(Exc_activity), neuron=10, bin_size=5
            ),
        )
        self.assertRaises(
            Exception,
            Statistics.firing_rate_network(spike_train=np.asarray(Exc_activity)),
        )
        self.assertRaises(
            Exception,
            Statistics.scale_dependent_smoothness_measure(
                firing_rates=[1, 1, 5, 6, 3, 7]
            ),
        )
        self.assertRaises(
            Exception, Statistics.autocorr(firing_rates=[1, 1, 5, 6, 3, 7], t=2)
        )
        self.assertRaises(
            Exception, Statistics.avg_corr_coeff(spike_train=np.asarray(Exc_activity))
        )
        self.assertRaises(
            Exception, Statistics.spike_times(spike_train=np.asarray(Exc_activity))
        )
        self.assertRaises(
            Exception,
            Statistics.hamming_distance(
                actual_spike_train=np.asarray(Exc_activity),
                perturbed_spike_train=np.asarray(Exc_activity),
            ),
        )
        self.assertRaises(
            Exception,
            Statistics.spike_time_intervals(spike_train=np.asarray(Exc_activity)),
        )
        self.assertRaises(
            Exception,
            Statistics.fanofactor(
                spike_train=np.asarray(Exc_activity), neuron=10, window_size=10
            ),
        )
        self.assertRaises(
            Exception,
            Statistics.spike_source_entropy(
                spike_train=np.asarray(Exc_activity), num_neurons=200
            ),
        )

if __name__ == "__main__":
    unittest.main()

