import unittest
import pickle
import numpy as np
from sorn.sorn import RunSorn,Generator
from sorn.utils import Plotter,Statistics,Initializer


# Get the pickled matrices:
with open('sample_matrices.pkl','rb') as f:
    matrices_dict, Exc_activity, Inh_activity, Rec_activity, num_active_connections = pickle.load(f)


class TestSorn(unittest.TestCase):
	
	def test_runsorn(self):
		self.assertRaises(Exception, Generator().get_initial_matrices()) 

		matrices_dict = Generator().get_initial_matrices()
        
		self.assertRaises(Exception, RunSorn(phase='Plasticity', matrices=None).run_sorn([0.]))

		self.assertRaises(Exception, RunSorn(phase='Training', matrices=matrices_dict).run_sorn([0.]))
        	
	def test_plotter(self):

		self.assertRaises(Exception, Plotter.hist_outgoing_conn(weights= matrices_dict['Wee'],bin_size=5, histtype = 'bar',savefig = False))

		self.assertRaises(Exception, Plotter.hist_outgoing_conn(weights= matrices_dict['Wee'],bin_size=5, histtype = 'bar',savefig = False))

		self.assertRaises(Exception, Plotter.hist_incoming_conn(weights= matrices_dict['Wee'],bin_size=5, histtype = 'bar',savefig = False))

		self.assertRaises(Exception, Plotter.network_connection_dynamics(connection_counts= num_active_connections, initial_steps= 10, final_steps = 10,savefig = False))

		self.assertRaises(Exception, Plotter.hist_firing_rate_network(spike_train = np.asarray(Exc_activity), bin_size = 5, savefig= False))
		
		self.assertRaises(Exception, Plotter.scatter_plot(spike_train = np.asarray(Exc_activity), savefig=False))

		self.assertRaises(Exception, Plotter.raster_plot(spike_train = np.asarray(Exc_activity), savefig=False))

		self.assertRaises(Exception, Plotter.isi_exponential_fit(spike_train = np.asarray(Exc_activity), neuron = 10, bin_size = 5, savefig = False))

		self.assertRaises(Exception, Plotter.weight_distribution(weights= matrices_dict['Wee'], bin_size = 5, savefig = False))

		self.assertRaises(Exception, Plotter.linear_lognormal_fit(weights= matrices_dict['Wee'],num_points = 10, savefig = False))

		self.assertRaises(Exception, Plotter.hamming_distance(hamming_dist=[0,0,0,1,1,1,1,1,1], savefig = False))


	def test_statistics(self):

		self.assertRaises(Exception, Statistics.firing_rate_neuron(spike_train = np.asarray(Exc_activity), neuron = 10, bin_size = 5))

		self.assertRaises(Exception, Statistics.firing_rate_network(spike_train = np.asarray(Exc_activity)))

		self.assertRaises(Exception, Statistics.scale_dependent_smoothness_measure(firing_rates = [1,1,5,6,3,7]))

		self.assertRaises(Exception, Statistics.autocorr(firing_rates = [1,1,5,6,3,7],t= 2))

		self.assertRaises(Exception, Statistics.avg_corr_coeff(spike_train =np.asarray(Exc_activity)))

		self.assertRaises(Exception, Statistics.spike_times(spike_train =np.asarray(Exc_activity)))

		self.assertRaises(Exception, Statistics.hamming_distance(actual_spike_train =np.asarray(Exc_activity), perturbed_spike_train =np.asarray(Exc_activity)))

		self.assertRaises(Exception, Statistics.spike_time_intervals(spike_train =np.asarray(Exc_activity)))

		self.assertRaises(Exception, Statistics.fanofactor(spike_train= np.asarray(Exc_activity),neuron = 10,window_size = 10))

		self.assertRaises(Exception, Statistics.spike_source_entropy(spike_train =np.asarray(Exc_activity), neurons_in_reservoir = 200))


if __name__ == '__main__':
	unittest.main()
