import unittest
from sorn.sorn import RunSorn

class TestRunSorn(unittest.TestCase):

	def test_runsorn(self):
		
        	self.assertRaises(Exception, RunSorn(phase='Plasticity', matrices=None,time_steps=5).run_sorn([0.])) 


if __name__ == '__main__':
	unittest.main()