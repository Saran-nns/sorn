import unittest
from sorn import RunSorn


class TestRunSorn(unittest.TestRunSorn):


	def test_runsorn(self, inputs):


		try:
        	RunSorn(phase='Plasticity', matrices=None,time_steps=5).run_sorn(inputs)
    	
    	except ExceptionType:
        	self.fail("RunSorn() raised ExceptionType unexpectedly!")


if __name__ == '__main__':
	unittest.main()