import sorn
from sorn import Simulator
import numpy as np
import timeit

# Sample input
num_features = 10
time_steps = 10
inputs = np.random.rand(num_features, time_steps)

start_time = timeit.default_timer()

# Simulate the network with default hyperparameters under gaussian white noise
Simulator.simulate_sorn(
    inputs=inputs,
    phase="plasticity",
    matrices=None,
    noise=True,
    time_steps=time_steps,
    ne=200,
)

print(timeit.default_timer() - start_time)
