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


class Callbacks:
    def __init__(self, timesteps, avail_callbacks, *argv):
        self.timesteps = timesteps
        self.argv = argv
        if argv:
            self.dispatcher = Validate().callbacks(argv, avail_callbacks)

            self.callbacks = []
            for class_name in list(self.dispatcher.keys()):
                instance = self.create(class_name)
                self.dispatcher[class_name] = instance

    def create(self, class_name) -> object:
        instance = globals()[class_name](self.timesteps)
        return instance

    def update(self, func, value, timestep):
        try:
            func[timestep] = value
        except:
            return f"Invalid callback instance {func}"

    def step(self, state, time_step) -> None:

        for callback in list(self.dispatcher.keys()):
            self.update(self.dispatcher[callback], state[callback], time_step)

    def get(self) -> dict:

        if self.argv:
            for name, callback in self.dispatcher.items():
                self.dispatcher[name] = callback.values
            return self.dispatcher

        else:
            return {}


class Validate:
    def __init__(self):
        pass

    def callbacks(self, req_callbacks, avail_callbacks):

        self.argv = req_callbacks
        self.avail_callbacks = list(avail_callbacks.keys())

        if self.argv:
            return self.assert_callbacks()

        else:
            return None

    def assert_callbacks(self) -> dict:

        if not set(*self.argv) == set(self.avail_callbacks):
            assert set(*self.argv).issubset(
                set(self.avail_callbacks)
            ), f"{list(set(*self.argv)-set(self.avail_callbacks))} not available"

            return dict.fromkeys(set(*self.argv) & set(self.avail_callbacks))

        else:
            return dict.fromkeys(self.avail_callbacks)


class ExcitatoryActivation:
    def __init__(self, timesteps=0):
        self.values = [0] * timesteps

    def __setitem__(self, index, value):
        self.values[index] = value

    def __getitem__(self, index):
        return f"Excitatory network state at time_step {index}: {self.values[index]}"

    def __str__(self):
        return str(self.values)


class InhibitoryActivation:
    def __init__(self, timesteps=0):
        self.values = [0] * timesteps

    def __setitem__(self, index, value):
        self.values[index] = value

    def __getitem__(self, index):
        return f"Inhibitory network state at time_step {index}: {self.values[index]}"

    def __str__(self):
        return str(self.values)


class RecurrentActivation:
    def __init__(self, timesteps=0):
        self.values = [0] * timesteps

    def __setitem__(self, index, value):
        self.values[index] = value

    def __getitem__(self, index):
        return f"Recurrent network state at time_step {index}: {self.values[index]}"

    def __str__(self):
        return str(self.values)


class WEE:
    def __init__(self, timesteps=0):
        self.values = [0] * timesteps

    def __setitem__(self, index, value):
        self.values[index] = value

    def __getitem__(self, index):
        return f"Excitatory to Excitatory Connection strength at time_step {index}: {self.values[index]}"

    def __str__(self):
        return str(self.values)


class WEI:
    def __init__(self, timesteps=0):
        self.values = [0] * timesteps

    def __setitem__(self, index, value):
        self.values[index] = value

    def __getitem__(self, index):
        return f"Inhibitory to Excitatory Connection strength at time_step {index}: {self.values[index]}"

    def __str__(self):
        return str(self.values)


class TE:
    def __init__(self, timesteps=0):
        self.values = [0] * timesteps

    def __setitem__(self, index, value):
        self.values[index] = value

    def __getitem__(self, index):
        return f"Excitatory neurons firing threshold values at time_step {index}: {self.values[index]}"

    def __str__(self):
        return str(self.values)


class TI:
    def __init__(self, timesteps=0):
        self.values = [0] * timesteps

    def __setitem__(self, index, value):
        self.values[index] = value

    def __getitem__(self, index):
        return f"Inhibitory neurons firing threshold values at time_step {index}: {self.values[index]}"

    def __str__(self):
        return str(self.values)


class EEConnectionCounts:
    def __init__(self, timesteps=0):
        self.values = [0] * timesteps

    def __setitem__(self, index, value):
        self.values[index] = value

    def __getitem__(self, index):
        return f"Number active connections in the Excitatory pool at time_step {index}: {self.values[index]}"

    def __str__(self):
        return str(self.values)


class EIConnectionCounts:
    def __init__(self, timesteps=0):
        self.values = [0] * timesteps

    def __setitem__(self, index, value):
        self.values[index] = value

    def __getitem__(self, index):
        return f"Number active connections from Inhibitory to Excitatory pool at time_step {index}: {self.values[index]}"

    def __str__(self):
        return str(self.values)


class IEConnectionCounts:
    def __init__(self, timesteps=0):
        self.values = [0] * timesteps

    def __setitem__(self, index, value):
        self.values[index] = value

    def __getitem__(self, index):
        return f"Number active connections from Excitatory to Inhibitory pool at time_step {index}: {self.values[index]}"

    def __str__(self):
        return str(self.values)
