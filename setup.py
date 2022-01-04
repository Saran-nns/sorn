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

import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="sorn",
    version="0.7.3",
    author="Saranraj Nambusubramaniyan",
    author_email="saran_nns@hotmail.com",
    description="Self-Organizing Recurrent Neural Networks",
    license="OSI Approved :: MIT License",
    keywords="""Brain-Inspired Computing,Artificial Neural Networks,Neuro Informatics,
                  Spiking Cortical Networks, Neural Connectomics,Neuroscience, Artificial General Intelligence, Neural Information Processing""",
    url="https://github.com/Saran-nns/sorn",
    packages=["sorn"],
    data_files=["LICENSE.md"],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    include_package_data=True,
    install_requires=[
        "numpy",
        "configparser",
        "scipy",
        "seaborn",
        "pandas",
        "networkx",
    ],
    zip_safe=False,
)
