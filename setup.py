import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "sorn",
    version = "0.2.7",
    author = "Saranraj Nambusubramaniyan",
    author_email = "saran_nns@hotmail.com",
    description ="Self-Organizing Recurrent Neural Networks",
    license = "OSI Approved :: MIT License",
    keywords = """Brain-Inspired Computing,Artificial Neural Networks,Neuro Informatics, 
                  Spiking Cortical Networks, Neural Connectomics,Neuroscience, Artificial General Intelligence, Neural Information Processing""" ,
    url = "https://github.com/Saran-nns/sorn",
    packages=['sorn'],
    data_files = ["LICENSE"],
    long_description=read('README.md'),
    long_description_content_type = "text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    include_package_data = True,
    install_requires = ['numpy','configparser','tqdm','scipy','seaborn'],
    zip_safe = False )

