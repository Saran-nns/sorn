.. Self-Organizing Recurrent Neural Network (SORN) documentation master file, created by
   sphinx-quickstart on Sat Jan 16 23:59:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Self-Organizing Recurrent Neural Network (SORN)
===========================================================================
Self-Organizing Recurrent Neural (SORN) networks are a class of reservoir computing models build based on plasticity mechanisms 
in biological brain. Recent studies on SORN shows that such models can mimic neocortical circuit’s ability of learning and adaptation 
through neuroplasticity mechanisms. Structurally, unlike other liquid state models, SORN consists of pool of excitatory neurons and 
small population of inhibitory neurons. First such network was introduced with three fundamental plasticity mechanisms found in 
neocortex, namely Spike timing dependent plasticity (STDP), intrinsic plasticity (IP) and Synaptic scaling (SS). Spike 
Timing-Dependent Plasticity or Hebbian Learning with positive feedback (rapid cycle of synaptic potentials) selectively strengthens 
correlated synapses and weaken the uncorrelated. Such activity dependent rules lead to Long Time Potentiation (LTP) and Long Time 
Depression (LTD).

Biologically, both LTP and LDP are assumed to possess substrates of learning and memory at the cellular level of neocortex. 
However, in dynamical systems, such phenomena will drive the network either towards the state of bursting activity in case of LTP or 
towards state of attenuation due to LTD. These destabilizing influences of STDP are counteracted by homeostatic plasticity 
mechanisms. Homeostatic mechanisms are a set of negative feedback (action potential suppressing) regulatory mechanisms that 
scales incoming synaptic strengths and balances neuronal activity through synaptic normalization and intrinsic plasticity. Experimental 
evidences also prove that synaptic scaling found to balance the activity between excitatory and inhibitory neurons in-vivo.
Together, they maintain the overall activity of network within subcritical range, despite the network being driven by positive feedback 
from fast Hebbian plasticity.

In recent proposed models, SORN is extended with two more plasticity mechanisms, inhibitory spike timing dependent plasticity and 
structural plasticity. While connections between excitatory neurons (E-E) subjected to STDP rules, connections from 
inhibitory population to excitatory populations(E-I) are regulated by iSTDP. Structural plasticity, generates new connections 
constantly at a smaller rate between unconnected synapses. Many studies argued that, such structural changes induce neuronal morphogenesis 
which leads to network re-organization with functional consequences over learning and memory.
The mathematical descriptions of plasticity mechanisms proposed in SORN simplifies the structural and functional connectivity mechanisms 
that resembles information processing, learning and memory phenomena that occur in neuro-synapses of neocortex region. Recent experimental 
evidences confirm that SORN outperforms other static reservoir networks in spatio-temporal tasks and maintains the dynamics of the network 
in subcritical state suitable for learning. Further research on such network mechanisms unravels the underlying features of 
synaptic connections and network activity in real cortical circuits. Hence investigating the characteristics of SORN and extending its 
structural and functional attributes by replicating the recent findings in neural connectomics may reveal the dominating principles of 
self-organization and self-adaptation in neocortical circuits at microscopic level. Moreover, characterizing these mechanisms individually 
at that level may also help us to understand some fundamental aspects of brain networks at mesoscopic and macroscopic scales.

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   
   Installation<installation>

   Usage<usage>

.. toctree::
   :maxdepth: 1
   :caption: Reference Guide

   Attributes and Methods <reference>

.. automodule:: sorn 
   :members:

.. automodule:: utils 
   :members:

.. toctree::
   :maxdepth: 1
   :caption: Contribution

   Steps<contribution>

.. toctree::
   :maxdepth: 2
   :caption: Citation
   
   Software<software>

   Paper<paper>

.. toctree::
   :maxdepth: 1
   :caption: License

   License<license>

.. toctree::
   :maxdepth: 1
   :caption: Contact

   Reach me<contact>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
