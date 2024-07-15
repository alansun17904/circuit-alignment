SynapticSync API
==========================
This API is split into submodules. The ``dproc`` module contains well-known fMRI datasets wrapped with their necessary pre-processing. 
The ``model`` module contains a wrapper for a ``HookedTransformer`` model that automatically keeps track of the internals of any HuggingFace model required for both the calculation of brain alignment and circuit discovery.This module also contains logic for patching.
Lastly, the ``utils`` model contains a variety of functionality including tools for running Ridge regression, alignment calculations, and visualizations.

.. toctree::
   :maxdepth: 2

   circuit_brain.dproc
   circuit_brain.model
   circuit_brain.utils
