Quick Start
============

Prerequisites
-------------
As a first step you need a Python environment with all required dependencies. 
The recommended way is to use Anaconda and to create a new environment using our predefined environment files in `environments <https://github.com/fuadyassin/NHS_PostProcessing/tree/main/environments>`_.

Use:

.. code-block::

    conda env create -f environments/environment.yml

Installation
-------------
**The Library is not yet avaialable on PyPi so it will have to be installed directly from the git repo**

To install the library use:

.. code-block::

    pip install git+https://github.com/fuadyassin/NHS_PostProcessing.git


If you want to install an editable version to implememt your own models or dataset you'll have to clone the repository  using:

.. code-block::

    git clone https://github.com/fuadyassin/NHS_PostProcessing.git


or just download the zip file `here <https://github.com/fuadyassin/NHS_PostProcessing/archive/refs/tags/v1.0.0.zip>`_

After this, you are then left with a directory called *NHS_PostProcessing* or *NHS_PostProcessing-main*.  Next, we’ll go to that directory and install a local, editable copy of the package:

.. code-block::

    cd NHS_PostProcessing
    pip install -e .

