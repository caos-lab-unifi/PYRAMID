.. _installation:

Installation
============

Requirements
------------

PYRAMID requires **Python ≥ 3.10**. All Python dependencies are declared in ``pyproject.toml`` and are installed automatically.

.. note::

   PYRAMID is developed and tested on Linux (x86-64). Compatibility with 
   macOS and Windows has not been formally tested.

Core dependencies
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Package
     - Purpose
   * - ``pandas ≤ 2.1.0``
     - Data I/O and manipulation
   * - ``numpy``
     - Numerical arrays
   * - ``scikit-learn == 1.6.1``
     - Machine learning estimators and pipelines
   * - ``scikit-survival``
     - Survival analysis estimators (CoxPH, RSF, GBS, SSVM …)
   * - ``scipy``
     - Statistical tests
   * - ``imbalanced-learn``
     - Class-imbalance resampling (SMOTE, ADASYN …)
   * - ``matplotlib``, ``seaborn``
     - Plotting
   * - ``joblib``
     - Model serialisation and parallel computation
   * - ``lifelines``
     - Kaplan-Meier estimation and log-rank tests
   * - ``plotly``
     - Interactive visualisation (optional outputs)

Install from source
-------------------

Clone the repository and install in editable mode:

.. code-block:: bash
   
   conda create -y -n pyramidenv "python>3.10"
   conda activate pyramidenv
   git clone https://github.com/caos-lab-unifi/PYRAMID.git
   cd PYRAMID
   pip install .

The ``pip install`` step reads ``pyproject.toml`` and resolves all dependencies automatically.

Verify the installation
-----------------------

.. code-block:: bash

   PYRAMID --help

You should see the ASCII banner followed by the list of available submodules.

.. note::

   PYRAMID submodule names are **case-insensitive** on the command line.
   ``PYRAMID datrax`` and ``PYRAMID DaTrax`` are equivalent.
