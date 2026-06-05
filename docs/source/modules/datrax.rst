.. _datrax:

DaTrax — Data Transformation
=============================

**DaTrax** applies a feature-space transformation to a radiomics training dataset and, optionally, a paired validation dataset. The same scaler is fitted on the training set and applied to the test set. Additional covariate TSV files can be joined before transformation.

Usage
-----

.. code-block:: bash

   PYRAMID DaTrax -i TRAIN.tsv -o OUTPUT_DIR -t TRANSFORMATION [options]

Required arguments
------------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Flag
     - Metavar
     - Description
   * - ``-i`` / ``--input``
     - ``TSV``
     - Training feature matrix (must have a ``ptid`` index column).
   * - ``-o`` / ``--output``
     - ``DIR``
     - Output directory (created if it does not exist; must be empty).
   * - ``-t`` / ``--transformation``
     - ``str``
     - Transformation to apply. See :ref:`datrax_transformations` for details.

Optional arguments
------------------

.. list-table::
   :widths: 30 15 15 40
   :header-rows: 1

   * - Flag
     - Metavar
     - Default
     - Description
   * - ``-v`` / ``--validation``
     - ``TSV``
     - —
     - Validation / test feature matrix. When provided, both train and test outputs are written.
   * - ``--additional_training``
     - ``TSV``
     - —
     - Extra covariate TSV to join with the training matrix before transformation.
   * - ``--additional_validation``
     - ``TSV``
     - —
     - Extra covariate TSV to join with the validation matrix before transformation.
   * - ``--robust-parameter``
     - ``VAL1,VAL2``
     - ``25.0,75.0``
     - Quantile range for the robust scaler (only used when ``-t robust``).
   * - ``--n-quantiles``
     - ``int``
     - ``100``
     - Number of quantiles (only used when ``-t quantile-uniform`` or ``quantile-normal``).

.. _datrax_transformations:

Transformations
---------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - ``scaler``
     - Z-score standardisation (zero mean, unit variance) using ``StandardScaler``.
   * - ``normalize``
     - Min–max scaling to the [0, 1] range using ``MinMaxScaler``.
   * - ``mixture``
     - Z-score standardisation followed by min–max normalisation.
   * - ``robust``
     - Robust scaling using inter-quantile range (default IQR: 25th–75th percentile). Provide a custom range with ``--robust-parameter``.
   * - ``yeo-johnson``
     - Power transformation using the Yeo-Johnson method (handles negative values). Falls back to a pre-standardised version if the direct fit fails due to extreme outliers.
   * - ``box-cox``
     - Box-Cox power transformation. Values are first rescaled to [1, 2] to satisfy the positivity constraint.
   * - ``quantile-uniform``
     - Quantile transformation mapping the marginal distribution to a uniform distribution.
   * - ``quantile-normal``
     - Quantile transformation mapping the marginal distribution to a normal (Gaussian) distribution.

.. note::

   When a validation set is provided (``-v``), only the columns present in **both** train and test matrices after transformation are retained. This ensures feature alignment at prediction time.

Outputs
-------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - File
     - Description
   * - ``train.transformed.tsv``
     - Transformed training matrix, tab-separated, indexed by ``ptid``.
   * - ``test.transformed.tsv``
     - Transformed validation matrix (only written when ``-v`` is provided).

Example
-------

.. code-block:: bash

   # Z-score standardise train and test
   PYRAMID DaTrax \
       -i features/train.tsv \
       -v features/test.tsv \
       -o results/DaTrax \
       -t scaler

   # Robust scaling with a custom IQR range
   PYRAMID DaTrax \
       -i features/train.tsv \
       -o results/DaTrax_robust \
       -t robust \
       --robust-parameter 10.0,90.0
