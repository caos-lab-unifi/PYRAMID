.. _slic:

SLIC — Sampling for Leveling Imbalanced Classes
================================================

**SLIC** corrects class imbalance in a training dataset by applying an over-, under-, or combined sampling strategy from the *imbalanced-learn* library. It produces a resampled TSV ready to feed into the feature selection stage.

Usage
-----

.. code-block:: bash

   PYRAMID SLIC -i TRAIN.tsv -m METADATA.tsv -l LABEL -o OUTPUT_DIR -a ALGORITHM [options]

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
     - Training feature matrix (``ptid`` index column required).
   * - ``-m`` / ``--metadata``
     - ``TSV``
     - Metadata TSV containing at least the class label column. Must be indexed by ``ptid``.
   * - ``-l`` / ``--label``
     - ``str``
     - Column name in the metadata file that contains the binary class label.
   * - ``-o`` / ``--output``
     - ``DIR``
     - Output directory.
   * - ``-a`` / ``--algorithm``
     - ``str``
     - Resampling algorithm. See :ref:`slic_algorithms`.

Optional arguments
------------------

.. list-table::
   :widths: 30 15 15 40
   :header-rows: 1

   * - Flag
     - Metavar
     - Default
     - Description
   * - ``--additional``
     - ``TSV``
     - —
     - Extra covariate TSV to merge with the training matrix before sampling.
   * - ``--sampling-strategy``
     - ``float|str``
     - ``auto``
     - Resampling ratio for the minority class. Accepts a float in [0.0, 1.0], ``"auto"`` (balance all classes), or ``"all"`` (resample all non-majority classes).
   * - ``-t`` / ``--threads``
     - ``int``
     - ``10``
     - Number of parallel threads.
   * - ``--seed``
     - ``int``
     - ``None``
     - Random seed for reproducibility.
   * - ``--neighbors``
     - ``int``
     - ``5``
     - Number of nearest neighbours (used by neighbour-based algorithms).

.. _slic_algorithms:

Algorithms
----------

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``SMOTE``
     - Over-sampling
     - Synthetic Minority Over-sampling TEchnique. Generates synthetic minority samples by interpolating between existing ones.
   * - ``ADASYN``
     - Over-sampling
     - Adaptive Synthetic Sampling. Like SMOTE but focuses synthetic generation on harder-to-learn examples.
   * - ``SMOTETomek``
     - Combined
     - SMOTE followed by Tomek-links under-sampling to clean the boundary between classes.
   * - ``SMOTEEN``
     - Combined
     - SMOTE followed by Edited Nearest Neighbours cleaning.
   * - ``AllKNN``
     - Under-sampling
     - Removes majority samples whose *k* nearest neighbours do not all share the same class.
   * - ``CNN``
     - Under-sampling
     - Condensed Nearest Neighbour. Retains a minimal subset of majority samples that still correctly classify all minority samples.
   * - ``RENN``
     - Under-sampling
     - Repeated Edited Nearest Neighbours. Iterates ENN until no majority samples are removed.

.. note::

   For all neighbour-based algorithms, SLIC automatically reduces ``k_neighbors`` / ``n_neighbors``
   to at most ``minority_class_size - 1`` to prevent fitting errors when the minority class is very small.
   If resampling still fails for any reason, the module exits with an error and a descriptive message.

Outputs
-------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - File
     - Description
   * - ``barplot.train.pdf``
     - Bar chart of the original class distribution.
   * - ``barplot.<algorithm>.pdf``
     - Bar chart of the class distribution after resampling.
   * - ``train.sampled.tsv``
     - Resampled feature matrix with the label column appended.

Example
-------

.. code-block:: bash

   # Balance classes with SMOTE
   PYRAMID SLIC \
       -i results/DaTrax/train.transformed.tsv \
       -m data/metadata.tsv \
       -l OS.status \
       -o results/SLIC \
       -a SMOTE \
       --seed 42

   # Under-sample with AllKNN using a custom ratio
   PYRAMID SLIC \
       -i results/DaTrax/train.transformed.tsv \
       -m data/metadata.tsv \
       -l OS.status \
       -o results/SLIC_knn \
       -a AllKNN \
       --sampling-strategy 0.8 \
       --threads 8