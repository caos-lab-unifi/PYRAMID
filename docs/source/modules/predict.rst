.. _predict:

PREDICT — Prediction on Test Data
===================================

**PREDICT** loads the ``.sav`` models saved by :ref:`hypertune`, applies them to a held-out test set, computes a comprehensive set of classification metrics, and optionally runs a permutation test to assess statistical significance.

Usage
-----

.. code-block:: bash

   PYRAMID PREDICT -i TRAIN.tsv -v TEST.tsv -f FS.tsv \
       -o OUTPUT_DIR -l LABEL -d HYPERTUNE_DIR \
       -mv METADATA_VALIDATION.tsv [options]

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
     - Training feature matrix (used for permutation baseline and re-fitting if needed).
   * - ``-v`` / ``--validation``
     - ``TSV``
     - Transformed test feature matrix.
   * - ``-f`` / ``--feature-selection``
     - ``TSV``
     - Feature selection results (same file used in :ref:`hypertune`).
   * - ``-o`` / ``--output``
     - ``DIR``
     - Output directory.
   * - ``-l`` / ``--label``
     - ``str``
     - Class label column name.
   * - ``-d`` / ``--dir``
     - ``DIR``
     - Directory containing the ``.sav`` models produced by :ref:`hypertune`.
   * - ``-mv`` / ``--metadata_validation``
     - ``TSV``
     - Metadata TSV for the test set (must contain the label column).

Optional arguments
------------------

.. list-table::
   :widths: 30 15 15 40
   :header-rows: 1

   * - Flag
     - Metavar
     - Default
     - Description
   * - ``-mt`` / ``--metadata_training``
     - ``TSV``
     - —
     - Metadata TSV for the training set (required when ``--permutation`` is used).
   * - ``--additional_training``
     - ``TSV``
     - —
     - Extra covariate TSV for training.
   * - ``--additional_validation``
     - ``TSV``
     - —
     - Extra covariate TSV for validation.
   * - ``--permutation``
     - flag
     - ``False``
     - Run a permutation test for each classifier.
   * - ``--n_perm``
     - ``int``
     - ``100``
     - Number of permutations. **Required** when ``--permutation`` is set.
   * - ``--seed``
     - ``int``
     - ``None``
     - Random seed.
   * - ``--n_splits``
     - ``int``
     - ``3``
     - CV folds (used for permutation test CV).
   * - ``--n_repeats``
     - ``int``
     - ``10``
     - CV repeats (used for permutation test CV).
   * - ``--verbose``
     - flag
     - ``False``
     - Enable scikit-learn warnings.

Metrics computed
----------------

For each model, PREDICT computes and reports:

* Accuracy, Precision, Recall, F1 score
* ROC-AUC
* Matthews Correlation Coefficient (MCC)
* Cohen's Kappa

Permutation test
----------------

When ``--permutation`` is supplied, each classifier is additionally evaluated as follows:

1. The classifier is re-fitted on the training set with the **true** labels and its score is recorded.
2. The labels are randomly shuffled ``--n_perm`` times; the classifier is re-fitted and scored on each shuffle.
3. A p-value is computed as the proportion of permuted scores that equal or exceed the true score.

.. note::

   ``--n_perm`` is **required** when ``--permutation`` is set. Omitting it will cause PYRAMID to exit with an error.

Outputs
-------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Description
   * - ``metrics.tsv``
     - Per-model metrics table (one row per model).
   * - ``<clf>_roc_curve.pdf``
     - ROC curve vs no-skill baseline for each classifier.
   * - ``<clf>_pr_curve.pdf``
     - Precision-Recall curve vs no-skill baseline for each classifier.
   * - ``permutation_test.tsv``
     - Permutation test results (true score, permuted score distribution, p-value) — only written when ``--permutation`` is set.

Example
-------

.. code-block:: bash

   # Basic prediction
   PYRAMID PREDICT \
       -i results/DaTrax/train.transformed.tsv \
       -v results/DaTrax/test.transformed.tsv \
       -f results/FeatX/features.consensus.tsv \
       -o results/PREDICT \
       -l label \
       -d results/HyPerTune \
       -mv data/metadata_test.tsv

   # With permutation test (100 shuffles)
   PYRAMID PREDICT \
       -i results/DaTrax/train.transformed.tsv \
       -v results/DaTrax/test.transformed.tsv \
       -f results/FeatX/features.consensus.tsv \
       -o results/PREDICT_perm \
       -l label \
       -d results/HyPerTune \
       -mv data/metadata_test.tsv \
       -mt data/metadata_train.tsv \
       --permutation \
       --n_perm 500 \
       --seed 42
