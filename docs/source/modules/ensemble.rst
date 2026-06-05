.. _ensemble:

ENSEMBLE — Model Ensembling
============================

**ENSEMBLE** combines predictions from the top-performing pipelines identified by :ref:`predict` into a single, more robust prediction. The combination strategy is **averaging of posterior probabilities** across all selected pipelines. An optimal decision threshold is then derived from the ROC curve (the point geometrically closest to the top-left corner) and applied to the averaged probability vector to produce the final binary labels.

Usage
-----

.. code-block:: bash

   PYRAMID ENSEMBLE -i RESULTS.tsv -o OUTPUT_DIR \
       -mv METADATA_VALIDATION.tsv -l LABEL -v TEST.tsv

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
     - TSV of top-performing pipelines produced by :ref:`predict` (see input format below).
   * - ``-o`` / ``--output``
     - ``DIR``
     - Output directory.
   * - ``-mv`` / ``--metadata_validation``
     - ``TSV``
     - Metadata TSV for the test set (must contain the label column, indexed by ``ptid``).
   * - ``-l`` / ``--label``
     - ``str``
     - Class label column name.
   * - ``-v`` / ``--validation``
     - ``TSV``
     - Transformed test feature matrix (``ptid`` index column required).

Input TSV format
----------------

The input TSV (``-i``) must contain one row per pipeline and at minimum the following columns:

.. code-block:: text

   pipeline         transformation  sampling    FS        ML                    roc_auc  f1_prediction  accuracy  pr    re    AP    prob
   standard_None_FeatX_RF    standard        None        FeatX     RandomForest          0.91     0.87           0.88      0.86  0.88  0.85  0.12,0.93,0.88,0.07,...
   minmax_SMOTE_FS_LR        minmax          SMOTE       FS        LogisticRegression    0.88     0.84           0.85      0.83  0.86  0.82  0.08,0.91,0.79,0.11,...
   robust_None_FeatX_SVC     robust          None        FeatX     SVC                   0.85     0.81           0.83      0.80  0.82  0.79  0.15,0.88,0.72,0.09,...

Key columns:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Column
     - Description
   * - ``transformation``
     - Feature transformation applied (e.g. ``standard``, ``minmax``, ``robust``).
   * - ``sampling``
     - Resampling strategy applied (e.g. ``SMOTE``, ``None``).
   * - ``FS``
     - Feature selection method used (e.g. ``FeatX``, ``FS``).
   * - ``ML``
     - Classifier name (e.g. ``RandomForest``, ``LogisticRegression``).
   * - ``roc_auc``
     - ROC-AUC score used to rank pipelines. This is the primary score column.
   * - ``f1_prediction``, ``accuracy``, ``pr``, ``re``, ``AP``
     - Additional performance metrics stored for comparison.
   * - ``prob``
     - **Required**. Comma-separated posterior probabilities (one value per test sample) produced by the pipeline. ENSEMBLE parses this column to build the averaged probability vector.

.. note::

   The ``prob`` column is mandatory. If it is absent the module exits with an error.
   The score column defaults to ``roc_auc``; if not found, ENSEMBLE falls back to
   ``f1_prediction``, ``accuracy``, or ``AP`` in that order.

How ensembling works
--------------------

1. **Pipeline loading**: ENSEMBLE reads the input TSV and parses the ``prob`` column into per-sample posterior probability arrays. Rows with missing or unparseable probabilities are discarded with a warning.
2. **Probability averaging**: posterior probability arrays from all selected pipelines are averaged element-wise across samples, producing a single ensemble probability vector.
3. **ROC threshold**: the optimal decision threshold is computed from the ROC curve (minimising the Euclidean distance to the top-left corner) and applied to the averaged probability vector to produce final binary predictions.
4. **Evaluation**: ensemble performance (accuracy, precision, recall, F1, AP, ROC-AUC) is computed and compared against the median performance of the individual top pipelines. A summary table and comprehensive visualisation are produced.

.. note::

   The single combination strategy is probability averaging followed by ROC-optimal thresholding.

Outputs
-------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Description
   * - ``table_pred.tsv``
     - Per-sample true label and binary ensemble prediction at the ROC-optimal threshold.
   * - ``ensemble_analysis.pdf``
     - Comprehensive visualisation including score distributions, ML model type breakdown, feature selection method breakdown, ensemble prediction distribution, individual vs ensemble probability traces, top model score ranking, ROC curve, Precision-Recall curve, and predicted vs true label scatter plot.
   * - ``summary_table.tsv``
     - Side-by-side comparison of ensemble metrics vs the median of the individual top pipelines (accuracy, precision, recall, F1, ROC-AUC, AP).

Example
-------

.. code-block:: bash

   PYRAMID ENSEMBLE \
       -i results/PREDICT/metrics.tsv \
       -o results/ENSEMBLE \
       -mv data/metadata_test.tsv \
       -l label \
       -v results/DaTrax/test.transformed.tsv