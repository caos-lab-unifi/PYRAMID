.. _outputs:

Output Reference
================

This page gives a bird's-eye view of all files produced across the PYRAMID pipeline, grouped by submodule.

DaTrax
------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Description
   * - ``train.transformed.tsv``
     - Transformed training feature matrix.
   * - ``test.transformed.tsv``
     - Transformed test feature matrix (only when ``-v`` is provided).

SLIC
----

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Description
   * - ``barplot.train.pdf``
     - Class distribution before resampling.
   * - ``barplot.<algorithm>.pdf``
     - Class distribution after resampling.
   * - ``train.sampled.tsv``
     - Resampled feature matrix with label column.

FS
--

**Preprocessing outputs** (always produced before any algorithm runs):

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Description
   * - ``Correlation_matrix.pdf``
     - Spearman correlation heatmap of all input features, computed before filtering.
   * - ``removed.spearman.txt``
     - Plain-text list of features removed because their pairwise Spearman :math:`r` ≥ 0.9 with another feature.
   * - ``train.sampled.filtered.tsv``
     - Feature matrix after Spearman filtering, with the label column appended. This is the input to all downstream algorithms.

**Algorithm outputs**:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File pattern
     - Description
   * - ``<algorithm>.features.tsv``
     - Ranked feature list per algorithm.
   * - ``FS.results.tsv``
     - Combined rank matrix: one rank column per algorithm, one row per feature surviving at least one algorithm.
   * - ``<estimator>.total.pdf``
     - Full RFECV score curve.
   * - ``<estimator>.optimal.pdf``
     - RFECV score curve (optimal region).
   * - ``violin.<algorithm>.pdf``
     - Violin plot of metric by parameter.
   * - ``heatmap_<n>.pdf``
     - Euclidean clustermap.
   * - ``coefficients.<reg>.pdf``
     - Coefficient path for regularised models.
   * - ``grid_results.<reg>.pdf``
     - GridSearchCV alpha plot.
   * - ``nonzero.<reg>.pdf``
     - Non-zero coefficient bar chart.
   * - ``AgglomerativeClustering.top.pdf``
     - Top features per cluster.
   * - ``AgglomerativeClustering.heatmap.distribution.pdf``
     - Per-cluster mean heatmap.

FeatX
-----

All FS preprocessing and algorithm outputs above, plus the following consensus outputs organised under ``suppN/`` subdirectories (one per supporting-algorithm count swept):

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Description
   * - ``suppN/heatmap_{parameter}.pdf``
     - Euclidean clustermap of the features selected at rank threshold ``parameter``. One file per rank threshold tested, saved alongside the corresponding ``FSS_dataframe_{parameter}.tsv``.
   * - ``suppN/FSS_dataframe_{parameter}.tsv``
     - Feature × algorithm rank matrix for features selected at rank threshold ``parameter``. One file per rank threshold tested. Missing values filled with ``9999``.
   * - ``suppN/FSS_dataframe.tsv``
     - Concatenation of all ``FSS_dataframe_{parameter}.tsv`` files for this ``suppN`` level.
   * - ``suppN/results.parameters.tsv``
     - Cross-validated scores (``accuracy``, ``precision``, ``recall``, ``f1``, ``roc_auc``) for every classifier at every rank threshold, plus ``parameter`` and ``n_features`` columns.
   * - ``suppN/results.parameters.selected.tsv``
     - Feature × algorithm rank matrix for the best ``(supp, parameter)`` combination. Final recommended feature set for this supporting-algorithm count.
   * - ``FS.results.tsv``
     - Root-level summary: winning feature set across all ``suppN`` levels, with one ``supporting_algorithm_N`` column per supp level where the feature was selected.

HyPerTune
---------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Description
   * - ``<estimator>_<features>.sav``
     - Serialised best model (one per passing estimator / feature-set pair).
   * - ``<estimator>_<features>_grid.pdf``
     - Score sweep plot (Grid search).
   * - ``<estimator>_<features>_random.pdf``
     - Iteration score plot (Random search).
   * - ``results.tsv``
     - Summary table of all tuned models.

PREDICT
-------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Description
   * - ``metrics.tsv``
     - Per-model performance metrics.
   * - ``<clf>_roc_curve.pdf``
     - ROC curve with no-skill baseline.
   * - ``<clf>_pr_curve.pdf``
     - Precision-Recall curve.
   * - ``permutation_test.tsv``
     - Permutation test results (only with ``--permutation``).

ENSEMBLE
--------

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

SURV
----

See :ref:`surv` for the full table. Key outputs:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Description
   * - ``model.metrics.tsv``
     - Combined metrics (AUC, C-Harrell, C-Uno) for all models.
   * - ``univariate.analysis.tsv``
     - Per-feature univariate CoxPH results.
   * - ``df_HR.<label>.txt``
     - Hazard ratios for each model variant.
   * - ``<estimator>_<features>.sav``
     - Serialised best ensemble survival models.
   * - ``km_binary_aa.<label>.pdf``
     - Kaplan-Meier plot by archetype group.
   * - ``km_intensity_aa.<label>.pdf``
     - Kaplan-Meier plot by risk intensity.