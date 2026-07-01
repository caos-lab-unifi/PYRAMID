.. _fs:

FS — Feature Selection
=======================

**FS** runs one or more feature selection algorithms on the (transformed, optionally resampled) training dataset and outputs ranked feature lists that feed into :ref:`featx` or directly into :ref:`hypertune`.

Before any algorithm is applied, FS performs a mandatory **Spearman correlation pre-filtering** step that removes redundant features and writes a filtered dataset. All algorithm-specific outputs are then produced on this filtered feature set.

Usage
-----

.. code-block:: bash

   PYRAMID FS -i TRAIN.tsv -o OUTPUT_DIR -l LABEL -a ALG1 [ALG2 ...] [options]

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
   * - ``-o`` / ``--output``
     - ``DIR``
     - Output directory.
   * - ``-l`` / ``--label``
     - ``str``
     - Name of the label column (must be present in the input TSV or the metadata file).
   * - ``-a`` / ``--algorithms``
     - ``str [...]``
     - One or more feature selection algorithms, or ``all``. See :ref:`fs_algorithms`.

Optional arguments
------------------

.. list-table::
   :widths: 30 15 15 40
   :header-rows: 1

   * - Flag
     - Metavar
     - Default
     - Description
   * - ``--metadata``
     - ``TSV``
     - —
     - Separate metadata TSV (when the label is not in the feature matrix).
   * - ``--additional``
     - ``TSV``
     - —
     - Extra covariate TSV merged with the training matrix before selection.
   * - ``-t`` / ``--threads``
     - ``int``
     - ``10``
     - Parallel threads for RFECV and embedded methods.
   * - ``-m`` / ``--metric``
     - ``str``
     - ``f1``
     - Scoring metric used by RFECV and regularised estimators. Choices: ``f1``, ``precision``, ``recall``, ``roc_auc``.
   * - ``--seed``
     - ``int``
     - ``None``
     - Random seed for cross-validation.
   * - ``--n_splits``
     - ``int``
     - ``3``
     - Number of folds in the repeated stratified k-fold CV.
   * - ``--n_repeats``
     - ``int``
     - ``10``
     - Number of repeats in the repeated stratified k-fold CV.
   * - ``-v`` / ``--verbose``
     - flag
     - ``False``
     - Print scikit-learn convergence warnings.

Preprocessing — Spearman correlation filter
--------------------------------------------

Before any feature selection algorithm runs, FS performs a mandatory correlation pre-filtering step:

1. A pairwise Spearman correlation matrix is computed across all input features and saved as ``Correlation_matrix.pdf``.
2. Features whose pairwise Spearman :math:`r` ≥ 0.9 are identified; one representative per correlated group is retained and the redundant ones are removed. The removed feature names are written to ``removed.spearman.txt``.
3. The filtered feature matrix (with label column appended) is written to ``train.sampled.filtered.tsv``. All downstream feature selection algorithms operate on this filtered matrix.

.. note::

   This step is mandatory and always runs, regardless of which algorithms are selected.

.. _fs_algorithms:

Algorithms
----------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - ``ANOVA``
     - Univariate ANOVA F-test. Retains features above the 95th percentile of F-scores after variance thresholding.
   * - ``MI``
     - Mutual information. Retains features above the top 10th percentile of MI scores.
   * - ``RFECV_LR``
     - Recursive Feature Elimination with Cross-Validation using Logistic Regression (L2, default).
   * - ``RFECV_LR_L1``
     - RFECV with L1-penalised Logistic Regression.
   * - ``RFECV_LR_L2``
     - RFECV with L2-penalised Logistic Regression.
   * - ``RFECV_LR_EN``
     - RFECV with ElasticNet-penalised Logistic Regression.
   * - ``RFECV_Perceptron``
     - RFECV with a Perceptron classifier.
   * - ``RFECV_RF``
     - RFECV with a Random Forest classifier.
   * - ``RFECV_GB``
     - RFECV with a Gradient Boosting classifier.
   * - ``RFECV_SVM``
     - RFECV with a linear Support Vector Machine.
   * - ``Agglomerative``
     - Agglomerative clustering-based selection. Identifies the optimal number of clusters via silhouette analysis, then selects the top feature per cluster by Cohen's d effect size.
   * - ``Lasso``
     - Regularisation path with L1 penalty via ``GridSearchCV`` over log-spaced alpha values.
   * - ``ElasticNet``
     - Regularisation path with ElasticNet penalty.
   * - ``all``
     - Runs all algorithms listed above.

Cross-validation strategy
-------------------------

All RFECV-based algorithms and the regularised methods use a **Repeated Stratified K-Fold** cross-validator:

* ``--n_splits`` folds × ``--n_repeats`` repeats (default: 3 × 10 = 30 fits per alpha).
* Scoring is controlled by ``--metric`` (default ``f1``).

Outputs
-------

**Preprocessing outputs** (always produced, before any algorithm runs):

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

**Algorithm outputs** (one set per requested algorithm):

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File pattern
     - Description
   * - ``<algorithm>.features.tsv``
     - Ranked feature list produced by each algorithm.
   * - ``FS.results.tsv``
     - Combined rank matrix: one rank column per algorithm, one row per feature surviving at least one algorithm.
   * - ``<estimator>.total.pdf``
     - RFECV score curve (all feature counts).
   * - ``<estimator>.optimal.pdf``
     - RFECV score curve zoomed to the optimal region.
   * - ``violin.<algorithm>.pdf``
     - Violin plot of per-parameter metric distributions.
   * - ``heatmap_<n>.pdf``
     - Euclidean clustermap of selected features.
   * - ``coefficients.<reg>.pdf``
     - Coefficient path plot for Lasso / ElasticNet.
   * - ``grid_results.<reg>.pdf``
     - GridSearchCV concordance-index vs alpha plot.
   * - ``nonzero.<reg>.pdf``
     - Bar chart of non-zero coefficients at the best alpha.
   * - ``AgglomerativeClustering.top.pdf``
     - Top features per cluster (Cohen's d).
   * - ``AgglomerativeClustering.heatmap.distribution.pdf``
     - Per-cluster mean heatmap.

Example
-------

.. code-block:: bash

   # Run ANOVA and RFECV with Logistic Regression
   PYRAMID FS \
       -i results/DaTrax/train.transformed.tsv \
       -o results/FS \
       -l label \
       -a ANOVA RFECV_LR \
       --seed 42 \
       --n_splits 5 \
       --n_repeats 5

   # Run all available algorithms
   PYRAMID FS \
       -i results/DaTrax/train.transformed.tsv \
       -o results/FS_all \
       -l label \
       -a all \
       -t 16 \
       -m roc_auc