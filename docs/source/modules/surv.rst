.. _surv:

SURV — Survival Analysis
=========================

**SURV** performs a complete survival analysis pipeline on radiomics features, from Spearman-filtered univariate Cox regression through penalised Cox models (Lasso, ElasticNet) and ensemble survival learners (RSF, GBS, CWGB, SSVM). An optional Archetypal Analysis step stratifies patients into high-risk and low-risk groups.

Usage
-----

.. code-block:: bash

   PYRAMID SURV -i TRAIN.tsv --validation TEST.tsv -o OUTPUT_DIR \
       -mt METADATA_TRAIN.tsv -mv METADATA_TEST.tsv \
       -s {GridSearchCV|RandomizedSearchCV} -j PARAMS.json [options]

Required arguments
------------------

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Flag
     - Metavar
     - Description
   * - ``-i`` / ``--input``
     - ``TSV``
     - Training feature matrix (``ptid`` index column required).
   * - ``--validation``
     - ``TSV``
     - Test / validation feature matrix.
   * - ``-o`` / ``--output``
     - ``DIR``
     - Output directory.
   * - ``-mt`` / ``--metadata_training``
     - ``TSV``
     - Training metadata TSV. Must contain ``OS.status`` (0/1) and ``OS.time`` (float) columns, indexed by ``ptid``.
   * - ``-mv`` / ``--metadata_validation``
     - ``TSV``
     - Validation metadata TSV (same format as training metadata).
   * - ``-s`` / ``--search``
     - ``str``
     - Hyperparameter search strategy: ``GridSearchCV`` or ``RandomizedSearchCV``.
   * - ``-j`` / ``--json``
     - ``JSON``
     - JSON file with hyperparameter grids. See :ref:`surv_json`.

Optional arguments
------------------

.. list-table::
   :widths: 35 15 15 35
   :header-rows: 1

   * - Flag
     - Metavar
     - Default
     - Description
   * - ``-t`` / ``--transformation``
     - ``str``
     - —
     - Data transformation to apply before analysis (same choices as :ref:`DaTrax <datrax_transformations>`). When omitted, data is used as-is.
   * - ``--additional_training``
     - ``TSV``
     - —
     - Extra covariate TSV merged with the training matrix.
   * - ``--additional_validation``
     - ``TSV``
     - —
     - Extra covariate TSV merged with the validation matrix.
   * - ``-f`` / ``--feature-selection``
     - ``TSV``
     - —
     - Pre-computed feature selection result. When provided, SURV runs the full model pipeline on those feature sets in addition to its internal feature selection.
   * - ``--threads``
     - ``int``
     - ``10``
     - Parallel threads.
   * - ``--seed``
     - ``int``
     - ``None``
     - Random seed.
   * - ``--n_splits``
     - ``int``
     - ``3``
     - CV folds.
   * - ``--n_repeats``
     - ``int``
     - ``5``
     - CV repeats.
   * - ``--archetypes``
     - flag
     - ``False``
     - Enable Archetypal Analysis for patient risk stratification.
   * - ``--verbose``
     - flag
     - ``False``
     - Enable scikit-survival warnings.
   * - ``--robust-parameter``
     - ``VAL1,VAL2``
     - ``25.0,75.0``
     - IQR for robust scaler (only with ``-t robust``).
   * - ``--n-quantiles``
     - ``int``
     - ``100``
     - Number of quantiles (only with ``-t quantile-*``).

Metadata format
---------------

The metadata TSV must have a ``ptid`` index column and at minimum:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Column
     - Description
   * - ``OS.status``
     - Event indicator: ``1`` = event occurred (e.g. death), ``0`` = censored.
   * - ``OS.time``
     - Time to event or censoring (any consistent unit, e.g. months).

Internal workflow
-----------------

SURV runs the following steps automatically in sequence:

1. **Feature loading and cleaning**: drops non-numeric and diagnostic columns, optionally merges covariates.
2. **Optional transformation**: applies the requested scaler / transformer (same engine as :ref:`datrax`).
3. **Spearman correlation filter**: removes features whose pairwise Spearman :math:`r` ≥ 0.9, retaining one representative per correlated group.
4. **Univariate Cox analysis**: fits an independent CoxPH model for each surviving feature, recording the C-index and coefficient. Results saved to ``univariate.analysis.tsv``.
5. **VIF cleaning** (fallback): if the basic CoxPH fit on Spearman-filtered features fails, VIF-based multicollinearity removal is attempted (threshold = 5.0).
6. **Multivariate feature selection**: ``SelectKBest`` with per-feature CoxPH C-index scores, optimised via Grid or Random search.
7. **Penalised Cox models**: Lasso (``l1_ratio=1.0``) and ElasticNet (``l1_ratio=0.5``) via ``CoxnetSurvivalAnalysis``, with alpha grid search.
8. **Ensemble survival learners**: Random Survival Forest (RSF), Gradient Boosting Survival (GBS), Component-wise Gradient Boosting (CWGB), and Fast Survival SVM (SSVM) are each tuned with Grid/Random search across the Spearman, Lasso, and ElasticNet feature sets.
9. **Permutation feature importance**: for each fitted model, permutation importance is computed in parallel and the optimal feature count *k* is selected by evaluating C-index from top 2 to top *k* features.
10. **Summary metrics**: Harrell's C-index, Uno's C-index, and time-dependent AUC are combined into ``model.metrics.tsv``.
11. **Archetypal Analysis** (optional): extreme archetypes are derived from the top and bottom quartile of training risk scores. An optimal threshold is found by log-rank test sweep. Test-set patients are projected and stratified; Kaplan-Meier curves with log-rank p-values are produced.

.. _surv_json:

Parameter grid JSON format
--------------------------

.. code-block:: json

   {
     "parameters_RF": {
       "n_estimators": [50, 100, 200],
       "max_depth":    [3, 5, null],
       "min_samples_split": [5, 10]
     },
     "parameters_GB": {
       "n_estimators":   [50, 100],
       "learning_rate":  [0.05, 0.1],
       "max_depth":      [2, 3]
     },
     "parameters_CWGB": {
       "n_estimators":  [50, 100, 200],
       "learning_rate": [0.05, 0.1]
     },
     "parameters_SSVM": {
       "alpha": [0.5, 1.0, 5.0],
       "rank_ratio": [0.0, 0.5, 1.0]
     }
   }

Survival metrics
----------------

For each model, SURV computes:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Metric
     - Description
   * - ``mean_auc``
     - Time-dependent AUC averaged across the 5th–95th percentile time-points (``cumulative_dynamic_auc``).
   * - ``c_harrell``
     - Harrell's concordance index (``concordance_index_censored``).
   * - ``c_uno``
     - Uno's concordance index (``concordance_index_ipcw``), which is less sensitive to censoring distribution.

.. note::

   Metric values below 0.5 are automatically flipped to ``1 - value`` in the final summary table. This corrects for models whose predicted risk direction is reversed.

Outputs
-------

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - File
     - Description
   * - ``estimated_curve.pdf``
     - Kaplan-Meier estimated survival curve for the training set.
   * - ``univariate.analysis.tsv``
     - Per-feature CoxPH C-index and coefficient.
   * - ``train.spearman.tsv``
     - Training matrix after Spearman filtering.
   * - ``removed.spearman.txt``
     - List of features removed by the Spearman filter.
   * - ``train.vif_selected.tsv``
     - Training matrix after VIF cleaning (only when VIF step runs).
   * - ``multivariate.features.txt``
     - Features selected by multivariate CoxPH.
   * - ``features_non_zero.Lasso.txt``
     - Non-zero Lasso features at the best alpha.
   * - ``features_non_zero.ElasticNet.txt``
     - Non-zero ElasticNet features at the best alpha.
   * - ``df_HR.<label>.txt``
     - Hazard ratios (Feature, Coefficient, Hazard_Ratio) for each CoxPH variant.
   * - ``df_risk_scores.<label>.txt``
     - Per-patient risk scores from each CoxPH model.
   * - ``ROC.time.dependent.<label>.pdf``
     - Time-dependent AUC plot for each model.
   * - ``<estimator>_<features>.sav``
     - Serialised best ensemble models.
   * - ``survival_function.<model>.pdf``
     - Predicted survival functions for test-set patients.
   * - ``cumulative_hazard_function.<model>.pdf``
     - Predicted cumulative hazard functions for test-set patients.
   * - ``permutation_importance.<model>.pdf``
     - C-index vs number of top features (permutation importance).
   * - ``permutation_importance.<model>.txt``
     - Importances table (mean ± std) for the best feature subset.
   * - ``model.metrics.tsv``
     - Combined metrics table for all models.
   * - ``<label>.coefficients.pdf``
     - Coefficient path plot for Lasso / ElasticNet.
   * - ``<label>.grid.results.pdf``
     - Alpha search curve for penalised Cox models.
   * - ``km_binary_aa.<label>.pdf``
     - KM curve (High Risk vs Low Risk) from Archetypal Analysis.
   * - ``km_intensity_aa.<label>.pdf``
     - KM curve by intensity group from Archetypal Analysis.
   * - ``archetype_analysis_results.<label>.tsv``
     - Per-patient archetype weights and group assignments.

Example
-------

.. code-block:: bash

   # Full pipeline with transformation and archetypal analysis
   PYRAMID SURV \
       -i data/train.tsv \
       --validation data/test.tsv \
       -o results/SURV \
       -mt data/metadata_train.tsv \
       -mv data/metadata_test.tsv \
       -s GridSearchCV \
       -j params_surv.json \
       -t scaler \
       --archetypes \
       --seed 42 \
       --threads 16

   # Randomised search, no transformation
   PYRAMID SURV \
       -i data/train.tsv \
       --validation data/test.tsv \
       -o results/SURV_rand \
       -mt data/metadata_train.tsv \
       -mv data/metadata_test.tsv \
       -s RandomizedSearchCV \
       -j params_surv.json \
       --seed 0
