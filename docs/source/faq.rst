.. _faq:

FAQ & Troubleshooting
=====================

General
-------

**The output folder already exists. PYRAMID refuses to run.**

PYRAMID requires the output directory to be **empty** (or non-existent). This prevents accidental overwriting of results. Either pass a new path with ``-o`` or remove the existing directory:

.. code-block:: bash

   rm -rf results/FS && PYRAMID FS ...

**Module names seem to be case-sensitive.**

They are not. PYRAMID normalises the submodule name before dispatching, so ``PYRAMID surv``, ``PYRAMID Surv``, and ``PYRAMID SURV`` are all equivalent.

**I get a ``ptid column not found`` warning and the run continues. Is that OK?**

Yes. PYRAMID first attempts to load the TSV with ``index_col='ptid'``. If that fails, it falls back to ``index_col=None``. The warning is purely informational; execution proceeds normally as long as the TSV is otherwise valid.

Data preparation
----------------

**What should the input TSV look like?**

.. code-block:: text

   ptid    feature_1    feature_2    ...
   P001    0.12         3.45         ...
   P002    1.78         2.01         ...

* Tab-separated, first column named ``ptid``.
* All feature columns must be numeric.
* Columns containing ``extraction_ID`` or ``diagnostics`` in their names are automatically dropped.

**My feature matrix has NaN values. What happens?**

NaN rows are dropped with ``dropna()`` on load. Ensure that the number of remaining samples is sufficient for the analysis you intend to run.

**Can I add clinical covariates alongside radiomic features?**

Yes. Use ``--additional_training`` (and ``--additional_validation`` for the test set) to provide a second TSV that is inner-joined on ``ptid`` before any processing begins.

DaTrax
------

**Which transformation should I choose?**

As a starting point, ``scaler`` (Z-score) is appropriate for most ML algorithms. If your features are highly skewed, ``yeo-johnson`` or ``quantile-normal`` may improve performance. Use ``robust`` when the dataset contains many outliers. ``mixture`` combines standardisation and normalisation for situations where both zero-mean and bounded range are desired.

**Why are some features dropped after transformation?**

When a validation set is provided, only the columns present in **both** matrices after transformation are kept. This avoids feature mismatch at prediction time.

SLIC
----

**SMOTE fails with a ``ValueError`` about sample size.**

This usually means the minority class has very few samples (< ``k_neighbors`` + 1). Try reducing ``--neighbors`` or switching to ``ADASYN`` with a small ``--sampling-strategy``. For extremely small classes, ``CNN`` or ``RENN`` (under-sampling) may be more appropriate.

FS / FeatX
----------

**RFECV takes a very long time.**

RFECV is the most computationally intensive step. Reduce ``--n_repeats`` or ``--n_splits``, or increase ``--threads``. Using ``FeatX`` with a restricted algorithm list (e.g. ``-a ANOVA RFECV_LR``) instead of ``all`` is also much faster.

**FeatX returns zero features.**

Increase ``--rank``, decrease ``--supporting-algorithm``, or lower ``--threshold``. The consensus filter is conservative by design; relaxing any of these parameters broadens the eligible feature pool.

HyPerTune
---------

**No models are saved after HyPerTune.**

All models scored below ``--threshold``. Lower the threshold (e.g. ``--threshold 0.7``) or broaden the parameter grid in the JSON file.

**What is the format for the JSON parameter file?**

See :ref:`hypertune_json` for a complete example. Keys must match the classifier names used internally by HyPerTune. Values are lists of candidate values for ``GridSearchCV``, or lists / distribution dicts for ``RandomizedSearchCV``.

PREDICT
-------

**``--permutation`` requires ``--n_perm``.**

If you set ``--permutation`` you must also provide ``--n_perm``. PYRAMID will exit immediately with an error if you forget.

ENSEMBLE
--------

**The ``prob`` column is missing and ENSEMBLE exits immediately.**

The ``prob`` column is mandatory — it contains the comma-separated posterior probabilities produced
by each pipeline in :ref:`predict`. Make sure you are passing the correct ``metrics.tsv`` output
from PREDICT, which includes this column, rather than a manually assembled file.

**The ensemble ROC-AUC is lower than the best individual pipeline.**

This can happen when the pipelines in the input TSV have very heterogeneous performance. Because
ENSEMBLE averages all pipelines with equal weight, poorly performing ones can dilute the signal
from the best ones. Consider filtering the input TSV to include only pipelines above a minimum
ROC-AUC threshold before passing it to ENSEMBLE.

The test set must contain at least one event (``OS.status == 1``) to compute time-dependent AUC. Check that your metadata split is not accidentally placing all events in the training set.

**Metric values look inverted (below 0.5).**

Some models predict *protective* risk scores (higher score = longer survival) rather than *hazard* scores. SURV automatically flips values below 0.5 to ``1 - value`` in the summary table. This is expected and correct.

**The VIF cleaning step runs even though I did not request it.**

VIF cleaning is a **fallback** that runs only when the initial CoxPH fit on Spearman-filtered features fails (typically due to multicollinearity that Spearman filtering did not fully resolve). It is not user-configurable.