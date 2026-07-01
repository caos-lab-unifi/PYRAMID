.. _featx:

FeatX — Consensus Feature Selection
=====================================

**FeatX** runs the same suite of feature selection algorithms as :ref:`fs` but adds a **consensus ranking** step: it sweeps combinations of supporting-algorithm counts and rank thresholds, evaluates each candidate feature subset with cross-validated classification across all metrics, and returns the subset that maximises the chosen metric while respecting a user-defined feature budget.

FeatX is the recommended path when you want a single, reproducible feature set as input to :ref:`hypertune`.

As in :ref:`fs`, a mandatory **Spearman correlation pre-filtering** step runs before any algorithm, removing redundant features and writing a filtered dataset. All algorithm-specific and consensus outputs are produced on this filtered feature set.

Usage
-----

.. code-block:: bash

   PYRAMID FeatX -i TRAIN.tsv -o OUTPUT_DIR -l LABEL [options]

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
     - Name of the class label column.

Optional arguments
------------------

.. list-table::
   :widths: 35 15 15 35
   :header-rows: 1

   * - Flag
     - Metavar
     - Default
     - Description
   * - ``-a`` / ``--algorithms``
     - ``str [...]``
     - ``all``
     - Feature selection algorithms to include in the consensus. Same choices as :ref:`fs_algorithms`.
   * - ``-t`` / ``--threads``
     - ``int``
     - ``10``
     - Parallel threads for RFECV and embedded methods.
   * - ``-m`` / ``--metric``
     - ``str``
     - ``f1``
     - Primary scoring metric used to select the best feature subset. Choices: ``accuracy``, ``f1``, ``precision``, ``recall``, ``roc_auc``. All metrics are computed and saved regardless of this choice.
   * - ``-n`` / ``--n-max-allowed-features``
     - ``int``
     - ``20``
     - Hard upper bound on the number of features in the final set.
   * - ``-r`` / ``--rank``
     - ``int``
     - ``20``
     - Maximum rank threshold swept by ``FindNfeatures``. Features must appear within the top ``--rank`` positions in at least ``--supporting-algorithm`` lists to be eligible. Swept from 2 up to this value.
   * - ``-s`` / ``--supporting-algorithm``
     - ``int``
     - ``3``
     - Minimum number of algorithms that must have ranked a feature within ``--rank`` for it to be eligible. Swept from 2 up to the total number of algorithms run.
   * - ``--threshold``
     - ``float``
     - ``0.6``
     - Minimum cross-validated score (for the primary metric) a candidate subset must achieve to be considered.
   * - ``--metadata``
     - ``TSV``
     - —
     - Separate metadata TSV merged before selection.
   * - ``--additional``
     - ``TSV``
     - —
     - Extra covariate TSV merged before selection.
   * - ``--seed``
     - ``int``
     - ``None``
     - Random seed for reproducibility of cross-validation splits.
   * - ``--n_splits``
     - ``int``
     - ``3``
     - Number of folds in repeated stratified k-fold cross-validation.
   * - ``--n_repeats``
     - ``int``
     - ``10``
     - Number of repetitions in repeated stratified k-fold cross-validation.
   * - ``-v`` / ``--verbose``
     - flag
     - ``False``
     - Enable scikit-learn warnings.

How FeatX works
---------------

**Step 1 — Spearman pre-filtering** (mandatory, identical to :ref:`fs`):

1. A pairwise Spearman correlation matrix is computed and saved as ``Correlation_matrix.pdf``.
2. Features whose pairwise Spearman :math:`r` ≥ 0.9 are removed; the removed names are written to ``removed.spearman.txt``.
3. The filtered matrix is written to ``train.sampled.filtered.tsv``. All downstream steps operate on this.

**Step 2 — Per-algorithm ranking** (identical to :ref:`fs`):

Each requested algorithm produces its own ranked feature list on the filtered matrix.

**Step 3 — Consensus sweep** (``FindNfeatures``):

1. All combinations of *supporting-algorithm count* ``supp`` (from 2 up to the total number of algorithms) and *rank threshold* ``parameter`` (from 2 up to ``--rank``) are tested.
2. For every ``(supp, parameter)`` combination, eligible features are those appearing within the top ``parameter`` positions in at least ``supp`` algorithm lists.

   *Example*: with 5 algorithms, ``--supporting-algorithm 3``, and ``--rank 10``, a feature must
   appear in the top-10 positions of at least 3 out of 5 algorithm-ranked lists to be eligible.
   Raising ``--supporting-algorithm`` to 4 produces a stricter, smaller feature set.

3. Each candidate subset is evaluated with repeated stratified k-fold CV across **all** metrics (``accuracy``, ``precision``, ``recall``, ``f1``, ``roc_auc``).
4. Among subsets whose primary metric exceeds ``--threshold`` and whose size does not exceed ``--n-max-allowed-features``, the one whose top-quartile score distribution dominates is selected — i.e. the most parsimonious feature set with the best cross-validated performance.
5. The final feature set is written to disk.

.. tip::

   Increasing ``--supporting-algorithm`` produces a more conservative, smaller feature set.
   Decreasing ``--rank`` restricts eligibility to features ranked very highly by each algorithm.

.. note::

   Consensus outputs are organised in subdirectories ``suppN/`` (one per supporting-algorithm count swept).
   Within each subdirectory, one file is produced per rank threshold value tested (``FSS_dataframe_{parameter}.tsv``),
   allowing inspection of how the feature set grows as the rank threshold is relaxed.

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

**Per-algorithm outputs**: identical to :ref:`fs` — see the FS Outputs table.

**Consensus outputs**, organised under ``OUTPUT_DIR/suppN/`` (one subdirectory per supporting-algorithm count swept):

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - File
     - Description
   * - ``suppN/heatmap_{parameter}.pdf``
     - Euclidean clustermap of the features selected at rank threshold ``parameter``. One file is produced per rank threshold value tested, saved alongside the corresponding ``FSS_dataframe_{parameter}.tsv``.
   * - ``suppN/FSS_dataframe_{parameter}.tsv``
     - Feature × algorithm rank matrix for features selected at rank threshold ``parameter``. Rows are features agreed upon by at least ``N`` algorithms within the top ``parameter`` ranks. Columns are the rank assigned by each algorithm (missing values filled with ``9999``). One file is produced per rank threshold value tested.
   * - ``suppN/FSS_dataframe.tsv``
     - Vertical concatenation of all ``FSS_dataframe_{parameter}.tsv`` files for this ``suppN`` level. Provides a complete view of how the feature set changes across all rank thresholds tested.
   * - ``suppN/results.parameters.tsv``
     - Cross-validated metric scores (``accuracy``, ``precision``, ``recall``, ``f1``, ``roc_auc``) for every classifier at every rank threshold tested. Also records ``parameter`` (rank threshold) and ``n_features`` (size of the feature subset). This table feeds the violin plot ``violin.pdf``.
   * - ``suppN/results.parameters.selected.tsv``
     - Feature × algorithm rank matrix for the single best ``(supp, parameter)`` combination selected by ``FilterParameter``. This is the final recommended feature set for this supporting-algorithm count.
   * - ``FS.results.tsv``
     - Root-level summary: the winning feature set across all ``suppN`` levels, with one ``supporting_algorithm_N`` column per supp level where the feature was selected. Ready to pass to :ref:`hypertune`.

Example
-------

.. code-block:: bash

   # Consensus from all algorithms, max 10 features, threshold 0.65
   PYRAMID FeatX \
       -i results/DaTrax/train.transformed.tsv \
       -o results/FeatX \
       -l label \
       -n 10 \
       --threshold 0.65 \
       --seed 42

   # Consensus from ANOVA and RFECV methods only
   PYRAMID FeatX \
       -i results/DaTrax/train.transformed.tsv \
       -o results/FeatX_subset \
       -l label \
       -a ANOVA RFECV_LR RFECV_RF \
       -s 2 \
       -r 15