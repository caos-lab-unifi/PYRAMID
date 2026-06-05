.. _hypertune:

HyPerTune — Hyperparameter Tuning
==================================

**HyPerTune** takes a feature-selection result (from :ref:`fs` or :ref:`featx`) and a JSON file of hyperparameter grids, runs ``GridSearchCV`` or ``RandomizedSearchCV`` for each classifier / feature-set combination, and saves every model that exceeds a performance threshold.

Usage
-----

.. code-block:: bash

   PYRAMID HyPerTune -i TRAIN.tsv -f FS.tsv -l LABEL -j PARAMS.json \
       -o OUTPUT_DIR -s {GridSearchCV|RandomizedSearchCV} [options]

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
     - Training feature matrix.
   * - ``-f`` / ``--feature-selection``
     - ``TSV``
     - Feature selection results file (output of :ref:`fs` or :ref:`featx`).
   * - ``-l`` / ``--label``
     - ``str``
     - Class label column name.
   * - ``-j`` / ``--json``
     - ``JSON``
     - Parameter grid file. See :ref:`hypertune_json`.
   * - ``-o`` / ``--output``
     - ``DIR``
     - Output directory.
   * - ``-s`` / ``--search``
     - ``str``
     - Search strategy: ``GridSearchCV`` or ``RandomizedSearchCV``.

Optional arguments
------------------

.. list-table::
   :widths: 30 15 15 40
   :header-rows: 1

   * - Flag
     - Metavar
     - Default
     - Description
   * - ``--threshold``
     - ``float``
     - ``0.9``
     - Minimum cross-validated score for a model to be saved to disk.
   * - ``-m`` / ``--metric``
     - ``str``
     - ``f1``
     - Scoring metric. Choices: ``f1``, ``precision``, ``recall``, ``roc_auc``.
   * - ``-t`` / ``--threads``
     - ``int``
     - ``10``
     - Parallel threads for the search.
   * - ``--metadata``
     - ``TSV``
     - —
     - Separate metadata TSV.
   * - ``--additional``
     - ``TSV``
     - —
     - Extra covariate TSV merged before tuning.
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
     - ``10``
     - CV repeats.
   * - ``-v`` / ``--verbose``
     - flag
     - ``False``
     - Enable scikit-learn warnings.

.. _hypertune_json:

Parameter grid JSON format
--------------------------

The JSON file specifies a hyperparameter grid for each classifier. Keys must match the estimator identifiers used internally by HyPerTune. Example:

.. code-block:: json

   {
     "SVM": {
       "C":     [0.01, 0.1, 1, 10, 100],
       "gamma": ["scale", "auto"]
     },
     "RF": {
       "n_estimators": [50, 100, 200],
       "max_depth":    [null, 5, 10]
     },
     "LR": {
       "C": [0.01, 0.1, 1, 10]
     },
     "GB": {
       "n_estimators":   [50, 100],
       "learning_rate":  [0.05, 0.1, 0.2],
       "max_depth":      [3, 5]
     }
   }

For ``RandomizedSearchCV``, values can also be ``scipy.stats`` distribution strings (when calling the API directly); plain lists are always valid for both strategies.

Search strategies
-----------------

``GridSearchCV``
   Exhaustively evaluates every combination of hyperparameters in the grid. Use for small grids or when full coverage is required.

``RandomizedSearchCV``
   Samples a fixed number of parameter combinations at random. Faster for large grids. Produces a plot of all sampled iterations sorted by test score, with a vertical marker at the best iteration (``<estimator>_<features>_random.pdf``).

Outputs
-------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Description
   * - ``<estimator>_<features>.sav``
     - Serialised best model (``joblib`` format) for each classifier / feature-set pair that exceeds ``--threshold``.
   * - ``<estimator>_<features>_grid.pdf``
     - Per-hyperparameter score sweep plot (Grid search).
   * - ``<estimator>_<features>_random.pdf``
     - Iteration score plot (Random search).
   * - ``results.tsv``
     - Summary table: estimator, feature set, best CV score, best parameters.

Example
-------

.. code-block:: bash

   # Grid search, save models with F1 ≥ 0.85
   PYRAMID HyPerTune \
       -i results/DaTrax/train.transformed.tsv \
       -f results/FeatX/features.consensus.tsv \
       -l label \
       -j params.json \
       -o results/HyPerTune \
       -s GridSearchCV \
       --threshold 0.85 \
       --metric f1 \
       --seed 42

   # Randomised search
   PYRAMID HyPerTune \
       -i results/DaTrax/train.transformed.tsv \
       -f results/FeatX/features.consensus.tsv \
       -l label \
       -j params.json \
       -o results/HyPerTune_rand \
       -s RandomizedSearchCV \
       --seed 42 \
       -t 16
