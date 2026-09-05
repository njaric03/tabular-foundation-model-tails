# Seven rules for running the measurements

Each one has been broken once already and cost measurements.

1. One process writes one output file. Every parallel stream gets its own `OUTPUT=`.
2. Every adjustable parameter is a column in the CSV and part of the resume key. `N_EST`
   moves the captured shape share for TabPFN from 19% to 42%; when it is not in the key,
   a re-run with a different value silently skips the work. `common/append.py` now
   refuses to resume when a key column is missing rather than skipping quietly.
3. A result enters the repo only when a finding cites it or a script reads it. Everything
   else stays in `archive/`.
4. The same seed for every model in the same comparison, and `random_state=seed` for all
   of them. This lives in one place, `common/models.py`.
5. Warnings are filtered by category, never blanket. `filterwarnings("ignore")` at the
   top of every script hid a divide-by-zero in the tail-index inversion for months.
   `common/quiet.py` silences package noise and leaves `RuntimeWarning` visible.
6. A test may not be counted more often than its free unit varies. Three seeds on one
   dataset are not three independent comparisons; `common/stats.py` reports the
   cluster-level p-value next to the row-level one, and the text quotes the first.
   The unit is the thing, not its label: `218_house_8L` and `house_16H` are one target
   vector under two feature sets, so nine dataset names were eight free units and
   p = 0.002 was p = 0.0039. Cluster with `stats.paired_by_target`, which groups by the
   recorded `y_sha1` rather than by name.

7. A model is written to the CSV under its canonical name, never under the spelling the
   caller used. `MODELS=XGBoost` and `MODELS=XGB` produced two labels for one control and
   split every `groupby("model")` that touched both files. `models.parse_list` normalises
   at the point the environment is read, and `tests/test_identity.py` fails on any name in
   `results/` that no script can reproduce.
