# Rules for running the measurements

Each of these was broken once and cost measurements.

1. **One process writes one output file.** Every parallel run gets its own `OUTPUT=`, and
   git counts as a second writer: a `checkout` or `reset` over a CSV that a run is
   appending to truncates it, and the run carries on appending to the stub. The wide
   leverage sweep lost three of freMTPL2sev's four bins that way. Commit a result file
   after its run has finished.
2. **Every parameter that changes a result is a column and part of the resume key.**
   Without it, a re-run with another value finds "already done" and measures nothing.
   `common/append.py` refuses to resume when a key column is missing, or present but empty
   in every row. When a knob becomes a column, old rows are backfilled only with a value
   `results/provenance.csv` proves. Scripts read knobs through `common.env`, and a test
   fails when one is missing from `provenance.ENV_KNOBS`.
3. **A result enters the repository only when a finding cites it or a script reads it.**
   Everything else stays in `archive/`.
4. **Every model in a comparison gets the same seed as `random_state`.** Models are built
   only in `common/models.py`; one hand-built copy once gave TabPFN a fixed 0.
5. **Warnings are filtered by category, never all at once.** `common/quiet.py` silences
   package noise and keeps `RuntimeWarning`, which a blanket filter once hid for months.
6. **A test counts each free unit once.** Three seeds on one dataset are not three
   comparisons, and two dataset names that serve one target vector (`218_house_8L` and
   `house_16H`) are one unit: nine names were eight units, and p = 0.002 was p = 0.0039.
   Cluster with `stats.paired_by_target`, which groups by the recorded `y_sha1`, and quote
   the cluster-level p-value.
7. **A model is written under its canonical name.** `MODELS=XGBoost` and `MODELS=XGB` once
   split one control into two labels. `models.parse_list` normalises names when the
   environment is read, and `tests/test_identity.py` fails on any name in `results/` that
   no script can produce.
