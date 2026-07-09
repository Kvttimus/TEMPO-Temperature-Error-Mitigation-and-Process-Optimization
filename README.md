# TEMPO — Temperature Error Mitigation and Process Optimization

TEMPO is an open-source Python pipeline that applies XGBoost machine learning to reduce
temperature-induced instrumental noise in NASA EZIE-Mag low-cost magnetometer
measurements, using co-located USGS Fredericksburg (FRD) observatory data as a reference.

## Setup

```
pip install -r requirements.txt
```

Requires Python 3.11 (dependency versions in `requirements.txt` are pinned to the
versions used to produce the paper's results; other Python 3 versions likely work
but are untested).

**Run every script from the repository root.** Each script resolves its input/output
paths (e.g. `humanReadable_EZIE_data/`, `regression/`) relative to the current working
directory, not its own file location — invoke scripts as
`python src/01_data_ingestion/step1.1_download_FRD_1sec.py`, not by `cd`-ing into a
subfolder first.

## Data

Raw and intermediate data are not included in this repository (data folders are
gitignored). To run the pipeline end-to-end:

- Place raw EZIE-Mag binary recordings in `uncompressed_EZIE_data/` (must be supplied by
  the user — not fetched automatically).
- FRD reference data is downloaded automatically by `step1.1_download_FRD_1sec.py` into
  `uncompressed_FRD_data/`.
- All other intermediate/output folders (`humanReadable_EZIE_data/`, `excel_output/`,
  `regression/`, `plots*/`, `training data/`) are created by the pipeline as you run it.

## Pipeline

### Stage 1 — Data Ingestion (`src/01_data_ingestion/`)
1. `step1.1_download_FRD_1sec.py` — downloads 1-second FRD ground-station observations from USGS/INTERMAGNET.
2. `step1.2_parse_EZIE.py` — parses raw EZIE-Mag binary recordings into human-readable CSVs.
3. `step1.3_calculate_H.py` — derives horizontal-field magnitudes (EZIEH, FRDH) from vector components, replacing IAGA fill values with NaN.
4. `step1.4_interpolate_frd_excel.py` — fills short FRD data gaps (≤600s) via time-based linear interpolation and recomputes FRDH.

### Stage 2 — Noise Signal Analysis (`src/02_regression/`)
1. `step2.1_regression.py` — fits a per-day linear regression `EZIEH_Predicted = a·FRDH + b` on the first 12 hours of each day, applies it to the full 24-hour record, and produces the FRD-derived prediction and residual.
2. `step2.2_daily_summary.py` — aggregates per-day statistics (means/ranges of EZIEH, estimated Bh noise, ctemp) and the Pearson correlation between ctemp and the residual, saving to `regression/daily_summary.csv`.

### Stage 3 — ML Noise Correction (`src/03_ml_correction/`)
1. `step3.1_filter_training_data.py` — filters to quiet (non-storm) days for training.
2. `step3.2_xgboost_noise_model.py --mode {filtered,unfiltered} --resolution {1sec,1min,5min}` — trains the XGBoost Bh-noise model at a given temporal resolution, on quiet-only (`filtered`) or all-day (`unfiltered`) data. Outputs are suffixed by resolution/mode (e.g. `xgboost_noise_model_5min_unfiltered.json`); the default `filtered`+`1sec` combination is unsuffixed.
3. `step3.3_xgboost_inference.py` — applies a saved model (default: the `filtered`+`1sec` model) to new data without retraining.
4. `step3.4_survey_plots.py` — generates the 6-panel raw-vs-corrected survey plot.
5. `step3.5_before_after_error_bars.py` — computes and plots before/after MAE/RMSE.

## License

GNU General Public License v3.0 — see `LICENSE`.
