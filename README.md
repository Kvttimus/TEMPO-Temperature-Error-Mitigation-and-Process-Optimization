# TEMPO — Temperature Error Mitigation and Process Optimization

TEMPO is an open-source Python pipeline that applies XGBoost machine learning to reduce
temperature-associated instrumental noise in NASA EZIE-Mag low-cost magnetometer
measurements, using data from the nearby USGS Fredericksburg (FRD) observatory as a reference.

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
    - Download the data from `https://github.com/Kvttimus/TEMPO_EZIEMag_Data_Public_Release/releases/latest` and unzip it into the root of this repo.
    - Or run `wget https://github.com/Kvttimus/TEMPO_EZIEMag_Data_Public_Release/releases/download/v1.0/uncompressed_EZIE_data.zip`
- FRD reference data is downloaded automatically by `step1.1_download_FRD_1sec.py` into
  `uncompressed_FRD_data/`.
- All other intermediate/output folders (`humanReadable_EZIE_data/`, `excel_output/`,
  `regression/`, `plots*/`, `training data/`) are created by the pipeline as you run it.

## Variable names

The code uses these names for the signals it passes between stages. Some are stored
under older column names in the CSV files, which are kept so existing data stays readable.

| Name in code | CSV column | Meaning |
|---|---|---|
| `EZIEH` | `EZIEH` | EZIE-Mag horizontal field, √(Bx² + By²), nT |
| `FRDH` | `FRDH` | FRD horizontal field, √(FRDX² + FRDY²), nT |
| `ctemp` | `ctemp` | EZIE-Mag sensor temperature, °C |
| `EZIEH_ref` | `EZIE_Bh_Predicted` | FRD-derived reference, `a·FRDH + b` (step2.1) |
| `EZIEH_noise_ref` | `residual` | Reference residual, `EZIEH − EZIEH_ref` (the model target) |
| `EZIEH_noise_pred` | `Bh_noise_prediction` | XGBoost-predicted `EZIEH_noise_ref` |
| `tempo_h` | — | TEMPO-corrected field, `EZIEH − EZIEH_noise_pred` |

## Pipeline

### Stage 1 — Data Ingestion (`src/01_data_ingestion/`)
1. `step1.1_download_FRD_1sec.py` — downloads 1-second FRD ground-station observations from USGS/INTERMAGNET.
2. `step1.2_parse_EZIE.py` — parses raw EZIE-Mag binary recordings into one human-readable CSV per day.
3. `step1.3_calculate_H.py` — derives horizontal-field magnitudes (EZIEH, FRDH) from vector components, replacing IAGA fill values with NaN, and writes per-day Excel workbooks.
4. `step1.4_interpolate_frd_excel.py` — fills short FRD data gaps (≤600s) in those workbooks via time-based linear interpolation and recomputes FRDH.

The Excel workbooks from steps 1.3–1.4 are an archival output. Stage 2 reads the per-day
EZIE-Mag CSVs and the raw FRD files directly, dropping FRD fill-value samples rather than
interpolating them.

### Stage 2 — Noise Signal Analysis (`src/02_regression/`)
1. `step2.1_regression.py` — fits a per-day linear regression `EZIEH_ref = a·FRDH + b` on 00:00–12:00 UTC of each day, applies it to the full 24-hour record, and writes `EZIEH_ref` and `EZIEH_noise_ref` to `regression/predicted/<YYYYMMDD>.csv`, plus the daily coefficients to `regression/daily_coeffs.csv`.
2. `step2.2_daily_summary.py` — aggregates per-day statistics (means/ranges of EZIEH, EZIEH_noise_ref, ctemp; the residual range uses 1-minute medians) and the Pearson correlation between ctemp and the residual, saving to `regression/daily_summary.csv`.
3. `step2.3_correlation_coefficients.py` — Pearson correlation, across days, between daily residual range and daily ranges (1st–99th percentile spreads) of ctemp and the inertial sensor channels.
4. `step2.4_daily_ctemp_residual_correlation.py` — supplementary within-day analysis: Pearson and Spearman correlation between ctemp and the residual (5-minute medians) for each day separately, summarized across days for filtered, unfiltered and removed days. **Run after `step3.1`**, which creates the filtered day list (`training data/`) this script reads.

### Stage 3 — ML Noise Correction (`src/03_ml_correction/`)
1. `step3.1_filter_training_data.py` — builds the filtered training set by removing days whose residual range (on 1-minute medians) exceeds 250 nT.
2. `step3.2_xgboost_noise_model.py --mode {filtered,unfiltered} --resolution {1sec,1min,5min} [--features {all,ctemp,ezieh}]` — trains the XGBoost EZIEH_noise_ref model at a given temporal resolution, on the step3.1-filtered days (`filtered`, squared-error loss) or on every day with no day singled out in training, early stopping or evaluation (`unfiltered`, pseudo-Huber loss), using a chronological 60/20/20 day-level train/validation/test split (early stopping on validation; test held out). Hyperparameters are fixed in `build_xgb_params`. `--features` selects the input ablation used by step3.7 (default `all`). Outputs are suffixed by resolution/mode/features (e.g. `xgboost_noise_model_5min_unfiltered.json`, `xgboost_noise_model_1min_ctemp_only.json`); the default `filtered`+`1sec`+`all` combination is unsuffixed.
   - `step3.2b_linear_baseline.py` (same `--mode`/`--resolution` flags) — linear baseline `EZIEH_noise_ref = m₁·ctemp + m₂·EZIEH + b` (same two raw inputs as XGBoost) on the identical split, for comparison against XGBoost.
3. `step3.3_xgboost_inference.py INPUT [--model PATH] [--out_dir DIR]` — applies a saved 1-second model (default: the `filtered`+`1sec` model) to a CSV or folder of CSVs without retraining, adding a `Bh_noise_prediction` column.
4. `step3.4_survey_plots.py [--model PATH] [--days YYYYMMDD ...] [--out_dir DIR]` — generates the 6-panel daily survey plot (FRDH, EZIEH, ctemp, EZIEH_noise_ref, EZIEH_noise_pred, tempo_h); default model is the 5-minute filtered one. The FRDH/EZIEH/tempo_h panels share one nT span and the two noise panels share identical limits, so panel amplitudes are directly comparable.
5. `step3.5_before_after_error_bars.py` — plots test-set RMSE/MAE before (EZIEH_noise_ref) and after (EZIEH_noise_ref − EZIEH_noise_pred) correction, for the filtered and unfiltered 1-minute and 5-minute models.
6. `step3.6_bootstrap_uncertainty.py` — 95% confidence intervals (day-level bootstrap over test days) for no correction vs linear baseline vs XGBoost test errors, plus paired-bootstrap significance tests between models (filtered and unfiltered, 1-sec, 1-min and 5-min).
7. `step3.7_feature_ablation.py [--mode {filtered,unfiltered}] [--resolutions ...] [--analyze-only]` — input ablation: retrains XGBoost via `step3.2 --features {all,ctemp,ezieh}` (full model vs ctemp-derived features only vs EZIEH-derived features only; identical params and split) and reports RMSE/MAE/R² with the same bootstrap CIs and paired tests. `--by-residual-range` additionally scores the trained models separately on test days with residual range ≤ 250 nT and > 250 nT.
8. `step3.8_tempo_vs_frd_plots.py --days YYYYMMDD ... [--model PATH]` — per-day plots of raw EZIE-Mag, TEMPO-corrected EZIE-Mag, and the FRD reference (FRD mapped into EZIE-Mag units by that day's step2.1 regression), with RMSE against the reference before/after correction.

#### Shared modules
These are imported by the Stage 3 scripts and are not run directly.
- `_feature_engineering.py` — causal features built from ctemp and EZIEH only (rolling, cumulative and EMA anomalies, rolling std, interactions, squared terms); 66 / 61 / 56 features at 1 s / 1 min / 5 min. Used by step3.2, 3.3, 3.4 and 3.8 so training and inference always match.
- `_training_data.py` — loads the filtered or unfiltered days, resamples them, and makes the chronological 60/20/20 split shared by step3.2 and step3.2b. Each run writes `regression/split_days_<mode>_<resolution>.csv`, listing which split every day fell in.
- `_bootstrap.py` — day-level bootstrap (10,000 resamples of the test days) for 95% confidence intervals and paired significance tests, shared by step3.6 and step3.7.

## License

GNU General Public License v3.0 — see `LICENSE`.
