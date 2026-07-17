# Biosensors-DMREF
# Voltammetry Concentration Regression

Machine-learning models that predict neurotransmitter concentration from
laser-induced graphene (LIG) biosensor voltammograms. A **separate specialized
model is trained per analyte × source group** (e.g. `EP single`, `DA mixture`).

Analytes: **DA** (dopamine), **SER** (serotonin), **EP** (epinephrine),
**NE** (norepinephrine). Sources: **single**-analyte and **mixture** measurements.

The target is `log10(concentration in M)`; the headline metric is the **factor
error** `10^MAE` (e.g. a MAE of 0.30 in log units = predictions within ~2× of
the true concentration).

---

## What's in this repo

```
config.py                 paths + hyperparameters + per-group working ranges
train_all_final.py        production trainer: engineered features -> tree models
train_parabolic.py        experiment: adds parabolic peak-fit features + cascade
src/
  __init__.py
  data.py                 load + clean the dataset, group masking, replicate handling
  features.py             the ~178 engineered features (binned I(V), dI/dV, stats)
  gdrive.py               resolve the dataset path (Google Drive or local)
  parabolic_fit.py        parabola + 8 fixed-center Gaussians (used by train_parabolic)
  calibration_plot.py     shared level-averaged parity plot (dual SD + SE bars)
```

Outputs are written to `artifacts/<analyte>_<source>_*/`.

---

## Setup
I uploaded all the parsed LIG voltammetry plots to https://drive.google.com/file/d/1dSDhQR2_Rc2rCb1UzcON-MN580HjfEaO/view?usp=sharing
Anyone who wishes to replicate the results from this repo can download the file named "voltammetry_dataset_aligned.pkl" from the link.
Requires Python 3.11 with the scientific stack. On Apple Silicon, a native
(arm64) environment is strongly recommended so NumPy 2.x can read the dataset
pickle.

```bash
conda create -n voltammetry python=3.11
conda activate voltammetry
pip install numpy scipy pandas scikit-learn matplotlib joblib
# optional, only if you want XGBoost in the comparison (needs libomp on macOS):
#   brew install libomp && pip install xgboost
```

### Point the code at your data

Edit `config.py`:

```python
DRIVE_ROOT = "/path/to/your/data/folder"          # or your Google Drive "My Drive"
DATASET_RELATIVE_PATH = "voltammetry_dataset_aligned.pkl"
```

The dataset is a pandas pickle with one row per trace and columns:
`title, voltage, current, conc_index, concentration_M, technique, scan, source,
group, channel, ph, analyte, filename`. The `voltage`/`current` columns hold the
parsed trace arrays. Use the **raw (aligned)** dataset — `train_parabolic.py`
fits a parabolic background and needs un-subtracted traces.

---

## Usage

### Production models — `train_all_final.py`

Trains the specialized model for each group on the engineered features,
auto-selecting the lowest-error regressor (ExtraTrees vs HGB; XGBoost too if
installed).

```bash
python train_all_final.py EP single        # one group
python train_all_final.py --all            # all groups, prints a summary table
```

Per group it writes to `artifacts/<group>_final/`:
- `model.joblib` — the fitted winning model
- `parity.png` — out-of-fold predicted vs true (raw scatter)
- `parity_calibration.png` — level-averaged parity with SD + SE error bars
- `metrics.json`, `all_models.csv` — scores for every candidate regressor

### Feature experiment — `train_parabolic.py`

Compares three feature sets on identical folds — **engineered**,
**parabolic (24 Gaussian params)**, and **engineered + parabolic** — and runs a
two-stage classify-then-quantify cascade.

```bash
python train_parabolic.py EP single
python train_parabolic.py DA mixture --no-cascade
```

Writes calibration plots for each feature set, `summary.csv`, and (if the
cascade runs) `cascade_tc_sweep.csv` to `artifacts/<group>_parabolic/`.

---

## How results are kept honest

All reported numbers are **out-of-fold cross-validated** predictions — no
training-set scores. The evaluation is designed to prevent leakage:

1. **Replicate averaging.** The 4 repeated scans of each measurement are averaged
   into one row *before* splitting, so a scan and its near-copy can't land on
   opposite sides of a fold.
2. **Grouped folds.** Splits are grouped by physical measurement
   (`filename + conc_index`), so all traces of one measurement (both DPV and SWV)
   are held out together — the model always predicts a measurement it never saw.
3. **Stratified folds.** Every concentration level appears in every training fold
   (`StratifiedGroupKFold`), avoiding range-restriction artifacts.
4. **Repeated.** The whole CV is rerun over several random seeds and reported as
   mean ± std.
5. **Blanks dropped.** 0 M (`log10(0)`) is excluded from regression.

Because every score is on held-out data, **overfitting would show up as worse
numbers, not better** — it cannot inflate the reported performance.

> Note: cross-validation rules out overfitting, but it cannot by itself
> distinguish genuine analyte signal from a reproducible experimental confound
> (e.g. measurement-order drift) that also correlates with concentration. Such
> confounds require separate controls (permutation / shuffle tests), not CV.

---

## Key configuration (`config.py`)

| Setting | Meaning |
|---|---|
| `DRIVE_ROOT`, `DATASET_RELATIVE_PATH` | where the dataset pickle lives |
| `BIN_SIZE = 3` | bin width for the I(V) and dI/dV features |
| `MIN_TRACE_LENGTH = 200` | drop traces shorter than this |
| `ZERO_CONC_FLOOR = 1e-13` | value 0 M maps to before `log10` (the blank) |
| `MIN_SAMPLES_PER_GROUP = 20` | skip groups smaller than this |
| `WORKING_RANGE_TOPK` | per-group top-k concentration levels to keep |
| `USE_WORKING_RANGE_BY_DEFAULT` | apply the working-range map automatically |
| `RANDOM_STATE = 42` | reproducibility seed |

The **working range** trims the lowest, hardest-to-quantify concentrations per
group (set in `WORKING_RANGE_TOPK`), so each model is graded on the range where
it is actually reliable.

---

## Method summary

- **Features (`features.py`):** per trace — binned current `I(V)` (~83 cols),
  binned derivative `dI/dV` (~83), 10 scalar stats (area, std, skew, peak height,
  peak width, signal-to-background, pH, …), and a technique one-hot. Constant
  columns are dropped within a group, leaving ~178 features.
- **Model:** `ExtraTreesRegressor` (500 randomized trees) is the default winner —
  well suited to small, noisy, high-dimensional data where variance reduction
  matters more than the bias reduction that gradient boosting provides.
- **Parabolic features (`parabolic_fit.py`):** a parabolic baseline plus 8
  Gaussians at fixed electrochemical peak centers are fitted to each raw trace;
  the 24 Gaussian parameters (amplitude/center/width × 8) become features. Fixed
  centers let a peak be fitted even when it is small, unlike auto peak detection.
