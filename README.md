# RCAVF Maturation Prediction Models

This repository implements three binary classifiers described in *Machine Learning with Segmental Duplex Ultrasound Predicts Ultrasound-Defined Radiocephalic Fistula Maturation Before Surgery and at the 2-Week Landmark*:

- Elastic-net logistic regression
- XGBoost
- CatBoost

Each algorithm trains two models:

- **Preoperative Segmental model:** 10 preoperative predictors.
- **2-week Full model:** the same 10 predictors plus 5 postoperative dynamic predictors.

Each program appends four columns to the selected prediction worksheet: preoperative maturation probability, preoperative prediction, 2-week maturation probability, and 2-week prediction. Direct writing is enabled by default and creates a timestamped backup before changing the workbook. Running all three programs on the same workbook produces 12 result columns.

## Important limitation

The manuscript and supplementary material provide summary statistics, hyperparameters, and test-set performance, but not the patient-level data or fitted model objects for the 367 participants. The original fitted models therefore cannot be reconstructed from the documents alone.

Each program requires a patient-level training workbook containing the 6-week outcome. It trains models using the fixed manuscript hyperparameters and then predicts a separate workbook. The software is intended for research validation and must not replace clinical judgement before external validation and appropriate regulatory review.

## Installation

Python 3.10 through 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

XGBoost on macOS also requires the OpenMP runtime:

```bash
brew install libomp
```

Activate the virtual environment on Windows with:

```text
.venv\Scripts\activate
```

## Graphical interface

Run one of the following commands from the project directory:

```bash
python lr_model.py
python xgboost_model.py
python catboost_model.py
```

In the application window, select:

1. The training Excel workbook and worksheet containing the 6-week outcome.
2. The Excel workbook and worksheet containing patients to predict.
3. The outcome column in the training data.
4. Whether results should be appended directly to the selected prediction workbook.

The recommended outcome column is `maturation_failure`, with 1 indicating failure of 6-week maturation and 0 indicating maturation. A column named `matured` may also be used, with 1 indicating maturation and 0 indicating failure. Text labels such as `Mature`, `Not mature`, `Matured`, `Failure`, `Yes`, and `No` are supported.

## Command-line interface

```bash
python catboost_model.py \
  --train training_data.xlsx \
  --predict patients_to_predict.xlsx \
  --train-sheet Sheet1 \
  --predict-sheet Sheet1 \
  --outcome maturation_failure
```

Use `--no-direct-write --output prediction_results.xlsx` to preserve the selected prediction workbook and save a separate result file.

## Required predictors

Binary variables may use 0/1, yes/no, or present/absent. Continuous variables must contain numeric values without units in the cells.

| Time point | Column name | Definition |
|---|---|---|
| Preoperative | `age` | Age, years |
| Preoperative | `diabetes` | Diabetes mellitus, 0/1 |
| Preoperative | `albumin` | Serum albumin, g/L |
| Preoperative | `radial_artery_diameter` | Radial artery diameter, mm |
| Preoperative | `radial_artery_ri` | Preoperative radial artery resistance index |
| Preoperative | `arterial_wall_calcification` | Arterial wall calcification, 0/1 |
| Preoperative | `cv_diameter_distal` | Distal cephalic vein diameter, mm |
| Preoperative | `min_cv_diameter` | Minimum of the three cephalic vein diameters, mm |
| Preoperative | `segmental_diameter_variability` | Standard deviation divided by mean of the three diameters |
| Preoperative | `mean_segmental_depth` | Mean of the three cephalic vein depths, mm |
| 2 weeks | `access_flow_2w` | Brachial artery access flow, mL/min |
| 2 weeks | `mean_diameter_change_2w` | Mean paired diameter change, 2 weeks minus preoperative, mm |
| 2 weeks | `mean_depth_change_2w` | Mean paired depth change, 2 weeks minus preoperative, mm |
| 2 weeks | `focal_stenosis_2w` | Focal stenosis, 0/1 |
| 2 weeks | `radial_artery_ri_change_2w` | Radial artery RI change, 2 weeks minus preoperative |

The three preoperative composite predictors do not need to be supplied directly when all three segmental diameters and depths are available. The program can derive them from:

- `cv_diameter_distal`, `cv_diameter_mid`, and `cv_diameter_proximal`
- `cv_depth_distal`, `cv_depth_mid`, and `cv_depth_proximal`

The postoperative change predictors can also be derived from matched measurements when these columns are available:

- `vein_diameter_distal_2w`, `vein_diameter_mid_2w`, and `vein_diameter_proximal_2w`
- `vein_depth_distal_2w`, `vein_depth_mid_2w`, and `vein_depth_proximal_2w`
- `radial_artery_ri_2w`

## Excel output

For example, the CatBoost program appends:

- `CatBoost_preoperative_maturation_probability`
- `CatBoost_preoperative_prediction`
- `CatBoost_2week_maturation_probability`
- `CatBoost_2week_prediction`

The probability columns contain the estimated probability of 6-week maturation. Prediction columns contain `Mature` or `Not mature`.

## Model parameters and thresholds

The estimators use the fixed hyperparameters reported in Supplementary Table S3. Classification thresholds are taken from Table 5 and apply to the modeled event of maturation failure.

| Algorithm | Preoperative failure threshold | 2-week failure threshold |
|---|---:|---:|
| Logistic regression | 0.298 | 0.339 |
| XGBoost | 0.221 | 0.187 |
| CatBoost | 0.254 | 0.412 |

The displayed maturation probability equals `1 - failure probability`. A patient is classified as `Mature` when the estimated failure probability is below the corresponding threshold; otherwise the prediction is `Not mature`.

