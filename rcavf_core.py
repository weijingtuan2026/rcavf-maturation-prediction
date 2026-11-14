"""Shared training, prediction, Excel, CLI, and GUI utilities for RCAVF models.

The outcome modeled by the manuscript is 6-week failure of unassisted,
ultrasound-defined maturation.  User-facing Excel output reports maturation
probability (1 - failure probability) and a Chinese maturation judgement.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


RANDOM_STATE = 2026

SEGMENTAL_FEATURES = [
    "age",
    "diabetes",
    "albumin",
    "radial_artery_diameter",
    "radial_artery_ri",
    "arterial_wall_calcification",
    "cv_diameter_distal",
    "min_cv_diameter",
    "segmental_diameter_variability",
    "mean_segmental_depth",
]

FULL_FEATURES = SEGMENTAL_FEATURES + [
    "access_flow_2w",
    "mean_diameter_change_2w",
    "mean_depth_change_2w",
    "focal_stenosis_2w",
    "radial_artery_ri_change_2w",
]

FEATURE_LABELS = {
    "age": "Age (years)",
    "diabetes": "Diabetes mellitus (0/1)",
    "albumin": "Serum albumin (g/L)",
    "radial_artery_diameter": "Radial artery diameter (mm)",
    "radial_artery_ri": "Preoperative radial artery RI",
    "arterial_wall_calcification": "Arterial wall calcification (0/1)",
    "cv_diameter_distal": "Distal cephalic vein diameter (mm)",
    "min_cv_diameter": "Minimum segmental cephalic vein diameter (mm)",
    "segmental_diameter_variability": "Segmental diameter variability",
    "mean_segmental_depth": "Mean segmental cephalic vein depth (mm)",
    "access_flow_2w": "2-week access flow (mL/min)",
    "mean_diameter_change_2w": "2-week mean diameter change (mm)",
    "mean_depth_change_2w": "2-week mean depth change (mm)",
    "focal_stenosis_2w": "2-week focal stenosis (0/1)",
    "radial_artery_ri_change_2w": "2-week radial artery RI change",
}

# The first matching alias is renamed to the canonical feature name.
ALIASES = {
    "age": ["age", "age_years"],
    "diabetes": ["diabetes", "diabetes_mellitus"],
    "albumin": ["albumin", "serum_albumin"],
    "radial_artery_diameter": ["radial_artery_diameter", "radial artery diameter"],
    "radial_artery_ri": ["radial_artery_ri", "preoperative_radial_artery_ri"],
    "arterial_wall_calcification": ["arterial_wall_calcification", "arterial calcification"],
    "cv_diameter_distal": ["cv_diameter_distal", "distal_cv_diameter"],
    "cv_diameter_mid": ["cv_diameter_mid", "mid_cv_diameter"],
    "cv_diameter_proximal": ["cv_diameter_proximal", "proximal_cv_diameter"],
    "cv_depth_distal": ["cv_depth_distal", "distal_cv_depth"],
    "cv_depth_mid": ["cv_depth_mid", "mid_cv_depth"],
    "cv_depth_proximal": ["cv_depth_proximal", "proximal_cv_depth"],
    "min_cv_diameter": ["min_cv_diameter", "minimum_cv_diameter"],
    "segmental_diameter_variability": ["segmental_diameter_variability", "cv_diameter_cv"],
    "mean_segmental_depth": ["mean_segmental_depth", "mean_cv_depth"],
    "access_flow_2w": ["access_flow_2w", "brachial_artery_access_flow_2w"],
    "vein_diameter_distal_2w": ["vein_diameter_distal_2w", "draining_vein_diameter_distal_2w"],
    "vein_diameter_mid_2w": ["vein_diameter_mid_2w", "draining_vein_diameter_mid_2w"],
    "vein_diameter_proximal_2w": ["vein_diameter_proximal_2w", "draining_vein_diameter_proximal_2w"],
    "vein_depth_distal_2w": ["vein_depth_distal_2w", "draining_vein_depth_distal_2w"],
    "vein_depth_mid_2w": ["vein_depth_mid_2w", "draining_vein_depth_mid_2w"],
    "vein_depth_proximal_2w": ["vein_depth_proximal_2w", "draining_vein_depth_proximal_2w"],
    "mean_diameter_change_2w": ["mean_diameter_change_2w", "mean_diameter_change"],
    "mean_depth_change_2w": ["mean_depth_change_2w", "mean_depth_change"],
    "focal_stenosis_2w": ["focal_stenosis_2w", "focal_stenosis"],
    "radial_artery_ri_2w": ["radial_artery_ri_2w"],
    "radial_artery_ri_change_2w": ["radial_artery_ri_change_2w", "radial_artery_ri_change"],
}

FAILURE_OUTCOME_ALIASES = [
    "maturation_failure",
    "failure",
    "6w_maturation_failure",
]
MATURITY_OUTCOME_ALIASES = [
    "matured",
    "maturation",
    "6w_maturation",
]

PAPER_THRESHOLDS = {
    "LR": {"segmental": 0.298, "full": 0.339},
    "XGBoost": {"segmental": 0.221, "full": 0.187},
    "CatBoost": {"segmental": 0.254, "full": 0.412},
}


def _normalized_name(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(columns: list[Any], candidates: list[str]) -> Any | None:
    exact = {_normalized_name(c): c for c in columns}
    for candidate in candidates:
        if _normalized_name(candidate) in exact:
            return exact[_normalized_name(candidate)]
    return None


def _binary_value(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value) in (0.0, 1.0):
            return float(value)
        raise ValueError(f"Binary values must be 0 or 1; received {value!r}.")
    text = str(value).strip().lower()
    positives = {"1", "yes", "y", "true", "present", "failure", "failed", "not mature"}
    negatives = {"0", "no", "n", "false", "absent", "mature", "matured", "success"}
    if text in positives:
        return 1.0
    if text in negatives:
        return 0.0
    raise ValueError(f"Unrecognized binary value {value!r}. Use 0/1, yes/no, or present/absent.")


def canonicalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rename aliases, derive manuscript composites, and coerce predictors."""
    result = df.copy()
    for canonical, aliases in ALIASES.items():
        if canonical in result.columns:
            continue
        source = _find_column(list(result.columns), aliases)
        if source is not None:
            result[canonical] = result[source]

    binary_columns = ["diabetes", "arterial_wall_calcification", "focal_stenosis_2w"]
    for column in binary_columns:
        if column in result.columns:
            result[column] = result[column].map(_binary_value)

    numeric_candidates = set(ALIASES) - set(binary_columns)
    for column in numeric_candidates:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    diameter_cols = ["cv_diameter_distal", "cv_diameter_mid", "cv_diameter_proximal"]
    if all(c in result.columns for c in diameter_cols):
        diameters = result[diameter_cols].apply(pd.to_numeric, errors="coerce")
        if "min_cv_diameter" not in result.columns:
            result["min_cv_diameter"] = diameters.min(axis=1)
        if "segmental_diameter_variability" not in result.columns:
            means = diameters.mean(axis=1)
            result["segmental_diameter_variability"] = diameters.std(axis=1, ddof=0) / means.replace(0, np.nan)

    depth_cols = ["cv_depth_distal", "cv_depth_mid", "cv_depth_proximal"]
    if "mean_segmental_depth" not in result.columns and all(c in result.columns for c in depth_cols):
        result["mean_segmental_depth"] = result[depth_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    post_diameter_cols = ["vein_diameter_distal_2w", "vein_diameter_mid_2w", "vein_diameter_proximal_2w"]
    if "mean_diameter_change_2w" not in result.columns and all(c in result.columns for c in post_diameter_cols + diameter_cols):
        post = result[post_diameter_cols].apply(pd.to_numeric, errors="coerce")
        pre = result[diameter_cols].apply(pd.to_numeric, errors="coerce")
        result["mean_diameter_change_2w"] = post.to_numpy().mean(axis=1) - pre.to_numpy().mean(axis=1)

    post_depth_cols = ["vein_depth_distal_2w", "vein_depth_mid_2w", "vein_depth_proximal_2w"]
    if "mean_depth_change_2w" not in result.columns and all(c in result.columns for c in post_depth_cols + depth_cols):
        post = result[post_depth_cols].apply(pd.to_numeric, errors="coerce")
        pre = result[depth_cols].apply(pd.to_numeric, errors="coerce")
        result["mean_depth_change_2w"] = post.to_numpy().mean(axis=1) - pre.to_numpy().mean(axis=1)

    if "radial_artery_ri_change_2w" not in result.columns and {"radial_artery_ri_2w", "radial_artery_ri"}.issubset(result.columns):
        result["radial_artery_ri_change_2w"] = result["radial_artery_ri_2w"] - result["radial_artery_ri"]
    return result


def validate_features(df: pd.DataFrame, features: list[str], context: str) -> None:
    missing = [f for f in features if f not in df.columns]
    if missing:
        detail = "\n".join(f"- {f}: {FEATURE_LABELS.get(f, f)}" for f in missing)
        raise ValueError(f"{context} is missing the following required fields:\n{detail}\n\nUse the English column names listed in README.md.")
    empty = [f for f in features if pd.to_numeric(df[f], errors="coerce").isna().all()]
    if empty:
        detail = "、".join(FEATURE_LABELS.get(f, f) for f in empty)
        raise ValueError(f"The following fields in {context} contain no valid numeric values: {detail}")


def extract_failure_outcome(df: pd.DataFrame, requested: str | None = None) -> tuple[pd.Series, str]:
    columns = list(df.columns)
    source = None
    means_maturity = False
    if requested:
        source = _find_column(columns, [requested])
        if source is None:
            raise ValueError(f"The requested outcome column was not found in the training data: {requested}")
        means_maturity = _normalized_name(requested) in {_normalized_name(x) for x in MATURITY_OUTCOME_ALIASES}
    else:
        source = _find_column(columns, FAILURE_OUTCOME_ALIASES)
        if source is None:
            source = _find_column(columns, MATURITY_OUTCOME_ALIASES)
            means_maturity = source is not None
    if source is None:
        raise ValueError("No outcome column was found. Provide maturation_failure (1=failure) or matured (1=mature).")

    raw = df[source]
    if means_maturity:
        # For maturity-named numeric columns, 1 means matured. Text is mapped explicitly.
        if pd.api.types.is_numeric_dtype(raw):
            y = 1.0 - raw.astype(float)
        else:
            maturity_map = {
                "mature": 0.0, "matured": 0.0, "success": 0.0, "yes": 0.0, "1": 0.0,
                "not mature": 1.0, "failed": 1.0, "failure": 1.0, "no": 1.0, "0": 1.0,
            }
            y = raw.map(lambda v: np.nan if pd.isna(v) else maturity_map.get(str(v).strip().lower(), np.nan))
    else:
        y = raw.map(_binary_value)
    if y.isna().any():
        bad_rows = list((y[y.isna()].index + 2)[:10])
        raise ValueError(f"The outcome column contains missing or unrecognized values. Example Excel rows: {bad_rows}")
    unique = set(y.astype(int).unique())
    if unique != {0, 1}:
        raise ValueError("The training outcome must contain both maturation and maturation-failure classes.")
    return y.astype(int), str(source)


def _imports() -> dict[str, Any]:
    try:
        import joblib
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Modeling dependencies are missing. Run: python3 -m pip install -r requirements.txt") from exc
    return {
        "joblib": joblib,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


def make_estimator(algorithm: str, y: pd.Series) -> Any:
    libs = _imports()
    Pipeline = libs["Pipeline"]
    SimpleImputer = libs["SimpleImputer"]
    if algorithm == "LR":
        model = libs["LogisticRegression"](
            penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5,
            class_weight="balanced", max_iter=10000, random_state=RANDOM_STATE,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", libs["StandardScaler"]()), ("model", model)])
    if algorithm == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            detail = str(exc).lower()
            if "libomp" in detail or "openmp" in detail:
                raise RuntimeError(
                    "XGBoost requires the OpenMP runtime. On macOS, run `brew install libomp` "
                    "and try again. Windows and Linux packages usually include the required runtime."
                ) from exc
            raise RuntimeError("XGBoost could not be loaded. Run: python3 -m pip install -r requirements.txt") from exc
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        model = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=10, objective="binary:logistic",
            eval_metric="logloss", scale_pos_weight=negatives / max(positives, 1),
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    if algorithm == "CatBoost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("CatBoost is missing. Run: python3 -m pip install -r requirements.txt") from exc
        model = CatBoostClassifier(
            iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=3,
            bagging_temperature=1, loss_function="Logloss", auto_class_weights="Balanced",
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def train_and_predict(algorithm: str, train_df: pd.DataFrame, predict_df: pd.DataFrame, outcome_column: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = canonicalize_features(train_df)
    predict = canonicalize_features(predict_df)
    y, outcome_used = extract_failure_outcome(train_df, outcome_column)
    output = predict_df.copy()
    bundles: dict[str, Any] = {}
    for block, features, time_label in [
        ("segmental", SEGMENTAL_FEATURES, "preoperative"),
        ("full", FULL_FEATURES, "2week"),
    ]:
        validate_features(train, features, "Training data")
        validate_features(predict, features, "Prediction data")
        estimator = make_estimator(algorithm, y)
        estimator.fit(train[features], y)
        failure_probability = estimator.predict_proba(predict[features])[:, 1]
        maturity_probability = 1.0 - failure_probability
        threshold = PAPER_THRESHOLDS[algorithm][block]
        judgement = np.where(failure_probability < threshold, "Mature", "Not mature")
        output[f"{algorithm}_{time_label}_maturation_probability"] = np.round(maturity_probability, 6)
        output[f"{algorithm}_{time_label}_prediction"] = judgement
        bundles[block] = {
            "algorithm": algorithm,
            "block": block,
            "features": features,
            "failure_threshold": threshold,
            "outcome_column": outcome_used,
            "estimator": estimator,
        }
    return output, bundles


def append_results_to_workbook(
    source_path: Path,
    sheet_name: str,
    result_df: pd.DataFrame,
    result_columns: list[str],
    output_path: Path,
    direct_write: bool,
) -> tuple[Path, Path | None]:
    from openpyxl import load_workbook

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    backup_path = None
    if direct_write:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = source_path.with_name(f"{source_path.stem}_backup_{stamp}{source_path.suffix}")
        shutil.copy2(source_path, backup_path)
        target = source_path
    else:
        if source_path != output_path:
            shutil.copy2(source_path, output_path)
        target = output_path

    keep_vba = target.suffix.lower() == ".xlsm"
    workbook = load_workbook(target, keep_vba=keep_vba)
    if sheet_name not in workbook.sheetnames:
        raise ValueError(f"Worksheet not found: {sheet_name}")
    worksheet = workbook[sheet_name]
    header_columns = {
        str(worksheet.cell(row=1, column=column).value): column
        for column in range(1, worksheet.max_column + 1)
        if worksheet.cell(row=1, column=column).value is not None
    }
    next_column = worksheet.max_column + 1
    for column in result_columns:
        column_number = header_columns.get(column)
        if column_number is None:
            column_number = next_column
            next_column += 1
        cell = worksheet.cell(row=1, column=column_number, value=column)
        cell.font = cell.font.copy(bold=True)
        for row_offset, value in enumerate(result_df[column], start=2):
            if pd.isna(value):
                value = None
            worksheet.cell(row=row_offset, column=column_number, value=value)
        if column.endswith("_probability"):
            for row in range(2, len(result_df) + 2):
                worksheet.cell(row=row, column=column_number).number_format = "0.00%"
    workbook.save(target)
    return target, backup_path


def run_prediction(
    algorithm: str,
    train_path: str,
    predict_path: str,
    train_sheet: str | int = 0,
    predict_sheet: str | int = 0,
    outcome_column: str | None = None,
    output_path: str | None = None,
    direct_write: bool = True,
) -> dict[str, Any]:
    train_file = Path(train_path)
    predict_file = Path(predict_path)
    train_df = pd.read_excel(train_file, sheet_name=train_sheet)
    predict_df = pd.read_excel(predict_file, sheet_name=predict_sheet)
    result_df, bundles = train_and_predict(algorithm, train_df, predict_df, outcome_column)
    resolved_predict_sheet = str(predict_sheet)
    if isinstance(predict_sheet, int):
        resolved_predict_sheet = pd.ExcelFile(predict_file).sheet_names[predict_sheet]
    if output_path:
        out = Path(output_path)
    else:
        out = predict_file.with_name(f"{predict_file.stem}_{algorithm}_prediction_results{predict_file.suffix}")
    result_columns = [
        f"{algorithm}_preoperative_maturation_probability",
        f"{algorithm}_preoperative_prediction",
        f"{algorithm}_2week_maturation_probability",
        f"{algorithm}_2week_prediction",
    ]
    written, backup = append_results_to_workbook(
        predict_file, resolved_predict_sheet, result_df, result_columns, out, direct_write
    )
    model_dir = written.parent / "trained_models"
    model_dir.mkdir(exist_ok=True)
    joblib = _imports()["joblib"]
    model_paths = []
    for block, bundle in bundles.items():
        model_path = model_dir / f"{algorithm.lower()}_{block}.joblib"
        joblib.dump(bundle, model_path)
        model_paths.append(model_path)
    return {"workbook": written, "backup": backup, "models": model_paths, "rows": len(result_df)}


def excel_sheets(path: str) -> list[str]:
    if not path:
        return []
    return pd.ExcelFile(path).sheet_names


def run_gui(algorithm: str) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title(f"RCAVF {algorithm} Maturation Prediction")
    root.geometry("780x430")
    root.minsize(720, 400)

    train_var = tk.StringVar()
    predict_var = tk.StringVar()
    train_sheet_var = tk.StringVar()
    predict_sheet_var = tk.StringVar()
    outcome_var = tk.StringVar(value="maturation_failure")
    direct_var = tk.BooleanVar(value=True)
    status_var = tk.StringVar(value="Select a training workbook and a prediction workbook.")

    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    def choose_file(target: tk.StringVar, sheet_var: tk.StringVar, combo: ttk.Combobox) -> None:
        path = filedialog.askopenfilename(title="Select an Excel workbook", filetypes=[("Excel workbooks", "*.xlsx *.xlsm"), ("All files", "*.*")])
        if not path:
            return
        target.set(path)
        try:
            sheets = excel_sheets(path)
            combo.configure(values=sheets)
            if sheets:
                sheet_var.set(sheets[0])
        except Exception as exc:
            messagebox.showerror("Excel read error", str(exc))

    ttk.Label(frame, text="Training Excel with 6-week outcome").grid(row=0, column=0, sticky="w", pady=8)
    ttk.Entry(frame, textvariable=train_var).grid(row=0, column=1, sticky="ew", padx=8)
    train_combo = ttk.Combobox(frame, textvariable=train_sheet_var, state="readonly", width=16)
    train_combo.grid(row=0, column=2, padx=4)
    ttk.Button(frame, text="Browse", command=lambda: choose_file(train_var, train_sheet_var, train_combo)).grid(row=0, column=3)

    ttk.Label(frame, text="Excel workbook to predict").grid(row=1, column=0, sticky="w", pady=8)
    ttk.Entry(frame, textvariable=predict_var).grid(row=1, column=1, sticky="ew", padx=8)
    predict_combo = ttk.Combobox(frame, textvariable=predict_sheet_var, state="readonly", width=16)
    predict_combo.grid(row=1, column=2, padx=4)
    ttk.Button(frame, text="Browse", command=lambda: choose_file(predict_var, predict_sheet_var, predict_combo)).grid(row=1, column=3)

    ttk.Label(frame, text="Training outcome column").grid(row=2, column=0, sticky="w", pady=8)
    ttk.Entry(frame, textvariable=outcome_var).grid(row=2, column=1, sticky="ew", padx=8)
    ttk.Label(frame, text="maturation_failure: 1=failure; matured: 1=mature").grid(row=2, column=2, columnspan=2, sticky="w")

    ttk.Checkbutton(frame, text="Append to the selected prediction workbook (automatic backup)", variable=direct_var).grid(row=3, column=0, columnspan=4, sticky="w", pady=10)

    note = (
        "Each program appends four columns: preoperative and 2-week maturation probabilities and predictions.\n"
        "Probabilities refer to 6-week maturation; predictions use the manuscript's Youden thresholds.\n"
        "This research tool is not a substitute for clinical judgement or external validation."
    )
    ttk.Label(frame, text=note, foreground="#444444", justify="left").grid(row=4, column=0, columnspan=4, sticky="w", pady=12)

    def execute() -> None:
        if not train_var.get() or not predict_var.get():
            messagebox.showwarning("Missing files", "Select both a training workbook and a prediction workbook.")
            return
        try:
            status_var.set("Training both time-point models and writing Excel results...")
            root.update_idletasks()
            result = run_prediction(
                algorithm=algorithm,
                train_path=train_var.get(),
                predict_path=predict_var.get(),
                train_sheet=train_sheet_var.get() or 0,
                predict_sheet=predict_sheet_var.get() or 0,
                outcome_column=outcome_var.get().strip() or None,
                direct_write=direct_var.get(),
            )
            message = f"Completed predictions for {result['rows']} patients.\nResult workbook: {result['workbook']}"
            if result["backup"]:
                message += f"\nBackup workbook: {result['backup']}"
            message += f"\nSaved models: {result['models'][0].parent}"
            status_var.set("Processing completed.")
            messagebox.showinfo("Completed", message)
        except Exception as exc:
            status_var.set("Processing failed.")
            messagebox.showerror("Error", str(exc))

    ttk.Button(frame, text=f"Run {algorithm} training and prediction", command=execute).grid(row=5, column=0, columnspan=4, pady=14)
    ttk.Label(frame, textvariable=status_var).grid(row=6, column=0, columnspan=4, sticky="w")
    root.mainloop()


def cli_or_gui(algorithm: str) -> None:
    parser = argparse.ArgumentParser(description=f"RCAVF {algorithm} maturation prediction")
    parser.add_argument("--train", help="Training Excel workbook with the 6-week outcome")
    parser.add_argument("--predict", help="Excel workbook containing patients to predict")
    parser.add_argument("--train-sheet", default=0, help="Training worksheet name; defaults to the first sheet")
    parser.add_argument("--predict-sheet", default=0, help="Prediction worksheet name; defaults to the first sheet")
    parser.add_argument("--outcome", default=None, help="Training outcome column name")
    parser.add_argument("--output", default=None, help="Output path when saving a copy")
    parser.add_argument("--no-direct-write", action="store_true", help="Save a separate result workbook instead of updating the selected file")
    args = parser.parse_args()
    if not args.train and not args.predict:
        run_gui(algorithm)
        return
    if not args.train or not args.predict:
        parser.error("--train and --predict must be provided together")
    try:
        result = run_prediction(
            algorithm=algorithm,
            train_path=args.train,
            predict_path=args.predict,
            train_sheet=args.train_sheet,
            predict_sheet=args.predict_sheet,
            outcome_column=args.outcome,
            output_path=args.output,
            direct_write=not args.no_direct_write,
        )
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Completed {result['rows']} rows. Result: {result['workbook']}")
    if result["backup"]:
        print(f"Backup: {result['backup']}")

