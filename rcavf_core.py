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
    "age": "年龄（岁）",
    "diabetes": "糖尿病（0/1）",
    "albumin": "白蛋白（g/L）",
    "radial_artery_diameter": "桡动脉直径（mm）",
    "radial_artery_ri": "术前桡动脉RI",
    "arterial_wall_calcification": "动脉壁钙化（0/1）",
    "cv_diameter_distal": "远端头静脉直径（mm）",
    "min_cv_diameter": "三段头静脉最小直径（mm）",
    "segmental_diameter_variability": "三段直径变异系数",
    "mean_segmental_depth": "三段头静脉平均深度（mm）",
    "access_flow_2w": "术后2周通路流量（mL/min）",
    "mean_diameter_change_2w": "术后2周平均直径变化（mm）",
    "mean_depth_change_2w": "术后2周平均深度变化（mm）",
    "focal_stenosis_2w": "术后2周局灶性狭窄（0/1）",
    "radial_artery_ri_change_2w": "术后2周桡动脉RI变化",
}

# The first matching alias is renamed to the canonical feature name.
ALIASES = {
    "age": ["age", "年龄", "年龄（岁）", "年龄(岁)"],
    "diabetes": ["diabetes", "diabetes_mellitus", "糖尿病", "糖尿病（0/1）", "糖尿病(0/1)"],
    "albumin": ["albumin", "serum_albumin", "白蛋白", "血清白蛋白", "白蛋白（g/L）", "白蛋白(g/L)"],
    "radial_artery_diameter": ["radial_artery_diameter", "radial artery diameter", "桡动脉直径", "桡动脉直径（mm）", "桡动脉直径(mm)"],
    "radial_artery_ri": ["radial_artery_ri", "preoperative_radial_artery_ri", "术前桡动脉ri", "桡动脉ri", "术前RI"],
    "arterial_wall_calcification": ["arterial_wall_calcification", "arterial calcification", "动脉壁钙化", "动脉壁钙化（0/1）", "动脉壁钙化(0/1)"],
    "cv_diameter_distal": ["cv_diameter_distal", "distal_cv_diameter", "远端头静脉直径", "远端头静脉直径（mm）", "远端头静脉直径(mm)"],
    "cv_diameter_mid": ["cv_diameter_mid", "mid_cv_diameter", "中段头静脉直径", "中段头静脉直径（mm）", "中段头静脉直径(mm)"],
    "cv_diameter_proximal": ["cv_diameter_proximal", "proximal_cv_diameter", "近端头静脉直径", "近端头静脉直径（mm）", "近端头静脉直径(mm)"],
    "cv_depth_distal": ["cv_depth_distal", "distal_cv_depth", "远端头静脉深度", "远端头静脉深度（mm）", "远端头静脉深度(mm)"],
    "cv_depth_mid": ["cv_depth_mid", "mid_cv_depth", "中段头静脉深度", "中段头静脉深度（mm）", "中段头静脉深度(mm)"],
    "cv_depth_proximal": ["cv_depth_proximal", "proximal_cv_depth", "近端头静脉深度", "近端头静脉深度（mm）", "近端头静脉深度(mm)"],
    "min_cv_diameter": ["min_cv_diameter", "minimum_cv_diameter", "三段头静脉最小直径", "最小头静脉直径"],
    "segmental_diameter_variability": ["segmental_diameter_variability", "cv_diameter_cv", "三段直径变异系数", "节段直径变异性"],
    "mean_segmental_depth": ["mean_segmental_depth", "mean_cv_depth", "三段头静脉平均深度", "平均头静脉深度"],
    "access_flow_2w": ["access_flow_2w", "brachial_artery_access_flow_2w", "术后2周通路流量", "2周通路流量", "通路流量（mL/min）", "通路流量(mL/min)"],
    "vein_diameter_distal_2w": ["vein_diameter_distal_2w", "draining_vein_diameter_distal_2w", "术后2周远端引流静脉直径"],
    "vein_diameter_mid_2w": ["vein_diameter_mid_2w", "draining_vein_diameter_mid_2w", "术后2周中段引流静脉直径"],
    "vein_diameter_proximal_2w": ["vein_diameter_proximal_2w", "draining_vein_diameter_proximal_2w", "术后2周近端引流静脉直径"],
    "vein_depth_distal_2w": ["vein_depth_distal_2w", "draining_vein_depth_distal_2w", "术后2周远端引流静脉深度"],
    "vein_depth_mid_2w": ["vein_depth_mid_2w", "draining_vein_depth_mid_2w", "术后2周中段引流静脉深度"],
    "vein_depth_proximal_2w": ["vein_depth_proximal_2w", "draining_vein_depth_proximal_2w", "术后2周近端引流静脉深度"],
    "mean_diameter_change_2w": ["mean_diameter_change_2w", "mean_diameter_change", "术后2周平均直径变化", "平均直径变化"],
    "mean_depth_change_2w": ["mean_depth_change_2w", "mean_depth_change", "术后2周平均深度变化", "平均深度变化"],
    "focal_stenosis_2w": ["focal_stenosis_2w", "focal_stenosis", "术后2周局灶性狭窄", "局灶性狭窄", "局灶性狭窄（0/1）", "局灶性狭窄(0/1)"],
    "radial_artery_ri_2w": ["radial_artery_ri_2w", "术后2周桡动脉ri", "2周桡动脉ri"],
    "radial_artery_ri_change_2w": ["radial_artery_ri_change_2w", "radial_artery_ri_change", "术后2周桡动脉ri变化", "桡动脉ri变化"],
}

FAILURE_OUTCOME_ALIASES = [
    "maturation_failure",
    "failure",
    "6w_maturation_failure",
    "未成熟",
    "是否未成熟",
    "6周未成熟",
]
MATURITY_OUTCOME_ALIASES = [
    "matured",
    "maturation",
    "6w_maturation",
    "成熟",
    "是否成熟",
    "6周成熟",
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
        raise ValueError(f"二分类值必须为0或1，收到：{value!r}")
    text = str(value).strip().lower()
    positives = {"1", "yes", "y", "true", "present", "有", "是", "阳性", "存在", "未成熟", "失败", "failure", "failed"}
    negatives = {"0", "no", "n", "false", "absent", "无", "否", "阴性", "不存在", "成熟", "成功", "matured", "success"}
    if text in positives:
        return 1.0
    if text in negatives:
        return 0.0
    raise ValueError(f"无法识别二分类值：{value!r}。请使用0/1、是/否或有/无。")


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
        raise ValueError(f"{context}缺少以下必需字段：\n{detail}\n\n列名可使用README中的英文名或中文别名。")
    empty = [f for f in features if pd.to_numeric(df[f], errors="coerce").isna().all()]
    if empty:
        detail = "、".join(FEATURE_LABELS.get(f, f) for f in empty)
        raise ValueError(f"{context}中的以下字段没有任何有效数值：{detail}")


def extract_failure_outcome(df: pd.DataFrame, requested: str | None = None) -> tuple[pd.Series, str]:
    columns = list(df.columns)
    source = None
    means_maturity = False
    if requested:
        source = _find_column(columns, [requested])
        if source is None:
            raise ValueError(f"训练数据中找不到结局列：{requested}")
        means_maturity = _normalized_name(requested) in {_normalized_name(x) for x in MATURITY_OUTCOME_ALIASES}
    else:
        source = _find_column(columns, FAILURE_OUTCOME_ALIASES)
        if source is None:
            source = _find_column(columns, MATURITY_OUTCOME_ALIASES)
            means_maturity = source is not None
    if source is None:
        raise ValueError("找不到结局列。请提供 maturation_failure（1=未成熟）或 是否成熟（1=成熟）列。")

    raw = df[source]
    if means_maturity:
        # For maturity-named numeric columns, 1 means matured. Text is mapped explicitly.
        if pd.api.types.is_numeric_dtype(raw):
            y = 1.0 - raw.astype(float)
        else:
            maturity_map = {
                "成熟": 0.0, "成功": 0.0, "matured": 0.0, "success": 0.0, "是": 0.0, "yes": 0.0, "1": 0.0,
                "未成熟": 1.0, "失败": 1.0, "failed": 1.0, "failure": 1.0, "否": 1.0, "no": 1.0, "0": 1.0,
            }
            y = raw.map(lambda v: np.nan if pd.isna(v) else maturity_map.get(str(v).strip().lower(), np.nan))
    else:
        y = raw.map(_binary_value)
    if y.isna().any():
        bad_rows = list((y[y.isna()].index + 2)[:10])
        raise ValueError(f"结局列含无法识别或缺失的值，Excel行号示例：{bad_rows}")
    unique = set(y.astype(int).unique())
    if unique != {0, 1}:
        raise ValueError("训练结局必须同时包含成熟和未成熟两类。")
    return y.astype(int), str(source)


def _imports() -> dict[str, Any]:
    try:
        import joblib
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("缺少建模依赖。请先执行：python3 -m pip install -r requirements.txt") from exc
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
                    "XGBoost需要OpenMP运行库。macOS请先执行 `brew install libomp`，"
                    "然后重新运行；Windows/Linux通常由安装包自动提供。"
                ) from exc
            raise RuntimeError("无法加载xgboost。请先执行：python3 -m pip install -r requirements.txt") from exc
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
            raise RuntimeError("缺少catboost。请先执行：python3 -m pip install -r requirements.txt") from exc
        model = CatBoostClassifier(
            iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=3,
            bagging_temperature=1, loss_function="Logloss", auto_class_weights="Balanced",
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    raise ValueError(f"不支持的算法：{algorithm}")


def train_and_predict(algorithm: str, train_df: pd.DataFrame, predict_df: pd.DataFrame, outcome_column: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = canonicalize_features(train_df)
    predict = canonicalize_features(predict_df)
    y, outcome_used = extract_failure_outcome(train_df, outcome_column)
    output = predict_df.copy()
    bundles: dict[str, Any] = {}
    for block, features, time_label in [
        ("segmental", SEGMENTAL_FEATURES, "术前"),
        ("full", FULL_FEATURES, "术后2周"),
    ]:
        validate_features(train, features, "训练数据")
        validate_features(predict, features, "待预测数据")
        estimator = make_estimator(algorithm, y)
        estimator.fit(train[features], y)
        failure_probability = estimator.predict_proba(predict[features])[:, 1]
        maturity_probability = 1.0 - failure_probability
        threshold = PAPER_THRESHOLDS[algorithm][block]
        judgement = np.where(failure_probability < threshold, "成熟", "未成熟")
        output[f"{algorithm}_{time_label}_成熟概率"] = np.round(maturity_probability, 6)
        output[f"{algorithm}_{time_label}_判断"] = judgement
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
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = source_path.with_name(f"{source_path.stem}_备份_{stamp}{source_path.suffix}")
        shutil.copy2(source_path, backup_path)
        target = source_path
    else:
        if source_path != output_path:
            shutil.copy2(source_path, output_path)
        target = output_path

    keep_vba = target.suffix.lower() == ".xlsm"
    workbook = load_workbook(target, keep_vba=keep_vba)
    if sheet_name not in workbook.sheetnames:
        raise ValueError(f"工作表不存在：{sheet_name}")
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
        if column.endswith("概率"):
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
        out = predict_file.with_name(f"{predict_file.stem}_{algorithm}_预测结果{predict_file.suffix}")
    result_columns = [
        f"{algorithm}_术前_成熟概率",
        f"{algorithm}_术前_判断",
        f"{algorithm}_术后2周_成熟概率",
        f"{algorithm}_术后2周_判断",
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
    root.title(f"RCAVF {algorithm} 成熟预测")
    root.geometry("780x430")
    root.minsize(720, 400)

    train_var = tk.StringVar()
    predict_var = tk.StringVar()
    train_sheet_var = tk.StringVar()
    predict_sheet_var = tk.StringVar()
    outcome_var = tk.StringVar(value="maturation_failure")
    direct_var = tk.BooleanVar(value=True)
    status_var = tk.StringVar(value="请选择训练Excel和待预测Excel。")

    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    def choose_file(target: tk.StringVar, sheet_var: tk.StringVar, combo: ttk.Combobox) -> None:
        path = filedialog.askopenfilename(title="选择Excel文件", filetypes=[("Excel工作簿", "*.xlsx *.xlsm"), ("所有文件", "*.*")])
        if not path:
            return
        target.set(path)
        try:
            sheets = excel_sheets(path)
            combo.configure(values=sheets)
            if sheets:
                sheet_var.set(sheets[0])
        except Exception as exc:
            messagebox.showerror("无法读取Excel", str(exc))

    ttk.Label(frame, text="带6周结局的训练Excel").grid(row=0, column=0, sticky="w", pady=8)
    ttk.Entry(frame, textvariable=train_var).grid(row=0, column=1, sticky="ew", padx=8)
    train_combo = ttk.Combobox(frame, textvariable=train_sheet_var, state="readonly", width=16)
    train_combo.grid(row=0, column=2, padx=4)
    ttk.Button(frame, text="选择", command=lambda: choose_file(train_var, train_sheet_var, train_combo)).grid(row=0, column=3)

    ttk.Label(frame, text="待预测Excel").grid(row=1, column=0, sticky="w", pady=8)
    ttk.Entry(frame, textvariable=predict_var).grid(row=1, column=1, sticky="ew", padx=8)
    predict_combo = ttk.Combobox(frame, textvariable=predict_sheet_var, state="readonly", width=16)
    predict_combo.grid(row=1, column=2, padx=4)
    ttk.Button(frame, text="选择", command=lambda: choose_file(predict_var, predict_sheet_var, predict_combo)).grid(row=1, column=3)

    ttk.Label(frame, text="训练结局列名").grid(row=2, column=0, sticky="w", pady=8)
    ttk.Entry(frame, textvariable=outcome_var).grid(row=2, column=1, sticky="ew", padx=8)
    ttk.Label(frame, text="maturation_failure: 1=未成熟；是否成熟: 1=成熟").grid(row=2, column=2, columnspan=2, sticky="w")

    ttk.Checkbutton(frame, text="直接追加到待预测Excel（运行前自动备份）", variable=direct_var).grid(row=3, column=0, columnspan=4, sticky="w", pady=10)

    note = (
        "每个程序会追加4列：术前成熟概率、术前判断、术后2周成熟概率、术后2周判断。\n"
        "概率是6周成熟概率；判断使用论文对应算法与时间点的Youden阈值。\n"
        "本工具用于研究验证，不应在外部验证和临床审批前替代医生判断。"
    )
    ttk.Label(frame, text=note, foreground="#444444", justify="left").grid(row=4, column=0, columnspan=4, sticky="w", pady=12)

    def execute() -> None:
        if not train_var.get() or not predict_var.get():
            messagebox.showwarning("缺少文件", "请先选择训练Excel和待预测Excel。")
            return
        try:
            status_var.set("正在训练两个时间点模型并写入Excel，请稍候……")
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
            message = f"完成：已处理 {result['rows']} 名患者。\n结果文件：{result['workbook']}"
            if result["backup"]:
                message += f"\n备份文件：{result['backup']}"
            message += f"\n模型文件夹：{result['models'][0].parent}"
            status_var.set("处理完成。")
            messagebox.showinfo("完成", message)
        except Exception as exc:
            status_var.set("处理失败。")
            messagebox.showerror("错误", str(exc))

    ttk.Button(frame, text=f"运行 {algorithm} 训练和预测", command=execute).grid(row=5, column=0, columnspan=4, pady=14)
    ttk.Label(frame, textvariable=status_var).grid(row=6, column=0, columnspan=4, sticky="w")
    root.mainloop()


def cli_or_gui(algorithm: str) -> None:
    parser = argparse.ArgumentParser(description=f"RCAVF {algorithm} maturation prediction")
    parser.add_argument("--train", help="带6周结局的训练Excel")
    parser.add_argument("--predict", help="待预测Excel")
    parser.add_argument("--train-sheet", default=0, help="训练工作表名，默认第一张")
    parser.add_argument("--predict-sheet", default=0, help="预测工作表名，默认第一张")
    parser.add_argument("--outcome", default=None, help="训练结局列名")
    parser.add_argument("--output", default=None, help="另存结果路径；不指定时默认直接回写并备份")
    parser.add_argument("--no-direct-write", action="store_true", help="不回写原文件，另存预测结果")
    args = parser.parse_args()
    if not args.train and not args.predict:
        run_gui(algorithm)
        return
    if not args.train or not args.predict:
        parser.error("--train 和 --predict 必须同时提供")
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
        parser.exit(1, f"错误：{exc}\n")
    print(f"完成：{result['rows']}行；结果：{result['workbook']}")
    if result["backup"]:
        print(f"备份：{result['backup']}")

