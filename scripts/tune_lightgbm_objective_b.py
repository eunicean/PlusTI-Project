#!/usr/bin/env python3
"""Hyperparameter search for objective b fraud detection with LightGBM.

The script uses BO_VIP and BR_PRIVATE as labeled training data and GT_STATE as
the unlabeled bank to score. It mirrors the feature engineering already used in
test.ipynb, then runs a randomized LightGBM search on a stratified validation
split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_FILES = {
    "BO_VIP": "Copia de 01_bo_vip_seed22_n100000.csv",
    "BR_PRIVATE": "Copia de 02_br_privado_seed33_n100000.csv",
    "GT_STATE": "Copia de 03_gt_estatal_seed3_n100000.csv",
}

COLS_DROP_NULL = [
    "DE56_original_data",
    "DE103_account_id_2",
    "DE54_additional_amounts",
    "DE48_additional_data",
    "DE44_additional_response_data",
]

COLS_DROP_ID = [
    "transaction_id",
    "pan_masked",
    "pan_hash",
    "DE2_PAN",
    "DE11_STAN",
    "DE37_retrieval_reference_number",
    "DE35_track2_data_masked",
    "DE43_card_acceptor_name_location",
    "DE58_authorizing_agent_id",
    "DE63_network_specific",
    "DE56_original_data",
]

COLS_DROP_REDUNDANT = [
    "DE12_local_time",
    "DE13_local_date",
    "DE50_currency_code_settlement",
    "DE51_currency_code_billing",
    "bank_name",
    "bank_country",
    "DE15_settlement_date",
]

CAT_ENCODE = [
    "bank_tier",
    "client_segment",
    "channel",
    "card_brand",
    "DE3_processing_code",
    "DE22_pos_entry_mode",
    "DE25_pos_condition_code",
    "DE39_response_code",
    "DE49_currency_code_transaction",
    "currency_tx_alpha",
    "MTI",
    "source_bank",
]

FEATURES = [
    "hour_local",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_night",
    "tx_month",
    "tx_weekday",
    "card_txn_number",
    "client_txn_number",
    "is_first_card_txn",
    "is_early_card_txn",
    "is_new_client",
    "is_early_client_txn",
    "card_age_days",
    "time_since_last_txn_min",
    "amount_usd_log",
    "amount_local",
    "client_baseline_amount_log",
    "amount_ratio_baseline",
    "amount_above_baseline",
    "amount_zscore",
    "is_online",
    "is_contactless",
    "is_manual_entry",
    "is_fallback",
    "is_cnp",
    "DE52_pin_data_present",
    "DE55_emv_data_present",
    "is_international",
    "distance_log",
    "high_distance",
    "client_tx_count",
    "client_avg_amount",
    "client_std_amount",
    "mcc_high_risk",
    "has_conversion",
    "tx_declined",
    "approved_int",
    "bank_tier_enc",
    "client_segment_enc",
    "channel_enc",
    "card_brand_enc",
    "DE3_processing_code_enc",
    "DE22_pos_entry_mode_enc",
    "DE25_pos_condition_code_enc",
    "DE39_response_code_enc",
    "DE49_currency_code_transaction_enc",
    "MTI_enc",
    "source_bank_enc",
]

PARAM_DISTRIBUTIONS = {
    "n_estimators": [500, 800, 1100, 1400],
    "learning_rate": [0.015, 0.02, 0.03, 0.045, 0.06],
    "num_leaves": [15, 31, 47, 63, 95, 127],
    "max_depth": [-1, 5, 7, 9, 11, 13],
    "min_child_samples": [20, 40, 60, 90, 120, 180],
    "subsample": [0.70, 0.80, 0.85, 0.90, 1.00],
    "colsample_bytree": [0.65, 0.75, 0.85, 0.95, 1.00],
    "reg_alpha": [0.0, 0.05, 0.1, 0.3, 0.7, 1.0],
    "reg_lambda": [0.0, 0.1, 0.3, 0.7, 1.0, 2.0],
    "min_split_gain": [0.0, 0.01, 0.03, 0.06, 0.1],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional stratified sample for faster experiments.",
    )
    return parser.parse_args()


def load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw = {}
    for bank, filename in DATA_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo esperado: {path}")
        raw[bank] = pd.read_csv(path, sep=";", low_memory=False)
        raw[bank]["source_bank"] = bank

    gt_transaction_id = raw["GT_STATE"]["transaction_id"].copy()
    train = pd.concat([raw["BO_VIP"], raw["BR_PRIVATE"]], ignore_index=True)
    test = raw["GT_STATE"].copy()
    return train, test, gt_transaction_id


def drop_unneeded_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_drop = set(COLS_DROP_NULL + COLS_DROP_ID + COLS_DROP_REDUNDANT)
    train = train.drop(columns=[col for col in final_drop if col in train.columns])
    test = test.drop(columns=[col for col in final_drop if col in test.columns])
    return train.drop_duplicates(), test


def add_early_detection_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create chronological features focused on first card fraud detection."""
    df = df.copy()
    if "pan_hash" not in df.columns or "client_id" not in df.columns:
        return df

    tx_datetime = pd.to_datetime(df["DE7_transmission_datetime"], errors="coerce")
    ordered = pd.DataFrame(
        {
            "original_index": df.index,
            "pan_hash": df["pan_hash"].astype(str).fillna("UNKNOWN_CARD"),
            "client_id": df["client_id"].astype(str).fillna("UNKNOWN_CLIENT"),
            "tx_datetime": tx_datetime,
        }
    ).sort_values(["pan_hash", "tx_datetime", "original_index"])

    card_group = ordered.groupby("pan_hash", sort=False)
    ordered["card_txn_number"] = card_group.cumcount() + 1
    ordered["card_first_datetime"] = card_group["tx_datetime"].transform("min")
    ordered["card_age_days"] = (
        (ordered["tx_datetime"] - ordered["card_first_datetime"]).dt.total_seconds() / 86400
    ).fillna(0)
    ordered["time_since_last_txn_min"] = (
        card_group["tx_datetime"].diff().dt.total_seconds() / 60
    ).fillna(-1)

    ordered = ordered.sort_values(["client_id", "tx_datetime", "original_index"])
    client_group = ordered.groupby("client_id", sort=False)
    ordered["client_txn_number"] = client_group.cumcount() + 1

    ordered = ordered.set_index("original_index").sort_index()
    df["card_txn_number"] = ordered["card_txn_number"]
    df["client_txn_number"] = ordered["client_txn_number"]
    df["is_first_card_txn"] = (df["card_txn_number"] == 1).astype(int)
    df["is_early_card_txn"] = (df["card_txn_number"] <= 3).astype(int)
    df["is_new_client"] = (df["client_txn_number"] == 1).astype(int)
    df["is_early_client_txn"] = (df["client_txn_number"] <= 3).astype(int)
    df["card_age_days"] = ordered["card_age_days"]
    df["time_since_last_txn_min"] = ordered["time_since_last_txn_min"]
    return df


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["DE7_transmission_datetime"], errors="coerce")
    df["tx_month"] = dt.dt.month
    df["tx_day"] = dt.dt.day
    df["tx_hour"] = dt.dt.hour
    df["tx_weekday"] = dt.dt.dayofweek
    return df.drop(columns=["DE7_transmission_datetime"])


def impute_from_train(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    num_cols = train.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        if col == "is_fraud":
            continue
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        if col in test.columns:
            test[col] = test[col].fillna(median_val)

    cat_cols = train.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        train[col] = train[col].fillna("UNKNOWN")
        if col in test.columns:
            test[col] = test[col].fillna("UNKNOWN")

    return train, test


def feature_engineering(
    df: pd.DataFrame,
    client_stats: pd.DataFrame | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    df["is_weekend"] = df["day_of_week"].isin(["Sat", "Sun"]).astype(int)
    df["is_night"] = df["hour_local"].between(0, 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_local"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_local"] / 24)

    df["is_online"] = df["DE22_pos_entry_mode"].isin(["081", "810", "81", 81]).astype(int)
    df["is_contactless"] = df["DE22_pos_entry_mode"].isin(["070", "710", "70", 70]).astype(int)
    df["is_manual_entry"] = df["DE22_pos_entry_mode"].isin(["010", "10", 10]).astype(int)
    df["is_fallback"] = df["DE22_pos_entry_mode"].isin(["080", "80", 80]).astype(int)

    cnp_modes = ["081", "810", "81", 81, "010", "10", 10]
    df["is_cnp"] = (
        df["DE22_pos_entry_mode"].isin(cnp_modes)
        | df["DE25_pos_condition_code"].isin(["01", "08", "59", 1, 8, 59])
    ).astype(int)

    df["pin_used"] = (df["DE52_pin_data_present"].astype(str).str.upper() == "Y").astype(int)
    df["emv_used"] = (df["DE55_emv_data_present"].astype(str).str.upper() == "Y").astype(int)
    df["approved_int"] = df["approved"].astype(bool).astype(int)
    df = df.drop(columns=["approved"])

    df["amount_usd_log"] = np.log1p(df["amount_usd"])
    df["client_baseline_amount_log"] = np.log1p(df["client_baseline_amount"])
    df["amount_ratio_baseline"] = df["amount_usd"] / (df["client_baseline_amount"] + 1)
    df["amount_above_baseline"] = (df["amount_usd"] > df["client_baseline_amount"]).astype(int)

    df["is_international"] = df["is_international"].astype(int)
    df["distance_log"] = np.log1p(df["distance_from_home_km"])
    df["high_distance"] = (df["distance_from_home_km"] > 500).astype(int)
    df["has_conversion"] = (df["DE9_conversion_rate_billing"] != 1.0).astype(int)

    if fit:
        agg_dict = {
            "client_tx_count": ("amount_usd", "count"),
            "client_avg_amount": ("amount_usd", "mean"),
            "client_std_amount": ("amount_usd", "std"),
        }
        client_stats = df.groupby("client_id").agg(**agg_dict).reset_index()
    if client_stats is None:
        raise ValueError("client_stats es requerido cuando fit=False")

    stats_cols = [
        "client_tx_count",
        "client_avg_amount",
        "client_std_amount",
        "client_tx_count_x",
        "client_avg_amount_x",
        "client_std_amount_x",
        "client_tx_count_y",
        "client_avg_amount_y",
        "client_std_amount_y",
    ]
    df = df.drop(columns=[col for col in stats_cols if col in df.columns])
    df = df.merge(client_stats, on="client_id", how="left")
    df["client_tx_count"] = df["client_tx_count"].fillna(0)
    df["client_avg_amount"] = df["client_avg_amount"].fillna(df["amount_usd"].median())
    df["client_std_amount"] = df["client_std_amount"].fillna(0)
    df["amount_zscore"] = (
        (df["amount_usd"] - df["client_avg_amount"]) / (df["client_std_amount"] + 1e-6)
    )

    high_risk_mcc = {6011, 6051, 7995, 5912, 4829, 6010}
    df["mcc_high_risk"] = df["DE18_merchant_category_code"].isin(high_risk_mcc).astype(int)
    df["tx_declined"] = (df["DE39_response_code"] != "00").astype(int)

    return df, client_stats


def encode_categories(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    for col in CAT_ENCODE:
        if col not in train.columns:
            continue
        encoder = LabelEncoder()
        train[col + "_enc"] = encoder.fit_transform(train[col].astype(str))
        mapping = {value: idx for idx, value in enumerate(encoder.classes_)}
        test[col + "_enc"] = test[col].astype(str).map(mapping).fillna(-1).astype(int)
    return train, test


def prepare_features(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train, test, gt_transaction_id = load_raw_data(args.data_dir)
    train = add_early_detection_features(train)
    test = add_early_detection_features(test)
    train, test = drop_unneeded_columns(train, test)
    train = add_datetime_features(train)
    test = add_datetime_features(test)
    train, test = impute_from_train(train, test)
    train, client_stats = feature_engineering(train, fit=True)
    test, _ = feature_engineering(test, client_stats=client_stats, fit=False)
    train, test = encode_categories(train, test)

    selected = [feature for feature in FEATURES if feature in train.columns]
    x_train = train[selected].copy()
    y_train = train["is_fraud"].astype(int)
    x_test = test[selected].copy()

    numeric_features = x_train.select_dtypes(include=[np.number, bool]).columns.tolist()
    x_train = x_train[numeric_features].replace([np.inf, -np.inf], np.nan)
    x_test = x_test[numeric_features].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(medians)
    x_test = x_test.fillna(medians)

    if args.max_train_rows and args.max_train_rows < len(x_train):
        x_train, _, y_train, _ = train_test_split(
            x_train,
            y_train,
            train_size=args.max_train_rows,
            random_state=args.random_state,
            stratify=y_train,
        )

    return x_train, y_train, x_test, gt_transaction_id


def threshold_metrics(y_true: pd.Series, y_proba: np.ndarray) -> dict[str, float | int]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1_scores[:-1]))
    best_threshold = float(thresholds[best_idx])
    y_pred = (y_proba >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "best_threshold": best_threshold,
        "f1_best": float(f1_scores[best_idx]),
        "precision_best": float(precision[best_idx]),
        "recall_best": float(recall[best_idx]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_model(params: dict, scale_pos_weight: float, random_state: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        **params,
    )


def run_search(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train,
        y_train,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_train,
    )
    scale_pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    sampler = ParameterSampler(
        PARAM_DISTRIBUTIONS,
        n_iter=args.n_iter,
        random_state=args.random_state,
    )

    rows = []
    best = None
    for trial, params in enumerate(sampler, start=1):
        model = make_model(params, scale_pos_weight, args.random_state + trial)
        model.fit(x_tr, y_tr)
        y_proba = model.predict_proba(x_val)[:, 1]

        metrics = threshold_metrics(y_val, y_proba)
        row = {
            "trial": trial,
            "roc_auc": float(roc_auc_score(y_val, y_proba)),
            "pr_auc": float(average_precision_score(y_val, y_proba)),
            **metrics,
            **params,
        }
        rows.append(row)

        if best is None or (row["f1_best"], row["pr_auc"]) > (best["f1_best"], best["pr_auc"]):
            best = row
            print(
                f"Trial {trial:02d}/{args.n_iter}: nuevo mejor "
                f"F1={row['f1_best']:.4f}, PR-AUC={row['pr_auc']:.4f}, "
                f"umbral={row['best_threshold']:.4f}"
            )
        else:
            print(
                f"Trial {trial:02d}/{args.n_iter}: "
                f"F1={row['f1_best']:.4f}, PR-AUC={row['pr_auc']:.4f}"
            )

    results = pd.DataFrame(rows).sort_values(["f1_best", "pr_auc"], ascending=False)
    return results, dict(best)


def save_final_outputs(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    gt_transaction_id: pd.Series,
    best: dict,
    output_dir: Path,
    random_state: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    param_keys = set(PARAM_DISTRIBUTIONS)
    best_params = {key: best[key] for key in param_keys}
    final_scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    final_model = make_model(best_params, final_scale_pos_weight, random_state)
    final_model.fit(x_train, y_train)

    gt_proba = final_model.predict_proba(x_test)[:, 1]
    gt_pred = (gt_proba >= best["best_threshold"]).astype(int)
    predictions = pd.DataFrame(
        {
            "transaction_id": gt_transaction_id.values,
            "row_id": np.arange(len(x_test)),
            "fraud_probability": gt_proba,
            "fraud_pred": gt_pred,
        }
    )
    predictions.to_csv(output_dir / "predicciones_gt_lightgbm_tuned.csv", index=False)

    importances = (
        pd.DataFrame(
            {
                "feature": x_train.columns,
                "importance": final_model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importances.to_csv(output_dir / "lgbm_feature_importance.csv", index=False)

    summary = {
        "primary_metric": "f1_best",
        "best_validation_metrics": {
            key: best[key]
            for key in [
                "f1_best",
                "precision_best",
                "recall_best",
                "roc_auc",
                "pr_auc",
                "best_threshold",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        },
        "best_params": best_params,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "frauds_estimated_gt": int(gt_pred.sum()),
        "frauds_estimated_gt_pct": float(gt_pred.mean()),
    }
    (output_dir / "lgbm_best_params.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparando features desde los CSV originales...")
    x_train, y_train, x_test, gt_transaction_id = prepare_features(args)
    print(f"Train: {x_train.shape} | distribución target: {y_train.value_counts().to_dict()}")
    print(f"GT test: {x_test.shape}")

    print(f"\nEjecutando búsqueda aleatoria LightGBM ({args.n_iter} trials)...")
    results, best = run_search(x_train, y_train, args)
    results.to_csv(args.output_dir / "lgbm_tuning_results.csv", index=False)

    print("\nMejor configuración:")
    print(results.head(1).T.to_string(header=False))

    print("\nReentrenando mejor modelo con todo BO+BR y generando predicciones GT...")
    save_final_outputs(
        x_train,
        y_train,
        x_test,
        gt_transaction_id,
        best,
        args.output_dir,
        args.random_state,
    )
    print(f"Archivos guardados en: {args.output_dir.resolve()}")
    print("\nNota: usa lgbm_tuning_results.csv para comparar todos los trials.")


if __name__ == "__main__":
    main()
