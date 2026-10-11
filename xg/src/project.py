"""Educational expected-goals model using StatsBomb 2022 World Cup open data."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sklearn


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = ROOT.parent / "models"
REPORTS = ROOT.parent / "reports" / "xg"
BASE_URL = "https://raw.githubusercontent.com/hudl/open-data/master/data/"
COMPETITION_ID, SEASON_ID = 43, 106  # Men's FIFA World Cup 2022
FEATURES = ["distance", "angle", "body_part", "shot_type", "under_pressure", "first_time"]
NUMERIC = ["distance", "angle", "under_pressure", "first_time"]
CATEGORICAL = ["body_part", "shot_type"]
MODEL_PATH = MODELS / "xg_world_cup_2022.joblib"


def fetch_json(url: str, path: Path) -> object:
    """Download once; a cached copy makes later runs independent of the network."""
    if not path.exists():
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return json.loads(path.read_text(encoding="utf-8"))


def shot_geometry(x: float, y: float) -> tuple[float, float]:
    """StatsBomb coordinates: pitch 120 x 80, goal centre (120, 40)."""
    dx, dy = 120.0 - x, 40.0 - y
    distance = math.hypot(dx, dy)
    # Angle between vectors from the shot to the two posts (120,36/44).
    a = np.array([120.0 - x, 36.0 - y])
    b = np.array([120.0 - x, 44.0 - y])
    angle = math.degrees(math.atan2(abs(np.linalg.det(np.stack([a, b]))), float(np.dot(a, b))))
    return distance, angle


def event_to_row(event: dict, match_id: int) -> dict | None:
    if event.get("type", {}).get("name") != "Shot" or event.get("period") == 5:
        return None  # Penalty shootouts are a different situation.
    shot = event.get("shot", {})
    if shot.get("type", {}).get("name") == "Penalty":
        return None  # A fixed-distance penalty deserves a separate model.
    loc = event.get("location", [])
    if len(loc) < 2:
        return None
    x, y = map(float, loc[:2])
    if not (0 <= x <= 120 and 0 <= y <= 80):
        return None
    distance, angle = shot_geometry(x, y)
    return {
        "match_id": match_id,
        "shot_id": event["id"],
        "team": event.get("team", {}).get("name", ""),
        "player": event.get("player", {}).get("name", ""),
        "x": x, "y": y,
        "distance": distance, "angle": angle,
        "body_part": shot.get("body_part", {}).get("name", "Unknown"),
        "shot_type": shot.get("type", {}).get("name", "Unknown"),
        "under_pressure": int(bool(event.get("under_pressure", False))),
        "first_time": int(bool(shot.get("first_time", False))),
        "goal": int(shot.get("outcome", {}).get("name") == "Goal"),
        "statsbomb_xg": shot.get("statsbomb_xg"),  # Reference only; NEVER a model input.
    }


def prepare() -> pd.DataFrame:
    matches = fetch_json(
        f"{BASE_URL}matches/{COMPETITION_ID}/{SEASON_ID}.json",
        RAW / "matches_2022.json",
    )
    if len(matches) != 64:
        raise ValueError(f"Expected 64 World Cup matches, found {len(matches)}")
    rows, failures = [], []
    for match in matches:
        match_id = match["match_id"]
        try:
            events = fetch_json(f"{BASE_URL}events/{match_id}.json", RAW / f"events_{match_id}.json")
            rows.extend(row for event in events if (row := event_to_row(event, match_id)) is not None)
        except (requests.RequestException, OSError, ValueError, KeyError) as exc:
            failures.append({"match_id": match_id, "error": str(exc)})
    if failures:
        REPORTS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(REPORTS / "download_errors.csv", index=False)
        raise RuntimeError(f"Failed to read {len(failures)} matches; see reports/xg/download_errors.csv")
    frame = pd.DataFrame(rows).drop_duplicates("shot_id").sort_values(["match_id", "shot_id"])
    if frame.empty or frame.match_id.nunique() != 64 or frame.goal.nunique() != 2:
        raise ValueError("Shot extraction failed validation")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED / "shots_2022.csv", index=False)
    return frame


def load_data() -> pd.DataFrame:
    path = PROCESSED / "shots_2022.csv"
    return pd.read_csv(path) if path.exists() else prepare()


def make_pipeline(kind: str):
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    transform = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    estimator = {
        "baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                                                random_state=42, n_jobs=-1),
    }[kind]
    return Pipeline([("features", transform), ("model", estimator)])


def scores(y: pd.Series, prob: np.ndarray) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    result = {"brier": float(brier_score_loss(y, prob)),
              "log_loss": float(log_loss(y, prob, labels=[0, 1]))}
    result["roc_auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None
    return result


def train() -> dict:
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    frame = load_data()
    groups = frame.match_id
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(frame, frame.goal, groups))
    train_df, test_df = frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()
    if set(train_df.match_id) & set(test_df.match_id):
        raise AssertionError("A match appears in both train and test")
    comparisons = []
    for kind in ("baseline", "logistic_regression", "random_forest"):
        fold_scores = []
        for fit_idx, val_idx in GroupKFold(n_splits=5).split(train_df, train_df.goal, train_df.match_id):
            fit, val = train_df.iloc[fit_idx], train_df.iloc[val_idx]
            pipeline = make_pipeline(kind).fit(fit[FEATURES], fit.goal)
            fold_scores.append(scores(val.goal, pipeline.predict_proba(val[FEATURES])[:, 1])["brier"])
        comparisons.append({"model": kind, "cv_brier_mean": float(np.mean(fold_scores)),
                            "cv_brier_std": float(np.std(fold_scores))})
    comparison = pd.DataFrame(comparisons).sort_values("cv_brier_mean")
    winner = str(comparison.iloc[0].model)
    model = make_pipeline(winner).fit(train_df[FEATURES], train_df.goal)
    prob = model.predict_proba(test_df[FEATURES])[:, 1]
    prior = make_pipeline("baseline").fit(train_df[FEATURES], train_df.goal)
    baseline_scores = scores(test_df.goal, prior.predict_proba(test_df[FEATURES])[:, 1])
    result = {
        "model": winner, "matches": int(frame.match_id.nunique()),
        "shots": len(frame), "goals": int(frame.goal.sum()),
        "train_matches": int(train_df.match_id.nunique()), "test_matches": int(test_df.match_id.nunique()),
        "train_shots": len(train_df), "test_shots": len(test_df),
        "test": scores(test_df.goal, prob), "baseline_test": baseline_scores,
        "statsbomb_reference_test": scores(test_df.goal, test_df.statsbomb_xg.to_numpy()) if test_df.statsbomb_xg.notna().all() else None,
    }
    bounds = {name: [float(train_df[name].min()), float(train_df[name].max())] for name in ("x", "y", "distance", "angle")}
    categories = {name: sorted(train_df[name].dropna().unique().tolist()) for name in CATEGORICAL}
    bundle = {"model": model, "model_name": winner, "features": FEATURES,
              "bounds": bounds, "categories": categories, "metrics": result,
              "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    test_df = test_df.assign(predicted_xg=prob, absolute_error=np.abs(test_df.goal - prob))
    test_df.sort_values("absolute_error", ascending=False).to_csv(REPORTS / "test_predictions.csv", index=False)
    (REPORTS / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_evaluation(test_df)
    return result


def plot_evaluation(test_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(test_df.loc[test_df.goal == 0, "predicted_xg"], bins=20, alpha=.7, label="Không bàn thắng")
    axes[0].hist(test_df.loc[test_df.goal == 1, "predicted_xg"], bins=20, alpha=.7, label="Bàn thắng")
    axes[0].set(xlabel="xG dự đoán", ylabel="Số cú sút", title="Phân bố xác suất")
    axes[0].legend()
    # Equal-width bins; empirical goal rate compares probabilities with outcomes.
    bins = np.linspace(0, 1, 11)
    bins[-1] += 1e-9
    grouped = test_df.groupby(pd.cut(test_df.predicted_xg, bins, include_lowest=True), observed=True)
    centres = grouped.predicted_xg.mean()
    rates = grouped.goal.mean()
    sizes = grouped.size()
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].scatter(centres, rates, s=np.maximum(30, sizes * 2))
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="xG trung bình", ylabel="Tỷ lệ thành bàn", title="Kiểm tra hiệu chỉnh (cỡ chấm = số cú sút)")
    fig.tight_layout()
    fig.savefig(REPORTS / "evaluation.png", dpi=150)
    plt.close(fig)


def validate_shot(shot: dict) -> dict:
    required = ["x", "y", "body_part", "shot_type", "under_pressure", "first_time"]
    missing = [key for key in required if key not in shot]
    if missing:
        raise ValueError("Thiếu trường: " + ", ".join(missing))
    for key, upper in (("x", 120), ("y", 80)):
        if isinstance(shot[key], bool) or not isinstance(shot[key], (int, float)) or not math.isfinite(shot[key]):
            raise ValueError(f"{key} phải là số hữu hạn")
        if not 0 <= shot[key] <= upper:
            raise ValueError(f"{key} phải trong khoảng 0–{upper}")
    for key in CATEGORICAL:
        if not isinstance(shot[key], str) or not shot[key].strip():
            raise ValueError(f"{key} phải là chuỗi không rỗng")
    for key in ("under_pressure", "first_time"):
        if not isinstance(shot[key], bool):
            raise ValueError(f"{key} phải là true hoặc false")
    distance, angle = shot_geometry(float(shot["x"]), float(shot["y"]))
    return {"x": float(shot["x"]), "y": float(shot["y"]), "distance": distance,
            "angle": angle, "body_part": shot["body_part"], "shot_type": shot["shot_type"],
            "under_pressure": int(shot["under_pressure"]), "first_time": int(shot["first_time"])}


def predict(shot: dict, path: Path = MODEL_PATH) -> dict:
    row = validate_shot(shot)
    bundle = joblib.load(path)  # Load only a model file you trust.
    warnings = []
    for name in ("x", "y", "distance", "angle"):
        low, high = bundle["bounds"][name]
        if not low <= row[name] <= high:
            warnings.append(f"{name} nằm ngoài khoảng dữ liệu train [{low:.2f}, {high:.2f}]")
    for name in CATEGORICAL:
        if row[name] not in bundle["categories"][name]:
            warnings.append(f"{name} chưa xuất hiện trong dữ liệu train")
    probability = float(bundle["model"].predict_proba(pd.DataFrame([row])[FEATURES])[:, 1][0])
    return {"xg": round(probability, 4), "probability_percent": round(probability * 100, 2),
            "model": bundle["model_name"], "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mô hình xG học tập từ World Cup 2022")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("train")
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        frame = prepare()
        print(json.dumps({"matches": int(frame.match_id.nunique()), "shots": len(frame),
                          "goals": int(frame.goal.sum())}, ensure_ascii=False))
    elif args.command == "train":
        print(json.dumps(train(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(predict(json.loads(args.input.read_text(encoding="utf-8"))), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
