import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess_dataframe
from candidate_generation import generate_all_candidates
from feature_engineering import extract_features


# ============================================================
# CONFIG
# ============================================================

DEFAULT_DATASET_DIR = (
    r"C:\Users\Acer\Downloads"
    r"\6ab10eb3b23ba_student_resource"
    r"\student_resource"
)

S1_SAMPLE_SIZE = 5000
SOURCE_SAMPLE_SIZE = 50000
MAX_NEGATIVES_PER_S1 = 15
CHUNK_SIZE = 200000
RANDOM_SEED = 42

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "matching_model.pkl"


# ============================================================
# DATA LOADING
# ============================================================

def load_source1(dataset_dir):
    path = Path(dataset_dir) / "dataset" / "train" / "train_source1.tsv"

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=S1_SAMPLE_SIZE,
    )

    return preprocess_dataframe(df)


def load_ground_truth(dataset_dir, source1_ids):
    path = Path(dataset_dir) / "dataset" / "train" / "train_ground_truth.tsv"

    required_ids = set(source1_ids)
    parts = []
    found = set()

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        matched = chunk[
            chunk["source1_entity_id"].isin(required_ids)
        ]

        if not matched.empty:
            parts.append(matched)
            found.update(matched["source1_entity_id"].tolist())

        if len(found) == len(required_ids):
            break

    if not parts:
        raise RuntimeError("No ground-truth rows found.")

    return pd.concat(parts, ignore_index=True)


def parse_matched_ids(value):
    if value is None:
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def build_ground_truth_map(ground_truth):
    result = {}

    for _, row in ground_truth.iterrows():
        result[row["source1_entity_id"]] = set(
            parse_matched_ids(row["matched_entity_ids"])
        )

    return result


def infer_entity_prefix(path):
    sample = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id"],
        nrows=1,
    )

    entity_id = str(sample.iloc[0]["entity_id"])

    # Examples:
    # S2_12345 -> S2
    # source2_12345 -> source2
    prefix = re.split(r"[^A-Za-z0-9]+", entity_id)[0]

    return prefix


def load_source_subset(path, required_ids, max_random_rows):
    """
    Load:
      - all required positive IDs
      - a manageable sample of other records

    This avoids loading the entire 5M+ row source into memory
    during model training.
    """

    required_ids = set(required_ids)

    random_parts = []
    true_parts = []

    random_count = 0
    found_required = set()

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:

        if required_ids:
            true_rows = chunk[
                chunk["entity_id"].isin(required_ids)
            ]

            if not true_rows.empty:
                true_parts.append(true_rows)
                found_required.update(
                    true_rows["entity_id"].tolist()
                )

            non_true = chunk[
                ~chunk["entity_id"].isin(required_ids)
            ]
        else:
            non_true = chunk

        if random_count < max_random_rows:

            remaining = max_random_rows - random_count

            sample_rows = non_true.head(remaining)

            if not sample_rows.empty:
                random_parts.append(sample_rows)
                random_count += len(sample_rows)

        if (
            random_count >= max_random_rows
            and found_required >= required_ids
        ):
            break

    parts = []

    if random_parts:
        parts.append(pd.concat(random_parts, ignore_index=True))

    if true_parts:
        parts.append(pd.concat(true_parts, ignore_index=True))

    if not parts:
        raise RuntimeError(f"No rows loaded from {path}")

    result = pd.concat(parts, ignore_index=True)

    result = result.drop_duplicates(
        subset=["entity_id"]
    ).reset_index(drop=True)

    return preprocess_dataframe(result)


# ============================================================
# TRAINING PAIRS
# ============================================================

def build_selected_pairs(
    candidate_pairs,
    source1_df,
    source2_df,
    source3_df,
    ground_truth_map,
):
    """
    Keep:
      - all available positive candidate pairs
      - missing true pairs explicitly
      - limited negative candidates per Source1

    This keeps training data manageable.
    """

    source_lookup = pd.concat(
        [
            source2_df,
            source3_df,
        ],
        ignore_index=True,
    )

    source_lookup = (
        source_lookup
        .drop_duplicates("entity_id")
        .set_index("entity_id")
    )

    grouped = {
        entity_id: group
        for entity_id, group
        in candidate_pairs.groupby(
            "source1_entity_id",
            sort=False,
        )
    }

    output = []

    for source1_id in source1_df["entity_id"]:

        group = grouped.get(
            source1_id,
            pd.DataFrame(
                columns=[
                    "source1_entity_id",
                    "candidate_entity_id",
                    "candidate_source",
                ]
            ),
        )

        true_ids = ground_truth_map.get(
            source1_id,
            set(),
        )

        # ----------------------------------------------------
        # Positive candidates already found by blocking
        # ----------------------------------------------------

        if not group.empty:

            positives = group[
                group["candidate_entity_id"].isin(true_ids)
            ]

            for _, row in positives.iterrows():

                output.append(
                    {
                        "source1_entity_id": source1_id,
                        "candidate_entity_id": row[
                            "candidate_entity_id"
                        ],
                        "candidate_source": row[
                            "candidate_source"
                        ],
                        "label": 1,
                    }
                )

        # ----------------------------------------------------
        # Add true pairs missed by candidate generation
        # ----------------------------------------------------

        existing_positive_ids = set()

        if not group.empty:
            existing_positive_ids = set(
                group[
                    group["candidate_entity_id"].isin(true_ids)
                ]["candidate_entity_id"]
            )

        for true_id in true_ids:

            if true_id in existing_positive_ids:
                continue

            if true_id not in source_lookup.index:
                continue

            if true_id in source2_df["entity_id"].values:
                candidate_source = "source2"
            else:
                candidate_source = "source3"

            output.append(
                {
                    "source1_entity_id": source1_id,
                    "candidate_entity_id": true_id,
                    "candidate_source": candidate_source,
                    "label": 1,
                }
            )

        # ----------------------------------------------------
        # Negative candidates
        # ----------------------------------------------------

        if not group.empty:

            negatives = group[
                ~group["candidate_entity_id"].isin(true_ids)
            ].head(MAX_NEGATIVES_PER_S1)

            for _, row in negatives.iterrows():

                output.append(
                    {
                        "source1_entity_id": source1_id,
                        "candidate_entity_id": row[
                            "candidate_entity_id"
                        ],
                        "candidate_source": row[
                            "candidate_source"
                        ],
                        "label": 0,
                    }
                )

    return pd.DataFrame(output)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_training_features(
    pairs,
    source1_df,
    source2_df,
    source3_df,
):
    source1_lookup = (
        source1_df
        .drop_duplicates("entity_id")
        .set_index("entity_id")
    )

    source_lookup = pd.concat(
        [
            source2_df,
            source3_df,
        ],
        ignore_index=True,
    )

    source_lookup = (
        source_lookup
        .drop_duplicates("entity_id")
        .set_index("entity_id")
    )

    feature_rows = []

    for i, row in pairs.iterrows():

        source1_id = row["source1_entity_id"]
        candidate_id = row["candidate_entity_id"]

        source1_row = source1_lookup.loc[source1_id]
        candidate_row = source_lookup.loc[candidate_id]

        features = extract_features(
            source1_row,
            candidate_row,
        )

        feature_rows.append(features)

        if (i + 1) % 5000 == 0:
            print(
                f"Feature extraction: "
                f"{i + 1}/{len(pairs)}"
            )

    return pd.DataFrame(feature_rows)


# ============================================================
# F0.5
# ============================================================

def calculate_f05(
    pairs,
    probabilities,
    threshold,
    validation_ids,
    ground_truth_map,
):
    temp = pairs.copy()

    temp["probability"] = probabilities

    predicted = temp[
        temp["probability"] >= threshold
    ]

    predicted_map = {}

    for source1_id, group in predicted.groupby(
        "source1_entity_id"
    ):
        predicted_map[source1_id] = set(
            group["candidate_entity_id"]
        )

    scores = []

    for source1_id in validation_ids:

        actual = ground_truth_map.get(
            source1_id,
            set(),
        )

        pred = predicted_map.get(
            source1_id,
            set(),
        )

        tp = len(actual & pred)
        fp = len(pred - actual)
        fn = len(actual - pred)

        # Correct singleton prediction:
        # no actual match and no predicted match.
        if tp == 0 and fp == 0 and fn == 0:
            score = 1.0

        else:

            denominator = (
                1.25 * tp
                + 0.25 * fn
                + fp
            )

            if denominator == 0:
                score = 0.0
            else:
                score = (
                    1.25 * tp
                    / denominator
                )

        scores.append(score)

    if not scores:
        return 0.0

    return float(np.mean(scores))


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE - MATCHING MODEL TRAINING")
    print("=" * 70)

    dataset_dir = os.environ.get(
        "AMAZON_ML_DATASET",
        DEFAULT_DATASET_DIR,
    )

    dataset_dir = Path(dataset_dir)

    print(f"\nDataset: {dataset_dir}")

    train_dir = dataset_dir / "dataset" / "train"

    source1_path = train_dir / "train_source1.tsv"
    source2_path = train_dir / "train_source2.tsv"
    source3_path = train_dir / "train_source3.tsv"

    # --------------------------------------------------------
    # 1. Load Source1
    # --------------------------------------------------------

    print("\n[1/7] Loading Source1...")

    source1 = load_source1(dataset_dir)

    print(
        f"Source1 training sample: "
        f"{len(source1):,}"
    )

    # --------------------------------------------------------
    # 2. Ground truth
    # --------------------------------------------------------

    print("\n[2/7] Loading ground truth...")

    ground_truth = load_ground_truth(
        dataset_dir,
        source1["entity_id"],
    )

    ground_truth_map = build_ground_truth_map(
        ground_truth
    )

    total_true = sum(
        len(v)
        for v in ground_truth_map.values()
    )

    print(
        f"Ground-truth rows: "
        f"{len(ground_truth):,}"
    )

    print(
        f"Known positive matches: "
        f"{total_true:,}"
    )

    # --------------------------------------------------------
    # 3. Train/validation split
    # --------------------------------------------------------

    print("\n[3/7] Creating train/validation split...")

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    shuffled_ids = np.array(
        source1["entity_id"]
    )

    rng.shuffle(shuffled_ids)

    split_point = int(
        len(shuffled_ids) * 0.80
    )

    train_ids = set(
        shuffled_ids[:split_point]
    )

    validation_ids = set(
        shuffled_ids[split_point:]
    )

    print(
        f"Train Source1 entities: "
        f"{len(train_ids):,}"
    )

    print(
        f"Validation Source1 entities: "
        f"{len(validation_ids):,}"
    )

    # --------------------------------------------------------
    # 4. Load manageable Source2/Source3 subsets
    # --------------------------------------------------------

    print(
        "\n[4/7] Loading Source2/Source3 "
        "training subsets..."
    )

    source2_prefix = infer_entity_prefix(
        source2_path
    )

    source3_prefix = infer_entity_prefix(
        source3_path
    )

    print(
        f"Detected Source2 prefix: "
        f"{source2_prefix}"
    )

    print(
        f"Detected Source3 prefix: "
        f"{source3_prefix}"
    )

    all_true_ids = set()

    for ids in ground_truth_map.values():
        all_true_ids.update(ids)

    source2_true_ids = {
        entity_id
        for entity_id in all_true_ids
        if entity_id.startswith(source2_prefix)
    }

    source3_true_ids = {
        entity_id
        for entity_id in all_true_ids
        if entity_id.startswith(source3_prefix)
    }

    print(
        f"Required Source2 positive IDs: "
        f"{len(source2_true_ids):,}"
    )

    print(
        f"Required Source3 positive IDs: "
        f"{len(source3_true_ids):,}"
    )

    source2 = load_source_subset(
        source2_path,
        source2_true_ids,
        SOURCE_SAMPLE_SIZE,
    )

    source3 = load_source_subset(
        source3_path,
        source3_true_ids,
        SOURCE_SAMPLE_SIZE,
    )

    print(
        f"Source2 subset: "
        f"{len(source2):,}"
    )

    print(
        f"Source3 subset: "
        f"{len(source3):,}"
    )

    # --------------------------------------------------------
    # 5. Candidate generation
    # --------------------------------------------------------

    print(
        "\n[5/7] Generating training candidates..."
    )

    candidate_pairs = generate_all_candidates(
        source1,
        source2,
        source3,
    )

    print(
        f"Raw candidate pairs: "
        f"{len(candidate_pairs):,}"
    )

    # Candidate blocking recall on available
    # ground-truth pairs.

    candidate_true_pairs = candidate_pairs[
        candidate_pairs.apply(
            lambda row: row["candidate_entity_id"]
            in ground_truth_map.get(
                row["source1_entity_id"],
                set(),
            ),
            axis=1,
        )
    ]

    blocking_true = len(
        candidate_true_pairs
    )

    print(
        f"Ground-truth pairs found by blocking: "
        f"{blocking_true:,}"
    )

    if total_true:
        print(
            f"Approximate blocking recall: "
            f"{blocking_true / total_true:.4f}"
        )

    # --------------------------------------------------------
    # 6. Build training pairs + features
    # --------------------------------------------------------

    print(
        "\n[6/7] Building labeled training pairs..."
    )

    pairs = build_selected_pairs(
        candidate_pairs,
        source1,
        source2,
        source3,
        ground_truth_map,
    )

    print(
        f"Selected training pairs: "
        f"{len(pairs):,}"
    )

    print(
        f"Positive pairs: "
        f"{int(pairs['label'].sum()):,}"
    )

    print(
        f"Negative pairs: "
        f"{int((pairs['label'] == 0).sum()):,}"
    )

    print("\nExtracting features...")

    X_all = extract_training_features(
        pairs,
        source1,
        source2,
        source3,
    )

    y_all = pairs["label"].astype(int)

    feature_names = list(
        X_all.columns
    )

    print(
        f"\nFeature count: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Train/validation split
    # --------------------------------------------------------

    train_mask = pairs[
        "source1_entity_id"
    ].isin(train_ids)

    validation_mask = pairs[
        "source1_entity_id"
    ].isin(validation_ids)

    X_train = X_all.loc[
        train_mask
    ].reset_index(drop=True)

    y_train = y_all.loc[
        train_mask
    ].reset_index(drop=True)

    X_validation = X_all.loc[
        validation_mask
    ].reset_index(drop=True)

    y_validation = y_all.loc[
        validation_mask
    ].reset_index(drop=True)

    validation_pairs = pairs.loc[
        validation_mask
    ].reset_index(drop=True)

    print(
        f"\nTraining pairs: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation pairs: "
        f"{len(X_validation):,}"
    )

    print(
        f"Training positives: "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"Validation positives: "
        f"{int(y_validation.sum()):,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nTraining Logistic Regression model..."
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    C=2.0,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    # --------------------------------------------------------
    # Threshold search
    # --------------------------------------------------------

    print(
        "\nThreshold evaluation:"
    )

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    best_threshold = 0.80
    best_score = -1.0

    for threshold in thresholds:

        score = calculate_f05(
            validation_pairs,
            validation_probabilities,
            threshold,
            validation_ids,
            ground_truth_map,
        )

        print(
            f"  threshold={threshold:.2f}"
            f"  F0.5={score:.6f}"
        )

        if score > best_score:

            best_score = score
            best_threshold = threshold

    print(
        "\nBest validation threshold:"
    )

    print(
        f"  threshold = {best_threshold:.2f}"
    )

    print(
        f"  F0.5      = {best_score:.6f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "threshold": best_threshold,
    }

    with open(
        MODEL_PATH,
        "wb",
    ) as file:

        pickle.dump(
            artifact,
            file,
        )

    print(
        f"\nModel saved to: "
        f"{MODEL_PATH}"
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()