import os
import pandas as pd
import numpy as np
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

DATASET = r"C:\Users\Acer\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
TRAIN = os.path.join(DATASET, "train")

S1_FILE = os.path.join(TRAIN, "train_source1.tsv")
S2_FILE = os.path.join(TRAIN, "train_source2.tsv")
S3_FILE = os.path.join(TRAIN, "train_source3.tsv")
GT_FILE = os.path.join(TRAIN, "train_ground_truth.tsv")

OUTPUT = os.path.join("output", "overnight_validation_results.txt")

# Validate on a random subset of Source1.
# Large enough to be useful, small enough to finish overnight.
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace("&", " and ", regex=False)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def prepare(df):
    df = df.copy()

    df["name_norm"] = normalize_text(df["business_name"])
    df["address_norm"] = normalize_text(df["business_address"])
    df["country_norm"] = normalize_text(df["country"])

    df["name_key"] = (
        df["country_norm"] + "||" + df["name_norm"]
    )

    df["address_key"] = (
        df["country_norm"] + "||" + df["address_norm"]
    )

    return df


# ============================================================
# F0.5
# ============================================================

def f05(precision, recall):
    if precision + recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("OVERNIGHT TRAIN VALIDATION")
print("=" * 70)

print("\nLoading TRAIN Source 1...")
s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str)

print("Source1 rows:", len(s1))

print("\nLoading ground truth...")
gt = pd.read_csv(GT_FILE, sep="\t", dtype=str)

print("Ground truth rows:", len(gt))

# ------------------------------------------------------------
# SAMPLE SOURCE 1
# ------------------------------------------------------------

if SAMPLE_SIZE < len(s1):
    s1 = s1.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).copy()

print("\nValidation Source1 sample:", len(s1))

s1 = prepare(s1)

# ------------------------------------------------------------
# BUILD GROUND TRUTH SET
# ------------------------------------------------------------

print("\nPreparing ground truth...")

gt_sample = gt[
    gt["source1_entity_id"].isin(
        set(s1["entity_id"])
    )
].copy()

truth = defaultdict(set)

for row in gt_sample.itertuples(index=False):
    if pd.isna(row.matched_entity_ids):
        continue

    ids = str(row.matched_entity_ids).strip()

    if not ids:
        continue

    for entity_id in ids.split(","):
        entity_id = entity_id.strip()

        if entity_id:
            truth[row.source1_entity_id].add(entity_id)

print(
    "Ground-truth Source1 entities:",
    len(truth)
)

print(
    "Ground-truth matched pairs:",
    sum(len(v) for v in truth.values())
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_source(source_file, source_name):

    print("\n" + "=" * 70)
    print("EVALUATING", source_name)
    print("=" * 70)

    print("\nLoading source...")
    source = pd.read_csv(
        source_file,
        sep="\t",
        dtype=str
    )

    print("Rows:", len(source))

    source = prepare(source)

    # --------------------------------------------------------
    # Global uniqueness counts
    # --------------------------------------------------------

    print("\nCalculating uniqueness...")

    source_name_count = (
        source["name_key"]
        .value_counts()
    )

    source_address_count = (
        source["address_key"]
        .value_counts()
    )

    s1_name_count = (
        s1["name_key"]
        .value_counts()
    )

    s1_address_count = (
        s1["address_key"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Candidate generation using vectorized MERGE
    # --------------------------------------------------------

    print("\nBuilding exact candidate pairs...")

    s1_keys = s1[
        [
            "entity_id",
            "name_key",
            "address_key"
        ]
    ].rename(
        columns={
            "entity_id": "source1_entity_id"
        }
    )

    source_keys = source[
        [
            "entity_id",
            "name_key",
            "address_key"
        ]
    ].rename(
        columns={
            "entity_id": "candidate_entity_id"
        }
    )

    # NAME candidates
    name_candidates = s1_keys[
        ["source1_entity_id", "name_key"]
    ].merge(
        source_keys[
            ["candidate_entity_id", "name_key"]
        ],
        on="name_key",
        how="inner"
    )

    print(
        "Name candidates:",
        len(name_candidates)
    )

    # ADDRESS candidates
    address_candidates = s1_keys[
        ["source1_entity_id", "address_key"]
    ].merge(
        source_keys[
            ["candidate_entity_id", "address_key"]
        ],
        on="address_key",
        how="inner"
    )

    print(
        "Address candidates:",
        len(address_candidates)
    )

    # Combine candidate IDs
    candidates = pd.concat(
        [
            name_candidates[
                [
                    "source1_entity_id",
                    "candidate_entity_id"
                ]
            ],
            address_candidates[
                [
                    "source1_entity_id",
                    "candidate_entity_id"
                ]
            ]
        ],
        ignore_index=True
    ).drop_duplicates()

    print(
        "Unique candidate pairs:",
        len(candidates)
    )

    # --------------------------------------------------------
    # Add keys to candidates
    # --------------------------------------------------------

    candidates = candidates.merge(
        s1_keys.rename(
            columns={
                "name_key": "s1_name_key",
                "address_key": "s1_address_key"
            }
        ),
        on="source1_entity_id",
        how="left"
    )

    candidates = candidates.merge(
        source_keys.rename(
            columns={
                "name_key": "src_name_key",
                "address_key": "src_address_key"
            }
        ),
        on="candidate_entity_id",
        how="left"
    )

    # --------------------------------------------------------
    # Candidate signals
    # --------------------------------------------------------

    candidates["same_name"] = (
        candidates["s1_name_key"]
        == candidates["src_name_key"]
    )

    candidates["same_address"] = (
        candidates["s1_address_key"]
        == candidates["src_address_key"]
    )

    candidates["strong_both"] = (
        candidates["same_name"]
        & candidates["same_address"]
    )

    candidates["unique_name"] = (
        candidates["same_name"]
        &
        (
            candidates["s1_name_key"]
            .map(s1_name_count)
            .fillna(0)
            == 1
        )
        &
        (
            candidates["src_name_key"]
            .map(source_name_count)
            .fillna(0)
            == 1
        )
    )

    candidates["unique_address"] = (
        candidates["same_address"]
        &
        (
            candidates["s1_address_key"]
            .map(s1_address_count)
            .fillna(0)
            == 1
        )
        &
        (
            candidates["src_address_key"]
            .map(source_address_count)
            .fillna(0)
            == 1
        )
    )

    # --------------------------------------------------------
    # Rules
    # --------------------------------------------------------

    rules = {

        "RULE_1_STRONG_BOTH":
            candidates["strong_both"],

        "RULE_2_UNIQUE_NAME":
            candidates["unique_name"],

        "RULE_3_UNIQUE_ADDRESS":
            candidates["unique_address"],

        "RULE_4_UNIQUE_NAME_OR_ADDRESS":
            (
                candidates["unique_name"]
                | candidates["unique_address"]
            ),

        "RULE_5_STRONG_BOTH_OR_UNIQUE_ONE":
            (
                candidates["strong_both"]
                |
                candidates["unique_name"]
                |
                candidates["unique_address"]
            ),
    }

    # --------------------------------------------------------
    # Evaluate every rule
    # --------------------------------------------------------

    results = []

    for rule_name, mask in rules.items():

        selected = candidates.loc[
            mask,
            [
                "source1_entity_id",
                "candidate_entity_id"
            ]
        ]

        predicted = defaultdict(set)

        for row in selected.itertuples(index=False):

            predicted[
                row.source1_entity_id
            ].add(
                row.candidate_entity_id
            )

        tp = 0
        fp = 0
        fn = 0

        # Evaluate every sampled S1 entity,
        # including entities with zero predictions.
        for entity_id in s1["entity_id"]:

            actual = truth.get(
                entity_id,
                set()
            )

            pred = predicted.get(
                entity_id,
                set()
            )

            tp += len(actual & pred)
            fp += len(pred - actual)
            fn += len(actual - pred)

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0
        )

        score = f05(
            precision,
            recall
        )

        results.append(
            {
                "rule": rule_name,
                "predicted_pairs": len(selected),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "F0.5": score,
            }
        )

    return results


# ============================================================
# RUN SOURCE 2 + SOURCE 3
# ============================================================

all_results = []

results2 = evaluate_source(
    S2_FILE,
    "SOURCE 2"
)

for r in results2:
    r["source"] = "source2"
    all_results.append(r)

results3 = evaluate_source(
    S3_FILE,
    "SOURCE 3"
)

for r in results3:
    r["source"] = "source3"
    all_results.append(r)


# ============================================================
# DISPLAY RESULTS
# ============================================================

result_df = pd.DataFrame(all_results)

result_df = result_df[
    [
        "source",
        "rule",
        "predicted_pairs",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "F0.5"
    ]
]

print("\n")
print("=" * 70)
print("FINAL VALIDATION RESULTS")
print("=" * 70)

print(
    result_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "OVERNIGHT TRAIN VALIDATION RESULTS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        result_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    f.write("\n\n")

    f.write(
        "NOTE: Validation uses a random "
        f"{SAMPLE_SIZE:,}-entity Source1 sample.\n"
    )

print("\nResults saved to:")
print(OUTPUT)

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)