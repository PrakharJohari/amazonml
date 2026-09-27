import os
import sys
import pickle
import argparse

import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Allow imports when running:
# python src\predict.py
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import preprocess_dataframe
from candidate_generation import (
    _build_indexes,
    _get_candidates_for_row,
)
from feature_engineering import extract_features


DATASET_DIR = (
    r"C:\Users\Acer\Downloads"
    r"\6ab10eb3b23ba_student_resource"
    r"\student_resource"
)

MODEL_PATH = os.path.join(
    "models",
    "matching_model.pkl",
)

BATCH_SIZE = 5000
OUTPUT_DIR = "output"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    feature_names = saved["feature_names"]
    threshold = saved["threshold"]

    return model, feature_names, threshold


def generate_indexed_candidates(
    source1_batch,
    source2_df,
    source3_df,
    source2_indexes,
    source3_indexes,
):
    """
    Generate candidates for an S1 batch using indexes
    that were already built once for Source2 and Source3.
    """

    (
        source2_name_frequency,
        source2_address_frequency,
        source2_name_index,
        source2_address_index,
        source2_exact_name_index,
        source2_exact_address_index,
    ) = source2_indexes

    (
        source3_name_frequency,
        source3_address_frequency,
        source3_name_index,
        source3_address_index,
        source3_exact_name_index,
        source3_exact_address_index,
    ) = source3_indexes

    # Fast NumPy arrays instead of repeated DataFrame.iloc
    source2_ids = source2_df["entity_id"].to_numpy()
    source3_ids = source3_df["entity_id"].to_numpy()

    candidate_rows = []

    for row in source1_batch.itertuples(index=False):

        # -----------------------------
        # Source 2 candidates
        # -----------------------------

        candidates_s2 = _get_candidates_for_row(
            row,
            source2_name_frequency,
            source2_address_frequency,
            source2_name_index,
            source2_address_index,
            source2_exact_name_index,
            source2_exact_address_index,
        )

        for candidate_idx in candidates_s2:
            candidate_rows.append(
                (
                    row.entity_id,
                    source2_ids[candidate_idx],
                    "source2",
                    candidate_idx,
                )
            )

        # -----------------------------
        # Source 3 candidates
        # -----------------------------

        candidates_s3 = _get_candidates_for_row(
            row,
            source3_name_frequency,
            source3_address_frequency,
            source3_name_index,
            source3_address_index,
            source3_exact_name_index,
            source3_exact_address_index,
        )

        for candidate_idx in candidates_s3:
            candidate_rows.append(
                (
                    row.entity_id,
                    source3_ids[candidate_idx],
                    "source3",
                    candidate_idx,
                )
            )

    return candidate_rows

def build_feature_rows(
    source1_batch,
    candidate_rows,
    source2_df,
    source3_df,
):
    """
    Build model feature rows using parallel feature extraction.
    Candidate generation, features, model and threshold remain unchanged.
    """

    source1_lookup = {
        row["entity_id"]: row
        for row in source1_batch.to_dict("records")
    }

    source2_arrays = {
        column: source2_df[column].to_numpy(copy=False)
        for column in source2_df.columns
    }

    source3_arrays = {
        column: source3_df[column].to_numpy(copy=False)
        for column in source3_df.columns
    }

    source2_columns = list(source2_df.columns)
    source3_columns = list(source3_df.columns)

    def process_chunk(chunk):
        local_features = []
        local_metadata = []

        for (
            source1_id,
            candidate_id,
            candidate_source,
            candidate_idx,
        ) in chunk:

            source1_row = source1_lookup[source1_id]

            if candidate_source == "source2":
                candidate_row = {
                    column: source2_arrays[column][candidate_idx]
                    for column in source2_columns
                }
            else:
                candidate_row = {
                    column: source3_arrays[column][candidate_idx]
                    for column in source3_columns
                }

            features = extract_features(
                source1_row,
                candidate_row,
            )

            local_features.append(features)

            local_metadata.append(
                (
                    source1_id,
                    candidate_id,
                    candidate_source,
                )
            )

        return local_features, local_metadata

    # Split candidates into reasonably large chunks.
    num_workers = min(8, os.cpu_count() or 4)

    chunk_size = max(
        5000,
        len(candidate_rows) // (num_workers * 2),
    )

    chunks = [
        candidate_rows[i:i + chunk_size]
        for i in range(0, len(candidate_rows), chunk_size)
    ]

    feature_rows = []
    metadata_rows = []

    with ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:

        results = executor.map(
            process_chunk,
            chunks,
        )

        for local_features, local_metadata in results:
            feature_rows.extend(local_features)
            metadata_rows.extend(local_metadata)

    return feature_rows, metadata_rows


def main():

    # --------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge test prediction"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Number of Source1 rows to process. "
            "If omitted, processes the full test set."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    print("=" * 60)
    print("AMAZON ML CHALLENGE - TEST PREDICTION")
    print("=" * 60)

    if args.sample is not None:
        print(
            f"Sample mode enabled: "
            f"{args.sample:,} Source1 rows"
        )
    else:
        print("Full test mode enabled.")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------
    # 1. Load model
    # --------------------------------------------------

    print("\n[1/6] Loading trained model...")

    model, feature_names, threshold = load_model()

    print(
        f"Decision threshold: {threshold}"
    )

    print(
        f"Expected features: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------
    # 2. Load test data
    # --------------------------------------------------

    print("\n[2/6] Loading test data...")

    s1_path = os.path.join(
        DATASET_DIR,
        "dataset",
        "test",
        "test_source1.tsv",
    )

    s2_path = os.path.join(
        DATASET_DIR,
        "dataset",
        "test",
        "test_source2.tsv",
    )

    s3_path = os.path.join(
        DATASET_DIR,
        "dataset",
        "test",
        "test_source3.tsv",
    )

    source1 = pd.read_csv(
        s1_path,
        sep="\t",
    )

    source2 = pd.read_csv(
        s2_path,
        sep="\t",
    )

    source3 = pd.read_csv(
        s3_path,
        sep="\t",
    )

    print(
        f"Source1: {len(source1):,}"
    )

    print(
        f"Source2: {len(source2):,}"
    )

    print(
        f"Source3: {len(source3):,}"
    )

    # --------------------------------------------------
    # Apply sample ONLY to Source1
    # --------------------------------------------------

    if args.sample is not None:

        if args.sample <= 0:
            raise ValueError(
                "--sample must be greater than 0."
            )

        source1 = source1.head(
            args.sample
        ).copy()

        print(
            f"Using Source1 sample: "
            f"{len(source1):,}"
        )

    # --------------------------------------------------
    # 3. Preprocess ONCE
    # --------------------------------------------------

    print(
        "\n[3/6] Preprocessing sources once..."
    )

    source1 = preprocess_dataframe(
        source1
    )

    source2 = preprocess_dataframe(
        source2
    )

    source3 = preprocess_dataframe(
        source3
    )

    print(
        "Preprocessing complete."
    )

    # --------------------------------------------------
    # 4. Build indexes ONCE
    # --------------------------------------------------

    print(
        "\n[4/6] Building Source2 and Source3 indexes..."
    )

    print(
        "This may take some time, "
        "but it happens only once."
    )

    source2_indexes = _build_indexes(
        source2
    )

    print(
        "Source2 indexes ready."
    )

    source3_indexes = _build_indexes(
        source3
    )

    print(
        "Source3 indexes ready."
    )

    # --------------------------------------------------
    # 5. Generate candidates + predictions
    # --------------------------------------------------

    print(
        "\n[5/6] Generating candidates + predictions..."
    )

    candidate_output = os.path.join(
        OUTPUT_DIR,
        "candidate_pairs.tsv",
    )

    prediction_output = os.path.join(
        OUTPUT_DIR,
        "predictions.tsv",
    )

    # Remove previous incomplete outputs
    if os.path.exists(
        candidate_output
    ):
        os.remove(
            candidate_output
        )

    if os.path.exists(
        prediction_output
    ):
        os.remove(
            prediction_output
        )

    total_candidates = 0
    total_matches = 0

    num_batches = (
        len(source1)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    candidate_header_written = False
    prediction_header_written = False

    for batch_number, start in enumerate(
        range(
            0,
            len(source1),
            BATCH_SIZE,
        ),
        start=1,
    ):

        end = min(
            start + BATCH_SIZE,
            len(source1),
        )

        source1_batch = source1.iloc[
            start:end
        ].copy()

        print(
            f"Batch {batch_number}/{num_batches} "
            f"| S1 rows {start:,}-{end:,}"
        )

        # ------------------------------------------
        # Candidate generation
        # ------------------------------------------

        candidate_rows = (
            generate_indexed_candidates(
                source1_batch,
                source2,
                source3,
                source2_indexes,
                source3_indexes,
            )
        )

        batch_candidate_count = len(
            candidate_rows
        )

        total_candidates += (
            batch_candidate_count
        )

        print(
            f"  Candidates: "
            f"{batch_candidate_count:,}"
        )

        if not candidate_rows:
            continue

        # ------------------------------------------
        # Feature extraction
        # ------------------------------------------

        feature_rows, metadata_rows = (
            build_feature_rows(
                source1_batch,
                candidate_rows,
                source2,
                source3,
            )
        )

        print(
            f"  Features: "
            f"{len(feature_rows):,}"
        )

        feature_df = pd.DataFrame(
            feature_rows
        )

        feature_df = feature_df[
            feature_names
        ]

        # ------------------------------------------
        # Model prediction
        # ------------------------------------------

        probabilities = (
            model.predict_proba(
                feature_df
            )[:, 1]
        )

        predictions = (
            probabilities >= threshold
        )

        # ------------------------------------------
        # Candidate output
        # ------------------------------------------

        candidate_df = pd.DataFrame(
            [
                {
                    "source1_entity_id": row[0],
                    "candidate_entity_id": row[1],
                    "candidate_source": row[2],
                }
                for row in metadata_rows
            ]
        )

        candidate_df.to_csv(
            candidate_output,
            sep="\t",
            index=False,
            mode="a",
            header=(
                not candidate_header_written
            ),
        )

        candidate_header_written = True

        # ------------------------------------------
        # Prediction output
        # ------------------------------------------

        prediction_rows = []

        for (
            metadata,
            probability,
            is_match,
        ) in zip(
            metadata_rows,
            probabilities,
            predictions,
        ):

            if is_match:

                prediction_rows.append(
                    {
                        "source1_entity_id": metadata[0],
                        "matched_entity_id": metadata[1],
                        "matched_source": metadata[2],
                        "score": float(
                            probability
                        ),
                    }
                )

        if prediction_rows:

            prediction_df = pd.DataFrame(
                prediction_rows
            )

            prediction_df.to_csv(
                prediction_output,
                sep="\t",
                index=False,
                mode="a",
                header=(
                    not prediction_header_written
                ),
            )

            prediction_header_written = True

            total_matches += len(
                prediction_rows
            )

        print(
            f"  Matches: "
            f"{len(prediction_rows):,}"
        )

    # --------------------------------------------------
    # 6. Summary
    # --------------------------------------------------

    print(
        "\n[6/6] Prediction complete."
    )

    print("-" * 60)

    print(
        f"Total candidates: "
        f"{total_candidates:,}"
    )

    print(
        f"Total predicted matches: "
        f"{total_matches:,}"
    )

    print(
        f"Candidate file: "
        f"{candidate_output}"
    )

    print(
        f"Prediction file: "
        f"{prediction_output}"
    )

    print("-" * 60)


if __name__ == "__main__":
    main()

