import os
import pandas as pd


DATASET_DIR = (
    r"C:\Users\Acer\Downloads"
    r"\6ab10eb3b23ba_student_resource"
    r"\student_resource"
)

TEST_DIR = os.path.join(DATASET_DIR, "dataset", "test")
OUTPUT_DIR = "output"


def normalize(series):
    """Fast vectorized normalization matching project preprocessing."""
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

    df["name_norm"] = normalize(df["business_name"])
    df["address_norm"] = normalize(df["business_address"])
    df["country_norm"] = normalize(df["country"])

    return df


def generate_exact_candidates(s1, source, source_name):
    """Generate exact normalized name/address candidates."""

    # -----------------------------
    # Exact name + country
    # -----------------------------

    left = s1[
        ["entity_id", "name_norm", "country_norm"]
    ]

    right = source[
        ["entity_id", "name_norm", "country_norm"]
    ]

    name_matches = left.merge(
        right,
        on=["name_norm", "country_norm"],
        how="inner",
        suffixes=("_s1", "_candidate"),
    )

    name_matches = name_matches[
        name_matches["name_norm"].ne("")
    ]

    name_candidates = name_matches[
        [
            "entity_id_s1",
            "entity_id_candidate",
        ]
    ].rename(
        columns={
            "entity_id_s1": "source1_entity_id",
            "entity_id_candidate": "candidate_entity_id",
        }
    )

    name_candidates["candidate_source"] = source_name

    # -----------------------------
    # Exact address + country
    # -----------------------------

    left = s1[
        ["entity_id", "address_norm", "country_norm"]
    ]

    right = source[
        ["entity_id", "address_norm", "country_norm"]
    ]

    address_matches = left.merge(
        right,
        on=["address_norm", "country_norm"],
        how="inner",
        suffixes=("_s1", "_candidate"),
    )

    address_matches = address_matches[
        address_matches["address_norm"].ne("")
    ]

    address_candidates = address_matches[
        [
            "entity_id_s1",
            "entity_id_candidate",
        ]
    ].rename(
        columns={
            "entity_id_s1": "source1_entity_id",
            "entity_id_candidate": "candidate_entity_id",
        }
    )

    address_candidates["candidate_source"] = source_name

    # -----------------------------
    # Union
    # -----------------------------

    candidates = pd.concat(
        [
            name_candidates,
            address_candidates,
        ],
        ignore_index=True,
    )

    return candidates.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
        ]
    )


def main():

    print("=" * 60)
    print("AMAZON ML - FAST SUBMISSION GENERATION")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("\n[1/5] Loading test data...")

    s1 = pd.read_csv(
        os.path.join(
            TEST_DIR,
            "test_source1.tsv",
        ),
        sep="\t",
        dtype=str,
    )

    s2 = pd.read_csv(
        os.path.join(
            TEST_DIR,
            "test_source2.tsv",
        ),
        sep="\t",
        dtype=str,
    )

    s3 = pd.read_csv(
        os.path.join(
            TEST_DIR,
            "test_source3.tsv",
        ),
        sep="\t",
        dtype=str,
    )

    print(f"Source1: {len(s1):,}")
    print(f"Source2: {len(s2):,}")
    print(f"Source3: {len(s3):,}")

    print("\n[2/5] Normalizing...")

    s1 = prepare(s1)
    s2 = prepare(s2)
    s3 = prepare(s3)

    print("Normalization complete.")

    print("\n[3/5] Generating Source2 candidates...")

    candidates_s2 = generate_exact_candidates(
        s1,
        s2,
        "source2",
    )

    print(
        f"Source2 candidates: "
        f"{len(candidates_s2):,}"
    )

    print("\nGenerating Source3 candidates...")

    candidates_s3 = generate_exact_candidates(
        s1,
        s3,
        "source3",
    )

    print(
        f"Source3 candidates: "
        f"{len(candidates_s3):,}"
    )

    candidates = pd.concat(
        [
            candidates_s2,
            candidates_s3,
        ],
        ignore_index=True,
    )

    candidates = candidates.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
        ]
    )

    print(
        f"\nTotal candidates: "
        f"{len(candidates):,}"
    )

    print("\n[4/5] Writing candidate_pairs.tsv...")

    candidate_path = os.path.join(
        OUTPUT_DIR,
        "candidate_pairs.tsv",
    )

    candidates[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
        ]
    ].to_csv(
        candidate_path,
        sep="\t",
        index=False,
    )

    print(f"Saved: {candidate_path}")

    print("\n[5/5] Writing matching_results.tsv...")

    grouped = (
        candidates
        .groupby(
            "source1_entity_id",
            sort=False,
        )["candidate_entity_id"]
        .agg(
            lambda x: ",".join(
                pd.unique(x)
            )
        )
    )

    results = pd.DataFrame(
        {
            "source1_entity_id":
                s1["entity_id"]
        }
    )

    results["matched_entity_ids"] = (
        results["source1_entity_id"]
        .map(grouped)
        .fillna("")
    )

    matching_path = os.path.join(
        OUTPUT_DIR,
        "matching_results.tsv",
    )

    results.to_csv(
        matching_path,
        sep="\t",
        index=False,
    )

    print(f"Saved: {matching_path}")

    print("\n" + "=" * 60)
    print("SUBMISSION FILES GENERATED")
    print("=" * 60)


if __name__ == "__main__":
    main()