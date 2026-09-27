import os
import re
import sqlite3
import time
import gc

import joblib
import pandas as pd
from rapidfuzz import fuzz


# ============================================================
# CONFIG
# ============================================================

DB_PATH = r"output\entity_index.db"
CANDIDATE_FILE = r"output\candidate_pairs.tsv"
OUTPUT_FILE = r"output\matching_results_overnight.tsv"
MODEL_FILE = r"models\matching_model.pkl"

# Process candidates in chunks so RAM stays under control.
CANDIDATE_CHUNK = 50_000

# SQLite variable safety limit.
SQL_BATCH = 900


# ============================================================
# BUSINESS SUFFIXES
# ============================================================

BUSINESS_SUFFIXES = {
    "llc",
    "inc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "company",
    "co",
    "plc",
    "llp",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):
    """
    Normalize text consistently for matching.
    """

    if text is None:
        return ""

    text = str(text).casefold()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_tokens(text):
    """
    Return unique normalized tokens.
    """

    if not text:
        return set()

    return set(text.split())


def get_numbers(text):
    """
    Extract numeric components from an address.
    """

    if not text:
        return set()

    return set(
        re.findall(
            r"\d+",
            text
        )
    )


# ============================================================
# SIMILARITY HELPERS
# ============================================================

def jaccard(set1, set2):

    if not set1 or not set2:
        return 0.0

    union = set1 | set2

    if not union:
        return 0.0

    return len(set1 & set2) / len(union)


def number_similarity(set1, set2):

    if not set1 or not set2:
        return 0.0

    union = set1 | set2

    if not union:
        return 0.0

    return len(set1 & set2) / len(union)


def suffix_match(name1, name2):
    """
    Match business legal suffixes.

    Same logic as the trained feature engineering.
    """

    suffixes1 = {
        token
        for token in name1.split()
        if token in BUSINESS_SUFFIXES
    }

    suffixes2 = {
        token
        for token in name2.split()
        if token in BUSINESS_SUFFIXES
    }

    # Neither business has a suffix.
    if not suffixes1 and not suffixes2:
        return 1

    # Same suffix set.
    if suffixes1 == suffixes2:
        return 1

    return 0


# ============================================================
# PREPARE DATABASE RECORD
# ============================================================

def prepare_record(
    business_name,
    business_address,
    country
):

    name = normalize(
        business_name
    )

    address = normalize(
        business_address
    )

    country = normalize(
        country
    )

    return {
        "business_name": name,
        "business_address": address,
        "country": country,

        "_name_tokens":
            get_tokens(name),

        "_address_tokens":
            get_tokens(address),

        "_numbers":
            get_numbers(address),
    }


# ============================================================
# SQLITE LOOKUP
# ============================================================

def fetch_records(
    conn,
    ids,
    source
):

    if not ids:
        return {}

    ids = list(ids)

    result = {}

    for start in range(
        0,
        len(ids),
        SQL_BATCH
    ):

        batch = ids[
            start:
            start + SQL_BATCH
        ]

        placeholders = ",".join(
            ["?"] * len(batch)
        )

        query = f"""
            SELECT
                entity_id,
                business_name,
                business_address,
                country
            FROM entities
            WHERE source = ?
            AND entity_id IN ({placeholders})
        """

        rows = conn.execute(
            query,
            [source] + batch
        ).fetchall()

        for (
            entity_id,
            business_name,
            business_address,
            country
        ) in rows:

            result[entity_id] = prepare_record(
                business_name,
                business_address,
                country
            )

    return result


# ============================================================
# CHEAP FILTER
# ============================================================

def cheap_filter(
    source1,
    candidate
):
    """
    Fast rejection step.

    We keep candidates when there is:
      - country agreement
      - name token overlap
      - address token overlap
      - number overlap
      - reasonable fuzzy name similarity
    """

    country1 = source1["country"]
    country2 = candidate["country"]

    # Country mismatch should not match.
    if (
        country1
        and country2
        and country1 != country2
    ):
        return False

    name_overlap = bool(
        source1["_name_tokens"]
        &
        candidate["_name_tokens"]
    )

    address_overlap = bool(
        source1["_address_tokens"]
        &
        candidate["_address_tokens"]
    )

    number_overlap = bool(
        source1["_numbers"]
        &
        candidate["_numbers"]
    )

    if name_overlap:
        return True

    if address_overlap:
        return True

    if number_overlap:
        return True

    # Fuzzy rescue for spelling variations.
    name1 = source1["business_name"]
    name2 = candidate["business_name"]

    if name1 and name2:

        if fuzz.ratio(
            name1,
            name2
        ) >= 55:

            return True

    return False


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    source1,
    candidate
):

    name1 = source1["business_name"]
    name2 = candidate["business_name"]

    address1 = source1["business_address"]
    address2 = candidate["business_address"]

    name_tokens1 = source1["_name_tokens"]
    name_tokens2 = candidate["_name_tokens"]

    address_tokens1 = source1["_address_tokens"]
    address_tokens2 = candidate["_address_tokens"]

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    if name1 and name2:

        name_ratio = (
            fuzz.ratio(
                name1,
                name2
            ) / 100
        )

        name_token_similarity = (
            fuzz.token_set_ratio(
                name1,
                name2
            ) / 100
        )

        name_token_sort_similarity = (
            fuzz.token_sort_ratio(
                name1,
                name2
            ) / 100
        )

        name_length_ratio = (
            min(
                len(name1),
                len(name2)
            )
            /
            max(
                len(name1),
                len(name2)
            )
        )

    else:

        name_ratio = 0.0
        name_token_similarity = 0.0
        name_token_sort_similarity = 0.0
        name_length_ratio = 0.0

    name_jaccard = jaccard(
        name_tokens1,
        name_tokens2
    )

    common_name_tokens = len(
        name_tokens1
        &
        name_tokens2
    )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    if address1 and address2:

        address_ratio = (
            fuzz.ratio(
                address1,
                address2
            ) / 100
        )

        address_token_similarity = (
            fuzz.token_set_ratio(
                address1,
                address2
            ) / 100
        )

        address_token_sort_similarity = (
            fuzz.token_sort_ratio(
                address1,
                address2
            ) / 100
        )

        address_length_ratio = (
            min(
                len(address1),
                len(address2)
            )
            /
            max(
                len(address1),
                len(address2)
            )
        )

    else:

        address_ratio = 0.0
        address_token_similarity = 0.0
        address_token_sort_similarity = 0.0
        address_length_ratio = 0.0

    address_jaccard = jaccard(
        address_tokens1,
        address_tokens2
    )

    common_address_tokens = len(
        address_tokens1
        &
        address_tokens2
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    country_match = int(
        bool(source1["country"])
        and
        source1["country"]
        ==
        candidate["country"]
    )

    # --------------------------------------------------------
    # RETURN EXACT TRAINING FEATURES
    # --------------------------------------------------------

    return {

        "name_ratio":
            name_ratio,

        "name_token_similarity":
            name_token_similarity,

        "name_token_sort_similarity":
            name_token_sort_similarity,

        "name_jaccard":
            name_jaccard,

        "common_name_tokens":
            common_name_tokens,

        "name_length_ratio":
            name_length_ratio,

        "business_suffix_match":
            suffix_match(
                name1,
                name2
            ),

        "address_ratio":
            address_ratio,

        "address_token_similarity":
            address_token_similarity,

        "address_token_sort_similarity":
            address_token_sort_similarity,

        "address_jaccard":
            address_jaccard,

        "common_address_tokens":
            common_address_tokens,

        "address_length_ratio":
            address_length_ratio,

        "address_number_similarity":
            number_similarity(
                source1["_numbers"],
                candidate["_numbers"]
            ),

        "country_match":
            country_match,

        "name_missing":
            int(
                not name1
                or
                not name2
            ),

        "address_missing":
            int(
                not address1
                or
                not address2
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print()
    print("=" * 75)
    print("AMAZON ML CHALLENGE - OVERNIGHT MODEL SCORING")
    print("=" * 75)

    # --------------------------------------------------------
    # FILE CHECKS
    # --------------------------------------------------------

    required_files = [
        DB_PATH,
        CANDIDATE_FILE,
        MODEL_FILE,
    ]

    for file_path in required_files:

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"Missing required file: {file_path}"
            )

    print()
    print("Files verified.")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("Loading trained model...")

    saved = joblib.load(
        MODEL_FILE
    )

    model = saved["model"]

    feature_names = saved[
        "feature_names"
    ]

    threshold = saved[
        "threshold"
    ]

    print(
        f"Model threshold: {threshold}"
    )

    print(
        f"Feature count:    {len(feature_names)}"
    )

    # --------------------------------------------------------
    # OPEN SQLITE
    # --------------------------------------------------------

    print()
    print("Opening SQLite entity index...")

    conn = sqlite3.connect(
        DB_PATH
    )

    # --------------------------------------------------------
    # OUTPUT MATCH TABLE
    # --------------------------------------------------------

    conn.execute("""
        DROP TABLE IF EXISTS overnight_matches
    """)

    conn.execute("""
        CREATE TABLE overnight_matches (
            source1_entity_id TEXT,
            matched_entity_id TEXT,
            matched_source TEXT,
            score REAL
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_overnight_s1
        ON overnight_matches(source1_entity_id)
    """)

    conn.commit()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total_candidates = 0
    total_cheap_pass = 0
    total_model = 0
    total_matches = 0
    chunk_number = 0

    # --------------------------------------------------------
    # STREAM CANDIDATE FILE
    # --------------------------------------------------------

    print()
    print(
        "Starting candidate processing..."
    )

    print(
        f"Chunk size: {CANDIDATE_CHUNK:,}"
    )

    print()

    for chunk in pd.read_csv(
        CANDIDATE_FILE,
        sep="\t",
        dtype=str,
        chunksize=CANDIDATE_CHUNK,
        keep_default_na=False,
    ):

        chunk_number += 1

        # ----------------------------------------------------
        # GET REQUIRED IDs
        # ----------------------------------------------------

        s1_ids = set(
            chunk[
                "source1_entity_id"
            ]
        )

        s2_ids = set(
            chunk.loc[
                chunk[
                    "candidate_source"
                ] == "source2",
                "candidate_entity_id"
            ]
        )

        s3_ids = set(
            chunk.loc[
                chunk[
                    "candidate_source"
                ] == "source3",
                "candidate_entity_id"
            ]
        )

        # ----------------------------------------------------
        # FETCH ONLY THIS CHUNK'S RECORDS
        # ----------------------------------------------------

        s1_records = fetch_records(
            conn,
            s1_ids,
            "source1"
        )

        s2_records = fetch_records(
            conn,
            s2_ids,
            "source2"
        )

        s3_records = fetch_records(
            conn,
            s3_ids,
            "source3"
        )

        feature_rows = []
        metadata = []

        # ----------------------------------------------------
        # PROCESS CANDIDATES
        # ----------------------------------------------------

        for row in chunk.itertuples(
            index=False
        ):

            source1 = s1_records.get(
                row.source1_entity_id
            )

            if source1 is None:
                continue

            if row.candidate_source == "source2":

                candidate = s2_records.get(
                    row.candidate_entity_id
                )

            else:

                candidate = s3_records.get(
                    row.candidate_entity_id
                )

            if candidate is None:
                continue

            # ------------------------------------------------
            # CHEAP FILTER
            # ------------------------------------------------

            if not cheap_filter(
                source1,
                candidate
            ):
                continue

            total_cheap_pass += 1

            # ------------------------------------------------
            # FEATURES
            # ------------------------------------------------

            feature_rows.append(
                extract_features(
                    source1,
                    candidate
                )
            )

            metadata.append(
                (
                    row.source1_entity_id,
                    row.candidate_entity_id,
                    row.candidate_source
                )
            )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        if feature_rows:

            X = pd.DataFrame(
                feature_rows
            )

            # Guarantee exact feature order.
            X = X[
                feature_names
            ]

            probabilities = (
                model.predict_proba(X)
                [:, 1]
            )

            total_model += len(
                probabilities
            )

            match_rows = []

            for meta, probability in zip(
                metadata,
                probabilities
            ):

                probability = float(
                    probability
                )

                if probability >= threshold:

                    match_rows.append(
                        (
                            meta[0],
                            meta[1],
                            meta[2],
                            probability
                        )
                    )

            # ------------------------------------------------
            # STORE MATCHES
            # ------------------------------------------------

            if match_rows:

                conn.executemany(
                    """
                    INSERT INTO overnight_matches
                    VALUES (?, ?, ?, ?)
                    """,
                    match_rows
                )

                conn.commit()

                total_matches += len(
                    match_rows
                )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        total_candidates += len(
            chunk
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        print(
            f"Chunk {chunk_number:>4} | "
            f"Candidates {total_candidates:>12,} | "
            f"Passed {total_cheap_pass:>12,} | "
            f"Model {total_model:>12,} | "
            f"Matches {total_matches:>12,} | "
            f"Time {elapsed / 60:.1f} min"
        )

        # ----------------------------------------------------
        # FREE RAM
        # ----------------------------------------------------

        del chunk
        del s1_records
        del s2_records
        del s3_records
        del feature_rows
        del metadata

        gc.collect()

    # ========================================================
    # BUILD FINAL MATCHING RESULTS
    # ========================================================

    print()
    print("=" * 75)
    print("SCORING FINISHED")
    print("=" * 75)

    print()
    print("Building final matching_results file...")

    # --------------------------------------------------------
    # Load predicted matches
    # --------------------------------------------------------

    matches = {}

    cursor = conn.execute("""
        SELECT
            source1_entity_id,
            matched_entity_id
        FROM overnight_matches
        ORDER BY source1_entity_id
    """)

    for (
        source1_id,
        matched_id
    ) in cursor:

        if source1_id not in matches:

            matches[source1_id] = []

        matches[source1_id].append(
            matched_id
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Every Source1 entity must appear exactly once.
    # Read Source1 IDs directly from SQLite.
    # --------------------------------------------------------

    output_rows = []

    cursor = conn.execute("""
        SELECT entity_id
        FROM entities
        WHERE source = 'source1'
        ORDER BY entity_id
    """)

    for (
        source1_id,
    ) in cursor:

        ids = matches.get(
            source1_id,
            []
        )

        # Remove duplicate IDs.
        ids = list(
            dict.fromkeys(ids)
        )

        output_rows.append(
            {
                "source1_entity_id":
                    source1_id,

                "matched_entity_ids":
                    ",".join(ids)
            }
        )

    # --------------------------------------------------------
    # WRITE OUTPUT
    # --------------------------------------------------------

    result = pd.DataFrame(
        output_rows
    )

    result.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # BASIC CHECKS
    # --------------------------------------------------------

    empty_matches = int(
        (
            result[
                "matched_entity_ids"
            ] == ""
        ).sum()
    )

    duplicate_source1 = int(
        result[
            "source1_entity_id"
        ].duplicated().sum()
    )

    elapsed = (
        time.time()
        -
        start_time
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("OVERNIGHT SCORING COMPLETE")
    print("=" * 75)

    print(
        f"Source1 rows:       {len(result):,}"
    )

    print(
        f"Candidates:         {total_candidates:,}"
    )

    print(
        f"Cheap-filter pass:  {total_cheap_pass:,}"
    )

    print(
        f"Model evaluated:    {total_model:,}"
    )

    print(
        f"Predicted matches:  {total_matches:,}"
    )

    print(
        f"Empty matches:      {empty_matches:,}"
    )

    print(
        f"Duplicate S1 rows:  {duplicate_source1:,}"
    )

    print(
        f"Output file:        {OUTPUT_FILE}"
    )

    print(
        f"Total time:         {elapsed / 60:.1f} minutes"
    )

    print("=" * 75)

    conn.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()