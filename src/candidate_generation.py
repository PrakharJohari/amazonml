from collections import Counter, defaultdict

import pandas as pd

from preprocessing import preprocess_dataframe


# Blocking thresholds
# These are based on validation experiments.
STRONG_TOKEN_MAX_FREQUENCY = 25
COMBINED_TOKEN_MAX_FREQUENCY = 500


def _tokenize(text):
    """Return unique non-empty tokens from normalized text."""
    if not text:
        return set()

    return {
        token
        for token in str(text).split()
        if token
    }


def _build_token_frequency(values):
    """
    Count how many records contain each token.

    A token is counted once per record rather than once
    per occurrence.
    """
    counter = Counter()

    for value in values:
        counter.update(_tokenize(value))

    return counter


def _build_indexes(df):
    """
    Build token and exact-value indexes for a source.

    Returns:
        name_frequency
        address_frequency
        name_index
        address_index
        exact_name_index
        exact_address_index
    """

    name_frequency = _build_token_frequency(
        df["name_norm"]
    )

    address_frequency = _build_token_frequency(
        df["address_norm"]
    )

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    exact_name_index = defaultdict(list)
    exact_address_index = defaultdict(list)

    for row_idx, row in enumerate(
        df.itertuples(index=False)
    ):
        name_tokens = _tokenize(row.name_norm)
        address_tokens = _tokenize(row.address_norm)

        for token in name_tokens:
            name_index[token].append(row_idx)

        for token in address_tokens:
            address_index[token].append(row_idx)

        if row.name_norm:
            exact_name_index[
                row.name_norm
            ].append(row_idx)

        if row.address_norm:
            exact_address_index[
                row.address_norm
            ].append(row_idx)

    return (
        name_frequency,
        address_frequency,
        name_index,
        address_index,
        exact_name_index,
        exact_address_index,
    )


def _get_candidates_for_row(
    row,
    name_frequency,
    address_frequency,
    name_index,
    address_index,
    exact_name_index,
    exact_address_index,
):
    """
    Generate candidate row indices using multiple
    blocking rules.

    Blocking rules:

    1. Exact normalized business name
    2. Exact normalized address
    3. Rare name token
    4. Rare address token
    5. Two moderately frequent name tokens
    6. Two moderately frequent address tokens
    7. One name token + one address token
    """

    candidates = set()

    name_tokens = _tokenize(row.name_norm)
    address_tokens = _tokenize(row.address_norm)

    # --------------------------------------------------
    # Rule 1: Exact normalized name
    # --------------------------------------------------

    if row.name_norm:
        candidates.update(
            exact_name_index.get(
                row.name_norm,
                []
            )
        )

    # --------------------------------------------------
    # Rule 2: Exact normalized address
    # --------------------------------------------------

    if row.address_norm:
        candidates.update(
            exact_address_index.get(
                row.address_norm,
                []
            )
        )

    # --------------------------------------------------
    # Rule 3: Rare / strong name tokens
    # --------------------------------------------------

    strong_name_tokens = [
        token
        for token in name_tokens
        if name_frequency.get(token, 0)
        <= STRONG_TOKEN_MAX_FREQUENCY
    ]

    for token in strong_name_tokens:
        candidates.update(
            name_index.get(token, [])
        )

    # --------------------------------------------------
    # Rule 4: Rare / strong address tokens
    # --------------------------------------------------

    strong_address_tokens = [
        token
        for token in address_tokens
        if address_frequency.get(token, 0)
        <= STRONG_TOKEN_MAX_FREQUENCY
    ]

    for token in strong_address_tokens:
        candidates.update(
            address_index.get(token, [])
        )

    # --------------------------------------------------
    # Moderately frequent tokens
    # --------------------------------------------------

    name_common_tokens = [
        token
        for token in name_tokens
        if name_frequency.get(token, 0)
        <= COMBINED_TOKEN_MAX_FREQUENCY
    ]

    address_common_tokens = [
        token
        for token in address_tokens
        if address_frequency.get(token, 0)
        <= COMBINED_TOKEN_MAX_FREQUENCY
    ]

    # --------------------------------------------------
    # Rule 5: Two shared name tokens
    # --------------------------------------------------

    for i in range(
        len(name_common_tokens)
    ):
        token1 = name_common_tokens[i]

        rows1 = set(
            name_index.get(token1, [])
        )

        for j in range(
            i + 1,
            len(name_common_tokens)
        ):
            token2 = name_common_tokens[j]

            rows2 = set(
                name_index.get(token2, [])
            )

            candidates.update(
                rows1.intersection(rows2)
            )

    # --------------------------------------------------
    # Rule 6: Two shared address tokens
    # --------------------------------------------------

    for i in range(
        len(address_common_tokens)
    ):
        token1 = address_common_tokens[i]

        rows1 = set(
            address_index.get(token1, [])
        )

        for j in range(
            i + 1,
            len(address_common_tokens)
        ):
            token2 = address_common_tokens[j]

            rows2 = set(
                address_index.get(token2, [])
            )

            candidates.update(
                rows1.intersection(rows2)
            )

    # --------------------------------------------------
    # Rule 7: Shared name + address evidence
    # --------------------------------------------------

    for name_token in name_common_tokens:

        name_rows = set(
            name_index.get(
                name_token,
                []
            )
        )

        for address_token in address_common_tokens:

            address_rows = set(
                address_index.get(
                    address_token,
                    []
                )
            )

            candidates.update(
                name_rows.intersection(
                    address_rows
                )
            )

    return candidates


def generate_candidates(
    source1_df,
    source2_df,
    source_name="source2",
):
    """
    Generate candidate pairs between Source 1
    and one candidate source (Source 2 or Source 3).

    Country is used as a blocking restriction.

    The function returns:

        source1_entity_id
        candidate_entity_id
        candidate_source
    """

    s1 = preprocess_dataframe(
        source1_df
    )

    s2 = preprocess_dataframe(
        source2_df
    )

    (
        name_frequency,
        address_frequency,
        name_index,
        address_index,
        exact_name_index,
        exact_address_index,
    ) = _build_indexes(s2)

    # Country -> row indices
    country_index = defaultdict(set)

    for row_idx, country in enumerate(
        s2["country_norm"]
    ):
        if country:
            country_index[country].add(
                row_idx
            )

    entity_ids = s2[
        "entity_id"
    ].tolist()

    candidate_pairs = []

    for row in s1.itertuples(
        index=False
    ):

        candidates = _get_candidates_for_row(
            row=row,
            name_frequency=name_frequency,
            address_frequency=address_frequency,
            name_index=name_index,
            address_index=address_index,
            exact_name_index=exact_name_index,
            exact_address_index=exact_address_index,
        )

        # Apply country restriction
        country_rows = country_index.get(
            row.country_norm,
            set()
        )

        candidates.intersection_update(
            country_rows
        )

        for row_idx in candidates:

            candidate_pairs.append(
                {
                    "source1_entity_id":
                        row.entity_id,

                    "candidate_entity_id":
                        entity_ids[row_idx],

                    "candidate_source":
                        source_name,
                }
            )

    return pd.DataFrame(
        candidate_pairs,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
        ],
    )


def generate_all_candidates(
    source1_df,
    source2_df,
    source3_df,
):
    """
    Generate candidates against both Source 2
    and Source 3.

    Final candidate set is the union of both.
    """

    candidates_s2 = generate_candidates(
        source1_df=source1_df,
        source2_df=source2_df,
        source_name="source2",
    )

    candidates_s3 = generate_candidates(
        source1_df=source1_df,
        source2_df=source3_df,
        source_name="source3",
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

    return candidates