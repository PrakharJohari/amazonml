import re
import unicodedata

import pandas as pd


REQUIRED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


def normalize_text(value):
    """
    General text normalization for business names and addresses.

    Steps:
    1. Handle missing values.
    2. Normalize Unicode.
    3. Convert text to lowercase.
    4. Replace '&' with 'and'.
    5. Remove punctuation.
    6. Collapse extra spaces.
    """
    if pd.isna(value):
        return ""

    text = str(value)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Case normalization
    text = text.casefold()

    # Treat & as the word "and"
    text = text.replace("&", " and ")

    # Replace punctuation/special characters with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Remove underscores
    text = text.replace("_", " ")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(value):
    """Normalize a business name."""
    return normalize_text(value)


def normalize_address(value):
    """Normalize a business address."""
    return normalize_text(value)


def normalize_country(value):
    """Normalize a country value without hard-coding specific countries."""
    return normalize_text(value)


def preprocess_dataframe(df):
    """
    Validate required columns and add normalized fields.

    Returns a copy of the input DataFrame.
    """
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    output = df.copy()

    output["name_norm"] = output["business_name"].map(normalize_name)
    output["address_norm"] = output["business_address"].map(normalize_address)
    output["country_norm"] = output["country"].map(normalize_country)

    return output