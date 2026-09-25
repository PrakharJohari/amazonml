import re
import unicodedata

from rapidfuzz import fuzz


# ---------------------------------------
# 1. SAFE TEXT HANDLING
# ---------------------------------------

def safe_text(value):
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"nan", "none", "null", "n/a"}:
        return ""

    return text


# ---------------------------------------
# 2. TEXT NORMALIZATION
# ---------------------------------------

def normalize_text(text):
    text = safe_text(text).lower()

    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)

    # Remove combining marks
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------
# 3. TOKENIZATION
# ---------------------------------------

def get_tokens(text):
    normalized = normalize_text(text)

    if not normalized:
        return []

    return normalized.split()


# ---------------------------------------
# 4. SIMILARITY FEATURES
# ---------------------------------------

def calculate_similarity(text1, text2):

    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return 0.0, 0.0, 0.0, 0.0

    ratio = fuzz.ratio(text1, text2) / 100

    token_set_similarity = (
        fuzz.token_set_ratio(text1, text2) / 100
    )

    token_sort_similarity = (
        fuzz.token_sort_ratio(text1, text2) / 100
    )

    tokens1 = set(text1.split())
    tokens2 = set(text2.split())

    union = tokens1 | tokens2
    intersection = tokens1 & tokens2

    jaccard = (
        len(intersection) / len(union)
        if union else 0.0
    )

    return (
        ratio,
        token_set_similarity,
        token_sort_similarity,
        jaccard,
    )


# ---------------------------------------
# 5. NUMERIC FEATURE EXTRACTION
# ---------------------------------------

def extract_numbers(text):

    text = safe_text(text)

    return set(re.findall(r"\d+", text))


def calculate_number_similarity(text1, text2):

    numbers1 = extract_numbers(text1)
    numbers2 = extract_numbers(text2)

    if not numbers1 or not numbers2:
        return 0.0

    intersection = numbers1 & numbers2
    union = numbers1 | numbers2

    return len(intersection) / len(union)


# ---------------------------------------
# 6. BUSINESS SUFFIX FEATURES
# ---------------------------------------

BUSINESS_SUFFIXES = {
    "llc", "inc", "ltd", "limited",
    "corp", "corporation", "company",
    "co", "plc", "llp"
}


def get_business_suffixes(text):

    tokens = get_tokens(text)

    return {
        token for token in tokens
        if token in BUSINESS_SUFFIXES
    }


def suffix_match(text1, text2):

    suffixes1 = get_business_suffixes(text1)
    suffixes2 = get_business_suffixes(text2)

    if not suffixes1 and not suffixes2:
        return 1

    if suffixes1 == suffixes2:
        return 1

    return 0


# ---------------------------------------
# 7. MAIN FEATURE EXTRACTION
# ---------------------------------------

def extract_features(source1, candidate):

    name1 = safe_text(source1.get("business_name"))
    name2 = safe_text(candidate.get("business_name"))

    address1 = safe_text(source1.get("business_address"))
    address2 = safe_text(candidate.get("business_address"))

    country1 = normalize_text(source1.get("country"))
    country2 = normalize_text(candidate.get("country"))

    # Name similarity
    (
        name_ratio,
        name_token_similarity,
        name_token_sort_similarity,
        name_jaccard
    ) = calculate_similarity(name1, name2)

    # Address similarity
    (
        address_ratio,
        address_token_similarity,
        address_token_sort_similarity,
        address_jaccard
    ) = calculate_similarity(address1, address2)

    # Numeric similarity
    address_number_similarity = calculate_number_similarity(
        address1, address2
    )

    # Token counts
    name_tokens1 = set(get_tokens(name1))
    name_tokens2 = set(get_tokens(name2))

    address_tokens1 = set(get_tokens(address1))
    address_tokens2 = set(get_tokens(address2))

    # Shared token counts
    common_name_tokens = len(
        name_tokens1 & name_tokens2
    )

    common_address_tokens = len(
        address_tokens1 & address_tokens2
    )

    # Length ratios
    name_length_ratio = (
        min(len(name1), len(name2)) /
        max(len(name1), len(name2))
        if name1 and name2 else 0.0
    )

    address_length_ratio = (
        min(len(address1), len(address2)) /
        max(len(address1), len(address2))
        if address1 and address2 else 0.0
    )

    # Country comparison
    country_match = int(
        bool(country1) and country1 == country2
    )

    # Final feature dictionary
    features = {

        # Name features
        "name_ratio": name_ratio,
        "name_token_similarity": name_token_similarity,
        "name_token_sort_similarity": name_token_sort_similarity,
        "name_jaccard": name_jaccard,
        "common_name_tokens": common_name_tokens,
        "name_length_ratio": name_length_ratio,
        "business_suffix_match": suffix_match(name1, name2),

        # Address features
        "address_ratio": address_ratio,
        "address_token_similarity": address_token_similarity,
        "address_token_sort_similarity": address_token_sort_similarity,
        "address_jaccard": address_jaccard,
        "common_address_tokens": common_address_tokens,
        "address_length_ratio": address_length_ratio,
        "address_number_similarity": address_number_similarity,

        # Country feature
        "country_match": country_match,

        # Missing-value features
        "name_missing": int(not name1 or not name2),
        "address_missing": int(not address1 or not address2),
    }

    return features


# ---------------------------------------
# 8. TEST
# ---------------------------------------

if __name__ == "__main__":

    source1 = {
        "business_name": "Prime Money Services LLC",
        "business_address": "17560 Ellis Road, Tahlequah, OK 74464",
        "country": "United States",
    }

    candidate = {
        "business_name": "Services Prime Money",
        "business_address": "17560 Ellis Rd, Tahlequah, Oklahoma 74464",
        "country": "United States",
    }

    features = extract_features(source1, candidate)

    print("\n========== FEATURE ENGINEERING TEST ==========")

    for key, value in features.items():
        print(f"{key}: {value}")