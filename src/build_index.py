import os
import sqlite3
import pandas as pd


DATASET = (
    r"C:\Users\Acer\Downloads"
    r"\6ab10eb3b23ba_student_resource"
    r"\student_resource"
    r"\\dataset"
)

DB_PATH = os.path.join(
    "output",
    "entity_index.db"
)

os.makedirs("output", exist_ok=True)


def main():

    # Do not overwrite an existing index
    if os.path.exists(DB_PATH):
        print("=" * 60)
        print("Index already exists:")
        print(DB_PATH)
        print("=" * 60)
        return

    print("=" * 60)
    print("BUILDING DISK-BACKED ENTITY INDEX")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY,
            business_name TEXT,
            business_address TEXT,
            country TEXT,
            source TEXT
        )
    """)

    conn.commit()

    files = [
        ("test_source1.tsv", "source1"),
        ("test_source2.tsv", "source2"),
        ("test_source3.tsv", "source3"),
    ]

    for filename, source_name in files:

        path = os.path.join(
            DATASET,
            "test",
            filename
        )

        print("\n" + "=" * 60)
        print(f"Processing {source_name}")
        print(f"File: {filename}")
        print("=" * 60)

        total = 0

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            chunksize=100_000,
            keep_default_na=False,
        ):

            rows = [
                (
                    row.entity_id,
                    row.business_name,
                    row.business_address,
                    row.country,
                    source_name,
                )
                for row in chunk.itertuples(index=False)
            ]

            conn.executemany(
                """
                INSERT OR REPLACE INTO entities
                (
                    entity_id,
                    business_name,
                    business_address,
                    country,
                    source
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

            conn.commit()

            total += len(rows)

            print(
                f"{source_name}: "
                f"{total:,} rows indexed"
            )

        print(
            f"Finished {source_name}: "
            f"{total:,} rows"
        )

    print("\nCreating lookup index...")

    conn.execute("""
        CREATE INDEX idx_entity_source
        ON entities(source, entity_id)
    """)

    conn.commit()
    conn.close()

    print("\n" + "=" * 60)
    print("INDEX COMPLETE")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()