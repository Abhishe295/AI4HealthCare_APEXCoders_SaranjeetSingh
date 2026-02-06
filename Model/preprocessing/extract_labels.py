import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_DATA = os.path.join(BASE_DIR, "data", "original")
CURATED_DATA = os.path.join(BASE_DIR, "data", "curated")

MRI_FOLDER = os.path.join(ORIGINAL_DATA, "MRI")
METADATA_CSV = os.path.join(ORIGINAL_DATA, "MRI_metadata.csv")
OUTPUT_LABELS = os.path.join(CURATED_DATA, "labels.csv")

# Create curated folder if not exists
os.makedirs(CURATED_DATA, exist_ok=True)

def main():
    print("Reading metadata CSV...")
    df = pd.read_csv(METADATA_CSV)

    # Basic filtering as per problem statement
    df = df[
        (df["Modality"] == "MRI") &
        (df["Visit"] == "bl") &
        (df["Description"].str.contains("MPRAGE", case=False, na=False))
    ]

    # Keep only required columns
    df = df[["Subject", "Group"]]

    # Remove duplicate subjects (keep first baseline scan)
    df = df.drop_duplicates(subset="Subject")

    records = []
    skipped = 0

    for _, row in df.iterrows():
        subject_id = row["Subject"]
        label = row["Group"]

        subject_path = os.path.join(MRI_FOLDER, subject_id)

        # Ensure MRI folder exists
        if not os.path.isdir(subject_path):
            skipped += 1
            continue

        # Keep only valid labels
        if label not in ["CN", "MCI", "AD"]:
            skipped += 1
            continue

        records.append({
            "subject_id": subject_id,
            "label": label
        })

    labels_df = pd.DataFrame(records)
    labels_df.to_csv(OUTPUT_LABELS, index=False)

    # Reporting
    print("Labels extraction complete.")
    print("Total subjects:", len(labels_df))
    if "label" in labels_df.columns:
        print(labels_df["label"].value_counts())
    else:
        print("No labels extracted. Check filtering conditions.")

        print("Skipped subjects:", skipped)
        print(f"Saved to: {OUTPUT_LABELS}")

if __name__ == "__main__":
    main()
