import os
import pandas as pd
from feature_extraction import get_embeddings_batched

RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"

DISEASE_FILES = {
    "alzheimer_1000_sequences.xlsx": "Alzheimer's",
    "dementia_1000_sequences.xlsx": "Dementia",
    "huntington_1000_sequences.xlsx": "Huntington's",
    "parkinson_1000_sequences.xlsx": "Parkinson's",
    "sclerosis_1000_sequences.xlsx": "Multiple Sclerosis"
}


def process_datasets(sequence_column="Sequence"):
    all_features = []
    all_labels = []

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    for filename, label in DISEASE_FILES.items():
        filepath = os.path.join(RAW_DATA_DIR, filename)
        if not os.path.exists(filepath):
            continue

        print(f"\nProcessing {filename} ({label})...")
        df = pd.read_excel(filepath)

        sequences = df[sequence_column].dropna().astype(str).tolist()

        # We pass the entire list of 1000 sequences to the batched function
        print(f"⚡ Extracting embeddings in fast-batches for {label}...")
        embeddings = get_embeddings_batched(sequences, batch_size=32)

        all_features.extend(embeddings)
        all_labels.extend([label] * len(embeddings))

    print("\nCombining data into final dataset...")
    feature_columns = [f"Dim_{i + 1}" for i in range(len(all_features[0]))]
    final_df = pd.DataFrame(all_features, columns=feature_columns)
    final_df["Target_Disease"] = all_labels

    # SAVING WITH A NEW NAME
    output_path = os.path.join(PROCESSED_DATA_DIR, "mean_pooled_embeddings.csv")
    final_df.to_csv(output_path, index=False)
    print(f"✅ Success! Processed data saved to: {output_path}")


if __name__ == "__main__":
    process_datasets()