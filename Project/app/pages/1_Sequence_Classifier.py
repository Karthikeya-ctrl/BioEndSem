import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Add the project root to path so we can import our extraction logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.feature_extraction import get_embeddings_batched

# 1. Page Config
st.set_page_config(page_title="Sequence Classifier", page_icon="🧬")

# 2. Load Real Models and Artifacts
MODEL_DIR = "models/trained_classifiers/"


@st.cache_resource
def load_artifacts():
    """Loads the label encoder, scaler, PCA, and the best performing model."""
    # Loading your winning XGBoost model
    model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    return model, encoder, scaler, pca


try:
    classifier, label_encoder, scaler, pca = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Models not found. Error: {e}")
    st.stop()

st.title("🧬 Nucleotide Sequence Classifier")
st.markdown("Enter a raw nucleotide sequence to extract DNA-BERT features and predict the neurodegenerative disease.")

# 3. Input Area
sequence_input = st.text_area("Enter Nucleotide Sequence", height=200, placeholder="ATGC...")

if st.button("Predict Disease", type="primary"):
    if len(sequence_input.strip()) < 20:
        st.error("Please enter a valid sequence.")
    else:
        with st.spinner("Processing through DNA-BERT and XGBoost (This takes a few seconds)..."):

            # --- THE FIX: Clean the Excel formatting ---
            clean_sequence = sequence_input.upper().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t",
                                                                                                                 "").strip()

            try:
                # A. Extract Embeddings (Notice the brackets [ ] because it expects a list)
                raw_emb = get_embeddings_batched([clean_sequence])

                # B. Wrap in Pandas DataFrame so the Scaler knows the column names
                feature_columns = [f"Dim_{i + 1}" for i in range(768)]
                emb_df = pd.DataFrame(raw_emb, columns=feature_columns)

                # C. Apply Scaling and PCA
                scaled_emb = scaler.transform(emb_df)
                pca_emb = pca.transform(scaled_emb)

                # D. Predict
                prediction_numeric = classifier.predict(pca_emb)[0]
                confidence = float(max(classifier.predict_proba(pca_emb)[0]) * 100)

                # E. Convert number back to disease name
                disease_name = label_encoder.inverse_transform([prediction_numeric])[0]

                # 4. Display Result
                st.success("✅ Analysis Complete!")
                col1, col2 = st.columns(2)
                col1.metric("Predicted Condition", disease_name)
                col2.metric("Confidence Score", f"{confidence:.2f}%")

                st.progress(confidence / 100)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")