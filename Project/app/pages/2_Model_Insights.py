import streamlit as st
import pandas as pd
import joblib
import os
import plotly.figure_factory as ff

st.set_page_config(page_title="Model Insights", page_icon="📊", layout="wide")

MODEL_DIR = "models/trained_classifiers/"

st.title("📊 Model Performance Insights")
st.markdown("Explore the detailed evaluation metrics and confusion matrices for all trained models.")

# 1. Load and display the overall leaderboard
summary_path = os.path.join(MODEL_DIR, "model_performance_summary.csv")
if not os.path.exists(summary_path):
    st.error("⚠️ No model data found. Please run the training script first.")
    st.stop()

leaderboard = pd.read_csv(summary_path)

st.subheader("🏆 Model Leaderboard")
st.dataframe(
    leaderboard.style.highlight_max(subset=['Accuracy'], color='lightgreen').format({"Accuracy": "{:.2f}%"}),
    use_container_width=True
)

st.divider()

st.subheader("🔍 Deep Dive: All Model Diagnostics")

# Load label encoder once
try:
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    classes = label_encoder.classes_
except Exception as e:
    st.error("Could not load label encoder. Please run the training script.")
    st.stop()

# 2. LOOP THROUGH ALL MODELS IN THE LEADERBOARD
for model_name in leaderboard['Model'].tolist():
    model_prefix = model_name.lower()

    try:
        # Load specific model artifacts
        cm = joblib.load(os.path.join(MODEL_DIR, f"{model_prefix}_cm.pkl"))
        report = joblib.load(os.path.join(MODEL_DIR, f"{model_prefix}_report.pkl"))

        st.markdown(f"## 🔹 {model_name}")

        col1, col2 = st.columns([1, 1.5])

        # --- COLUMN 1: Precision / Recall / F1 Dataframe ---
        with col1:
            st.markdown(f"**Classification Report**")
            report_df = pd.DataFrame(report).transpose().drop(columns=['support'])
            st.dataframe(
                report_df.style.background_gradient(cmap='Blues').format("{:.3f}"),
                use_container_width=True
            )

        # --- COLUMN 2: Confusion Matrix Heatmap ---
        with col2:
            st.markdown(f"**Confusion Matrix**")
            fig = ff.create_annotated_heatmap(
                z=cm[::-1],
                x=list(classes),
                y=list(classes)[::-1],
                colorscale='Viridis',
                showscale=True
            )
            fig.update_layout(
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()  # Adds a nice visual line before the next model

    except Exception as e:
        st.warning(f"⚠️ Could not load charts for {model_name}. Did the training script finish generating its files?")