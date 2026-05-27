import streamlit as st

# 1. Configure the page settings (must be the first Streamlit command)
st.set_page_config(
    page_title="Genomic Disease Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Main Page Header
st.title("🧬 Neurodegenerative Disease Classifier")
st.subheader("Comparative Analysis of Nucleotide Sequences using BERT and ML")
st.markdown("---")

# 3. Project Overview Section
st.write("""
Welcome to the interactive portal for our genomic analysis project. 
This application is designed to analyze nucleotide sequences and predict associated 
neurodegenerative diseases by leveraging high-dimensional feature extraction and machine learning.
""")

# 4. Two-column layout for Methodology and Tech Stack
col1, col2 = st.columns(2)

with col1:
    st.header("🔬 Methodology")
    st.markdown("""
    * **Data Acquisition:** Sourcing nucleotide sequences for Alzheimer's, Parkinson's, Huntington's, ALS, and MS.
    * **Feature Extraction:** Using **DNA-BERT** to convert raw sequences into high-dimensional embeddings.
    * **Machine Learning:** Training robust classifiers (Random Forest, XGBoost) on the extracted genetic motifs.
    * **Comparative Analysis:** Visualizing disease-discriminative patterns using t-SNE/UMAP dimensionality reduction.
    """)

with col2:
    st.header("💻 Technical Stack")
    st.markdown("""
    * **Frontend:** Streamlit
    * **Data Processing:** Pandas, NumPy, BioPython
    * **Deep Learning / NLP:** Hugging Face Transformers, PyTorch
    * **Machine Learning:** Scikit-Learn, XGBoost
    * **Visualizations:** Plotly, Seaborn, Matplotlib
    """)

# 5. Sidebar Navigation Instructions
st.sidebar.success("👈 Select a page from the sidebar to begin.")
st.sidebar.info(
    "**Tip:** Go to the 'Sequence Classifier' page to test a custom nucleotide sequence against our trained models."
)

st.markdown("---")
st.caption("Developed for bioinformatics and genomic sequence research.")