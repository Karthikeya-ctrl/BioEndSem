# 🧬 Genomic Disease Classification: DNA-BERT & ML Pipeline

## 📖 Project Overview
This project is an end-to-end bioinformatics machine learning pipeline designed to classify raw nucleotide sequences into one of five neurodegenerative conditions: **Alzheimer's, Dementia, Huntington's, Parkinson's, and Multiple Sclerosis**. 

By treating DNA as a semantic language, the pipeline leverages a pre-trained **DNA-BERT** transformer model to extract high-dimensional mathematical embeddings from raw sequences. Through rigorous mathematical optimization—specifically **Batched Mean Pooling** and **Principal Component Analysis (PCA)**—the pipeline compresses these features and feeds them into an optimized **XGBoost** classifier, achieving an **82.14% macro-accuracy**. The entire backend is wrapped in an interactive, real-time **Streamlit** diagnostic dashboard.

---

## 🏗️ Core Architecture & Engineering Justifications

Throughout the development of this pipeline, several critical architectural decisions were made to optimize for biological reality, mathematical stability, and hardware constraints (specifically an NVIDIA RTX 4060 with 8GB VRAM).

### 1. Why DNABERT (v1) and 6-mers?
* **Biological Intuition:** DNABERT v1 forces overlapping **6-mer tokenization** (e.g., `ATGCGT`). Protein binding sites and regulatory motifs are typically 6-10 base pairs long, making 6-mers a highly biologically relevant representation.
* **Why not DNABERT-2?** Newer models use BPE (Byte Pair Encoding), which creates variable-length tokens (e.g., a 2-letter token next to an 8-letter token). Because our pipeline relies heavily on the mathematical average of the sequence, BPE would skew the weight of different motifs.
* **Hardware Limit:** DNABERT v1 (Base) has ~110M parameters, fitting perfectly into the 8GB VRAM of an RTX 4060 while leaving room for batch processing (Batch Size = 32).

### 2. The Context Window and Dimensionality (512 & 768)
* **512 Tokens:** The model reads sequences in a context window of 512 tokens. Due to the quadratic scaling of Multi-Head Self-Attention ($512 \times 512$ connections), this is the optimal limit for standard hardware memory limits.
* **768 Dimensions:** The mathematical meaning of each token is represented by a 768-dimensional vector. This specific number is chosen because it is perfectly divisible by the model's **12 Attention Heads** (64 dimensions per head), allowing the GPU to process parallel matrix multiplications at maximum hardware speed.

### 3. Batched Mean Pooling vs. The `[CLS]` Token
* **The `[CLS]` Trap:** Standard BERT models use the first token (`[CLS]`) as a summary of the sequence. For DNA, this artificially caps accuracy (~74%) because it dilutes or "blurs out" highly localized, distant disease-causing mutations.
* **The Solution:** We implemented **Mean Pooling**. By mathematically averaging the 768-dimensional hidden states of *every* active token in the 512-length sequence, every localized mutation gets a democratic vote in the final vector. An **Attention Mask** is applied via PyTorch tensor multiplication to completely zero out `[PAD]` token noise before the average is calculated.

### 4. Dimensionality Reduction (PCA to 100)
* **The Curse of Dimensionality:** Feeding 5,000 rows of 768-dimensional data into tree-based ML models causes overfitting, and caused distance-based models (SVM) to loop infinitely (Platt Scaling stall).
* **The Sweet Spot:** We passed the normalized embeddings through PCA, reducing 768 columns to exactly **100 Principal Components**. Based on the Scree Plot (Cumulative Explained Variance), the top 100 components successfully capture **>90% of the true biological variance** while mathematically deleting background noise.

### 5. The Classification Engine (XGBoost)
* A massive hyperparameter tournament was run across 8 models using `RandomizedSearchCV`. **XGBoost** emerged victorious, easily mapping the complex decision boundaries of the 100-dimensional PCA space. 
* During real-time inference, confidence scores are extracted using XGBoost's `predict_proba()` function, identifying the highest probability among the 5 target classes.

---

## 🛠️ Technology Stack
* **Language:** Python 3.11 *(Specifically downgraded from 3.12+ to ensure PyTorch CUDA binary compatibility with the RTX 4060 GPU)*.
* **Deep Learning:** PyTorch (CUDA), Hugging Face Transformers (`zhihan1996/DNA_bert_6`).
* **Machine Learning:** Scikit-Learn, XGBoost.
* **Frontend UI:** Streamlit, Plotly.

---

## 📂 Project Structure
```text
Project_Root/
├── app/
│   ├── Home.py                             # Streamlit entry point
│   └── pages/
│       ├── 1_Sequence_Classifier.py        # Live inference dashboard with \n scrubbing
│       └── 2_Model_Insights.py             # Performance metrics & Plotly heatmaps
├── data/
│   ├── raw/                                # Original sequence files (.xlsx, .csv)
│   └── processed/
│       └── mean_pooled_embeddings.csv      # Extracted 768-D features
├── models/
│   └── trained_classifiers/
│       ├── xgboost.pkl, randomforest.pkl...# Trained model artifacts
│       ├── pca.pkl, scaler.pkl             # Dimensionality reduction artifacts
│       └── label_encoder.pkl               # Target label mappings
├── src/
│   ├── data_processing.py                  # PyTorch tensor math & Mean Pooling logic
│   └── model_training.py                   # 8-model RandomizedSearchCV pipeline
├── README.md
└── requirements.txt
