import os
import pandas as pd
import joblib
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

PROCESSED_DATA_PATH = "data/processed/mean_pooled_embeddings.csv"
MODEL_SAVE_DIR = "models/trained_classifiers/"


def get_models_and_params():
    return {
        "RandomForest": (RandomForestClassifier(random_state=42, n_jobs=-1),
                         {'n_estimators': [100, 200], 'max_depth': [None, 20]}),
        "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1),
                    {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.2]}),
        "SVM": (SVC(probability=True, random_state=42, max_iter=2000), {'C': [0.1, 1], 'kernel': ['linear']}),
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1), {'C': [0.1, 1, 10]}),
        "KNN": (KNeighborsClassifier(n_jobs=-1), {'n_neighbors': [5, 7]}),
        "DecisionTree": (DecisionTreeClassifier(random_state=42), {'max_depth': [10, 20]}),
        "NeuralNetwork_MLP": (MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
                              {'hidden_layer_sizes': [(128, 64), (64, 32)]}),
        "AdaBoost": (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100]})
    }


def train_models():
    print("🚀 Starting High-Speed PCA Multi-Model Pipeline...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=["Target_Disease"])
    y = df["Target_Disease"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    joblib.dump(le, os.path.join(MODEL_SAVE_DIR, "label_encoder.pkl"))

    print("⚖️ Normalizing features and reducing dimensions (PCA)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=100, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler.pkl"))
    joblib.dump(pca, os.path.join(MODEL_SAVE_DIR, "pca.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, stratify=y_encoded,
                                                        random_state=42)

    models_dict = get_models_and_params()
    results = []

    print(f"\n⚙️ Tuning & Training {len(models_dict)} models...\n" + "=" * 50)

    for model_name, (model, param_grid) in models_dict.items():
        print(f"⏳ Training {model_name}...")
        start_time = time.time()

        search = RandomizedSearchCV(model, param_grid, n_iter=3, cv=3, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # --- NEW METRICS EXTRACTION ---
        # 1. Calculate Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        # 2. Calculate Detailed Report (Precision, Recall, F1) as a dictionary
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

        # --- SAVE ARTIFACTS ---
        joblib.dump(best_model, os.path.join(MODEL_SAVE_DIR, f"{model_name.lower()}.pkl"))
        joblib.dump(cm, os.path.join(MODEL_SAVE_DIR, f"{model_name.lower()}_cm.pkl"))
        joblib.dump(report, os.path.join(MODEL_SAVE_DIR, f"{model_name.lower()}_report.pkl"))

        elapsed_time = time.time() - start_time
        results.append({"Model": model_name, "Accuracy": acc * 100})
        print(f"✅ {model_name} Accuracy: {acc * 100:.2f}% (Took {elapsed_time:.1f}s)")

    print("\n" + "=" * 50 + "\n🏆 FINAL COMPARATIVE ANALYSIS REPORT\n" + "=" * 50)
    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    print(results_df)
    results_df.to_csv(os.path.join(MODEL_SAVE_DIR, "model_performance_summary.csv"), index=False)
    print("\n✨ All optimized models and metrics saved to models/trained_classifiers/")


if __name__ == "__main__":
    train_models()