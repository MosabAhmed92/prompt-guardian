import joblib
import pandas as pd
from sklearn.metrics import recall_score

model = joblib.load(filename="src/model/selected_classifier.joblib")


def test_model_quality():
    golden_df = pd.read_csv("test/golden_dataset.csv")
    true_labels = golden_df["label"]
    pred_proba = model.predict_proba(golden_df["text"])[:, 1]
    pred_labels = (pred_proba >= 0.35).astype(int)
    recall = recall_score(true_labels, pred_labels, pos_label=1)
    assert recall >= 0.80, f"Malicious recal {recall} below threshold 0.80"
