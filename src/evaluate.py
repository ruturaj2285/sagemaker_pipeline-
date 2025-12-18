import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score
import json
import os

if __name__ == "__main__":
    print("✅ Starting evaluation1...")

    test = pd.read_csv("/opt/ml/input/data/test/test.csv")
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    model = xgb.XGBClassifier()
    model.load_model("/opt/ml/processing/model/model.json")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model accuracy: {acc:.4f}")

    os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)
    report_dict = {
        "classification_metrics": {
            "accuracy": {"value": acc, "standard_deviation": 0.0}
        }
    }

    with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
        json.dump(report_dict, f)

    print("✅ Evaluation report written to evaluation.json")

