import xgboost as xgb
import pandas as pd
import os

if __name__ == "__main__":
    print("✅ Loading training data...")
    train = pd.read_csv("/opt/ml/input/data/train/train.csv")

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    print("✅ Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective="multi:softmax", num_class=3, n_estimators=100
    )
    model.fit(X_train, y_train)

    os.makedirs("/opt/ml/model", exist_ok=True)
    model.save_model("/opt/ml/model/model.json")
    print("✅ Model saved successfully at /opt/ml/model/model.json")

