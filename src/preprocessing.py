import pandas as pd
import os

if __name__ == "__main__":
    input_path = "/opt/ml/processing/input/iris.csv"
    output_dir = "/opt/ml/processing/output"

    print("✅ Reading input file from:", input_path)
    df = pd.read_csv(input_path)

    print("✅ Splitting data into train/test...")
    train = df.sample(frac=0.8, random_state=42)
    test = df.drop(train.index)

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(f"{output_dir}/train.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)

    print("✅ Preprocessing completed. Files saved to:", output_dir)

