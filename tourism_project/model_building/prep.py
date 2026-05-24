# for data manipulation
import pandas as pd
# for train-test split
from sklearn.model_selection import train_test_split
# for Hugging Face Hub authentication and file upload
from huggingface_hub import HfApi
import os

# Hugging Face dataset repository
DATASET_REPO = "rapidflow/wellness-tourism-prediction"
DATASET_PATH = f"hf://datasets/{DATASET_REPO}/tourism.csv"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset directly from Hugging Face
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully from Hugging Face.")
print(f"Original shape: {df.shape}")

# Drop unique identifier (not useful for modeling)
df.drop(columns=["CustomerID"], inplace=True)

# Fix data quality issue in Gender column
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

target_col = "ProdTaken"

# Split into features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split (80/20) with stratification for imbalanced target
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Train and test sets saved locally.")

# Upload processed datasets back to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face.")
