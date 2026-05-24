import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

REPO_ID = os.getenv("HF_SPACE_REPO", "rapidflow/Wellness-Tourism-Prediction")
FOLDER_PATH = "tourism_project/deployment"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=REPO_ID, repo_type="space")
    print(f"Space '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    print(f"Space '{REPO_ID}' not found. Creating...")
    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Space '{REPO_ID}' created.")

api.upload_folder(
    folder_path=FOLDER_PATH,
    repo_id=REPO_ID,
    repo_type="space",
    path_in_repo="",
)
print(f"Deployed {FOLDER_PATH} to https://huggingface.co/spaces/{REPO_ID}")
