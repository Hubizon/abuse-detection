import os
import zipfile
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split
import dotenv

# Configuration

dotenv.load_dotenv()
import kaggle

RAW_DATA_DIR = Path('../data/raw')
PROCESSED_DATA_DIR = Path('../data/processed')
HOLDOUT_DIR = PROCESSED_DATA_DIR / 'holdout'

DATASET_NAME = 'jigsaw-unintended-bias-in-toxicity-classification'
FILE_NAME = 'train.csv'
COLUMNS_TO_KEEP = ['id', 'comment_text', 'created_date', 'target',
                   'severe_toxicity', 'obscene', 'identity_attack',
                   'insult', 'threat', 'sexual_explicit']

HOLDOUT_YEAR = 2017
HOLDOUT_MONTH_START = 6
HOLDOUT_MONTH_END = 10

STRATIFY_THRESHOLD = 0.4


# Data download

def kaggle_download_file(
        file_name: Union[str, Path],
        path: Path,
        dataset_name: str
):
    zip_file_name = f"{file_name}.zip"
    zip_file_path = path / zip_file_name
    extracted_file_path = path / file_name

    # Check if the file we want to download already exists
    if extracted_file_path.exists():
        print(f"{file_name} already exists.")
        return

    # For some reason, Kaggle downloads .zip files as .csv in this case
    kaggle.api.competition_download_file(
        dataset_name,
        file_name=file_name,
        path=path
    )

    # Rename the downloaded file and unzip it
    downloaded_path = path / file_name
    downloaded_path.rename(zip_file_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(path)

    os.remove(zip_file_path)
    print(f"Downloaded and extracted {file_name}.")


kaggle_download_file(FILE_NAME, path=RAW_DATA_DIR, dataset_name=DATASET_NAME)

# Load & Clean

raw_df = pd.read_csv(RAW_DATA_DIR / FILE_NAME)
cleaned_df = raw_df[COLUMNS_TO_KEEP].copy()
cleaned_df = cleaned_df.dropna()
cleaned_df['created_date'] = pd.to_datetime(cleaned_df['created_date'], errors='coerce').dt.tz_localize(None)
print('Data cleaned.')

# Train/val/test split

cutoff_date = pd.Timestamp(f'{HOLDOUT_YEAR}-{HOLDOUT_MONTH_START:02d}-01')
train_val_test_df = cleaned_df[cleaned_df['created_date'] < cutoff_date].copy()

# binary label for stratification
train_val_test_df['label'] = (train_val_test_df['target'] >= STRATIFY_THRESHOLD).astype(int)

train_set, val_test_set = train_test_split(
    train_val_test_df,
    test_size=0.2,
    stratify=train_val_test_df['label'],
    random_state=42
)

val_set, test_set = train_test_split(
    val_test_set,
    test_size=0.5,
    stratify=val_test_set['label'],
    random_state=42
)

for df in [train_set, val_set, test_set]:
    df.drop(columns='label', inplace=True)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
train_set.to_csv(PROCESSED_DATA_DIR / 'train.csv', index=False)
val_set.to_csv(PROCESSED_DATA_DIR / 'val.csv', index=False)
test_set.to_csv(PROCESSED_DATA_DIR / 'test.csv', index=False)
print('Split into train/val/test.')

# Holdout split (June-October 2017)

HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

for month in range(HOLDOUT_MONTH_START, HOLDOUT_MONTH_END + 1):
    mask = (cleaned_df['created_date'].dt.year == HOLDOUT_YEAR) & (cleaned_df['created_date'].dt.month == month)
    cleaned_df[mask].to_csv(HOLDOUT_DIR / f'{HOLDOUT_YEAR}_{month:02}.csv', index=False)

print('Data pipeline completed.')
