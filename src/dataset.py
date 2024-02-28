import datetime
import os

import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.constants import (
    AGE_MEAN,
    AGE_STD,
    IMAGE_MEAN,
    IMAGE_STD,
    LYMPH_COUNT_MEAN,
    LYMPH_COUNT_STD,
    REFERENCE_DATE,
    SEED,
    TEST_CSV,
    TEST_FOLDER,
    TRAIN_CSV,
    TRAIN_FOLDER,
    DatasetType,
)


def get_split(
    dataframe: pd.DataFrame, fold_id: int = 0, fold_numbers: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=fold_numbers, shuffle=True, random_state=SEED)
    for i, (train_index, val_index) in enumerate(
        skf.split(dataframe, dataframe["LABEL"])
    ):
        if i == fold_id:
            return dataframe.iloc[train_index], dataframe.iloc[val_index]

    raise ValueError(f"fold_id {fold_id} is out of range")


def process_dataframe(
    dataframe: pd.DataFrame, dataset_type: DatasetType
) -> pd.DataFrame:
    dataframe = dataframe.copy()
    age = dataframe["DOB"]
    age = age.apply(
        lambda x: (
            (
                datetime.datetime.strptime(REFERENCE_DATE, "%m/%d/%Y")
                - (
                    datetime.datetime.strptime(x, "%m/%d/%Y")
                    if "/" in x
                    else datetime.datetime.strptime(x, "%d-%m-%Y")
                )
            ).days
            / 365.25
        )
    )
    dataframe["AGE"] = (age - AGE_MEAN) / AGE_STD
    dataframe = dataframe.drop(columns=["DOB"])

    dataframe["GENDER"] = dataframe["GENDER"].apply(lambda x: 1 if x == "M" else 0)

    dataframe["LYMPH_COUNT"] = (
        np.log(dataframe["LYMPH_COUNT"]) - LYMPH_COUNT_MEAN
    ) / LYMPH_COUNT_STD

    if dataset_type == DatasetType.TRAIN or dataset_type == DatasetType.VAL:
        folder = TRAIN_FOLDER
    else:
        folder = TEST_FOLDER

    files = []
    for patient_id in dataframe.index:
        for file in os.listdir(os.path.join(folder, patient_id)):
            files.append(
                {
                    "patient_id": patient_id,
                    "path": os.path.join(folder, patient_id, file),
                }
            )

    files_df = pd.DataFrame(files)

    dataframe = dataframe.merge(files_df, right_on="patient_id", left_index=True)

    dataframe["patient_id"] = dataframe["patient_id"].apply(lambda x: int(x[1:]))

    return dataframe


def load_csv(
    fold_id: int = 0, fold_numbers: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_csv = pd.read_csv(TRAIN_CSV, index_col=0)

    train_df, val_df = get_split(train_csv, fold_id, fold_numbers)

    train_df = process_dataframe(train_df, DatasetType.TRAIN)
    val_df = process_dataframe(val_df, DatasetType.VAL)

    return train_df, val_df


class ImageWiseDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.values = dataframe.drop(columns=["path", "LABEL", "patient_id"]).values
        self.paths = dataframe["path"].values
        self.labels = dataframe["LABEL"].values
        self.patient_ids = dataframe["patient_id"].values

        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tabular_data = torch.tensor(self.values[idx]).float()
        image = self.image_transform(Image.open(self.paths[idx]).convert("RGB"))  # type: ignore

        return (
            image,
            tabular_data,
            torch.tensor(self.labels[idx]),
            torch.tensor(self.patient_ids[idx]),
        )


def get_train_val_loaders(batch_size: int, fold_id: int = 0, fold_numbers: int = 5):
    train_df, val_df = load_csv(fold_id, fold_numbers)

    train_dataset = ImageWiseDataset(train_df)
    val_dataset = ImageWiseDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_test_loader(batch_size: int):
    test_csv = pd.read_csv(TEST_CSV, index_col=0)
    test_df = process_dataframe(test_csv, DatasetType.TEST)

    test_dataset = ImageWiseDataset(test_df)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
