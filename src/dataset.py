import datetime
import os
import random

import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

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
    dataframe: pd.DataFrame, dataset_type: DatasetType, add_file_path: bool = True
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

    dataframe["GENDER"] = dataframe["GENDER"].apply(lambda x: 0.5 if x == "M" else -0.5)

    dataframe["LYMPH_COUNT"] = (
        np.log(dataframe["LYMPH_COUNT"]) - LYMPH_COUNT_MEAN
    ) / LYMPH_COUNT_STD

    if add_file_path:
        dataframe = merge_file_path(dataframe, dataset_type)
        dataframe["patient_id"] = dataframe["patient_id"].apply(lambda x: int(x[1:]))

    else:
        dataframe["patient_id"] = dataframe.index
        dataframe["patient_id"] = dataframe["patient_id"].apply(lambda x: int(x[1:]))
        dataframe = dataframe.reset_index(drop=True)

    return dataframe


def merge_file_path(dataframe: pd.DataFrame, dataset_type: DatasetType) -> pd.DataFrame:
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

    return dataframe


def load_train_csv(
    fold_id: int = 0, fold_numbers: int = 5, add_file_path: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_csv = pd.read_csv(TRAIN_CSV, index_col=0)

    train_df, val_df = get_split(train_csv, fold_id, fold_numbers)

    train_df = process_dataframe(
        train_df, DatasetType.TRAIN, add_file_path=add_file_path
    )
    val_df = process_dataframe(val_df, DatasetType.VAL, add_file_path=add_file_path)

    return train_df, val_df


def load_test_csv(add_file_path: bool = True) -> pd.DataFrame:
    test_csv = pd.read_csv(TEST_CSV, index_col=0)
    test_df = process_dataframe(test_csv, DatasetType.TEST, add_file_path)

    return test_df


def tabular_masking(tabular_data: np.ndarray) -> np.ndarray:
    tabular_data = tabular_data.copy()
    masking = [random.random() > 0.5 for _ in range(len(tabular_data))]
    tabular_data[masking] = 0

    return tabular_data


class ImageWiseDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, augmentations: bool = False):
        self.values = dataframe.drop(columns=["path", "LABEL", "patient_id"]).values
        self.paths = dataframe["path"].values
        self.labels = dataframe["LABEL"].values.astype(np.float32)

        unique_labels, count = np.unique(self.labels, return_counts=True)
        print("Unique labels:", unique_labels, "Counts:", count)

        self.patient_ids = dataframe["patient_id"].values

        self.image_transform = transforms.Compose(
            [
                transforms.ToImageTensor(),
                transforms.ToDtype(torch.float32),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ]
        )

        self.augmentations = augmentations
        self.augmentations_transform = transforms.Compose(
            [
                transforms.ToImageTensor(),
                transforms.ToDtype(torch.float32),
                transforms.RandomRotation((-180, 180), expand=True),
                transforms.RandomResizedCrop(
                    224, ratio=(0.8, 1.2), scale=(0.5, 1.0), antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tabular_data = self.values[idx]
        pil_image = Image.open(self.paths[idx]).convert("RGB")  # type: ignore

        if self.augmentations and random.random() > 0.5:
            image = self.augmentations_transform(pil_image)
        else:
            image = self.image_transform(pil_image)

        if self.augmentations and random.random() > 0.5:
            tabular_data = tabular_masking(tabular_data)

        return (
            image,
            torch.tensor(tabular_data).float(),
            torch.tensor(self.labels[idx]).unsqueeze(0),
            torch.tensor(self.patient_ids[idx]),
        )


def get_train_val_loaders(
    batch_size: int,
    fold_id: int = 0,
    fold_numbers: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_df, val_df = load_train_csv(fold_id, fold_numbers)

    print("Train dataset:")
    train_dataset = ImageWiseDataset(train_df, augmentations=True)
    print("Val dataset:")
    val_dataset = ImageWiseDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def get_test_loader(batch_size: int) -> DataLoader:
    test_csv = pd.read_csv(TEST_CSV, index_col=0)
    test_df = process_dataframe(test_csv, DatasetType.TEST)

    test_dataset = ImageWiseDataset(test_df)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
