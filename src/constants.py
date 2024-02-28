import os
from enum import Enum

SEED = 42

# Data paths

DATA_FOLDER = "data/"

TRAIN_FOLDER = os.path.join(DATA_FOLDER, "trainset")
TRAIN_CSV = os.path.join(TRAIN_FOLDER, "trainset_true.csv")

TEST_FOLDER = os.path.join(DATA_FOLDER, "testset")
TEST_CSV = os.path.join(TEST_FOLDER, "testset_data.csv")

# Data constants

REFERENCE_DATE = "01/1/2020"
AGE_MEAN = 68.31626375749025
AGE_STD = 17.696206723535013
LYMPH_COUNT_MEAN = 2.4630287728533116
LYMPH_COUNT_STD = 1.1075497343746028

IMAGE_MEAN = [0.8188, 0.6997, 0.7045]
IMAGE_STD = [0.1896, 0.2114, 0.0913]


class DatasetType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
