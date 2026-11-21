from enum import Enum

class CVOptions(Enum):
    FOLDS = "folds"
    LOPO = "Leave-One-Patient-Out"

class InnerCVOptions(Enum):
    TRAIN_TEST_SPLIT = "TrainTestSplit"
    NESTEDCV = "NestedCV"
