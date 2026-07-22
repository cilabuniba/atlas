from .strenum import StrEnum


class ParameterKeys(StrEnum):
    GENERAL = "general"
    DATA = "data"
    OUT_DIR = "out_dir"
    DEVICE = "device"
    PBAR = "pbar"
    NUM_EPOCHS = "num_epochs"
    SEED = "seed"
    MODEL = "model"
    NAME = "name"
    CFG = "config"
    LOADER = "loader"
    OPTIMIZER = "optimizer"
    EARLY_STOP = "early_stop"
    SCHEDULER = "scheduler"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    TARGET = "target"
    CRITERION = "criterion"
    HF_COSINE_SCHEDULER = "HFCosineScheduler"
    METRICS = "metrics"
    SPLIT = "split"
    