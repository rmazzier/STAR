import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"

# ========== DATASET GENERATION PARAMETERS ==========
DATAGEN_PARAMS = {
    "N_PASSES": 1,
    "DIST_BOUNDS": (10, 120),
    "NWIN": 64,
    "TREP": 32,
    "Nd": 64,  # original 64
    "BP_SEL": 0,
}

N_KEPT_BINS = 10

RAW_DATA_PATH = "./data/raw_data"
DATA_PATH_V2 = "./data/sparse_dataset_V2"
RAW_REV1_DATA_PATH = "./data/newdata_rev1_raw"
REV1_DATA_PATH = "./data/newdata_rev1"
CNN_DATA_PATH = "./data/cnn_dataset"

ACTIVITIES = [
    "WALKING",
    "RUNNING",
    "SITTING",
    "HANDS",
]

SUBJECTS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
]

# ============ TRAINING HYPERPARAMETERS ============

# float in [0, 1], e.g. 0.5 = 50% of the dataset
DATASET_SUBSAMPLE_FACTOR = 1

# percentage of the dataset to use for training, validation and testing
SPLIT_PTGS = [0.8, 0.01, 0.19]

# random seed for dataset split
DATASET_SPLIT_SEED = 123

# default number of epochs to train
EPOCHS = 3

ADAM_LR = 0.0002

L1_LOSS = False

N_LIHT_ITERS = 1

# must sum up to one
# LOSS_WEIGHTS = [0.1, 0.1, 0.1, 0.2, 0.5]
# LOSS_WEIGHTS = [0.1, 0.2, 0.7]
# LOSS_WEIGHTS = [1.0]

LIHT_OMEGA = 10
LEARN_LIHT_S = True
INIT_W_D_AS_FOURIER = True
L_IHT_WEIGHT = 0.1
L_MD_WEIGHT = 0.9
W_D_REG_WEIGHT = 0.0

N_PAST_WINDOWS = 3
USE_ATTENTION = True
LEARN_ATTENTION = False
TEACHER_FORCING = False
LEARN_W_TRANSPOSED = False

ONLY_ADD = False
ONLY_MULT = False

LEARN_W = True
MODEL_TYPE = "LIHT"
# MODEL_TYPE = "DUST"

P_REMOVE_BOUNDS = [0.0, 0.9]
BURST_BOUNDS = [3, 12]
P_BURST = 0.0


NRANGE = 110
W = 64
L = 20

# =========== WANDB STUFF ===========
MODEL_NAME = "Model name here"

NOTES = """Final ablation runs"""


WANDB_TAG = ["AblationRuns"]

# WANDB_MODE = "online"
WANDB_MODE = "disabled"


l = locals().copy()
CONFIG = {}

for key in l:
    if key[:2] != "__" and key != "torch":
        CONFIG[key] = l[key]
