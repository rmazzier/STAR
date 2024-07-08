import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASET GENERATION PARAMETERS
DATAGEN_PARAMS = {"a": 0}

DATA_PATH = "../data/uniform_7subj/micro_doppler_stft/"

ACT_DICT = {
    "WALKING": 0,
    "RUNNING": 1,
    "SITTING": 2,
    "HANDS": 3,
}

MD_LEN = 200
STEP = 50

TEST_ACTIVITIES_LIST = ["xx"]

SUBJECTS = [1]
# PROB_REMOVE = 0.5

# TRAINING HYPERPARAMETERS

BATCH_SIZE = 32
# EPOCHS = 500
EPOCHS = 200
ADAM_LR = 0.001


# WANDB STUFF

CHECKPOINT_FREQUENCY = 2
MODEL_NAME = "XX"

# Setup dictionary with all hyperparameters
l = locals().copy()
CONFIG = {}

for key in l:
    if key[:2] != "__" and key != "torch":
        CONFIG[key] = l[key]
