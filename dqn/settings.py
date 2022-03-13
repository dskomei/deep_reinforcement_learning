import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE_TRAIN = 128
BATCH_SIZE_VALID = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

EPOCHS = 1000
