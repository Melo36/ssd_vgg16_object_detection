import torch

BATCH_SIZE = 16 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 5 # Number of epochs to train for.
NUM_WORKERS = 4 # Number of parallel workers for data loading.
DEVICE = torch.device('cpu')
# Training images and XML files directory.
TRAIN_DIR = 'data/train'
# Validation images and XML files directory.
VALID_DIR = 'data/valid'
# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'ferrero_klassik', 'leibniz_butter', 'leibniz_kakao', 'lindt_lindor', 'milka_vollmilch',
]
NUM_CLASSES = len(CLASSES)
# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False
# Location to save model and plots.
OUT_DIR = 'outputs'