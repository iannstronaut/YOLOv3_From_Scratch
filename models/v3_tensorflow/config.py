import tensorflow as tf
import os
import numpy as np

# Dataset path
DATASET = "/content/pascal/PASCAL_VOC"
IMG_DIR = os.path.join(DATASET, "images/")
LABEL_DIR = os.path.join(DATASET, "labels/")

# Device configuration
DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"

# Hyperparameters
NUM_WORKERS = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.4
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_FILE = "checkpoint.ckpt"

# Anchors (rescaled to [0, 1])
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]


def train_transform(image, bbox):
    image = tf.image.resize_with_pad(
        image, int(IMAGE_SIZE * 1.2), int(IMAGE_SIZE * 1.2)
    )
    image = tf.image.random_crop(image, size=[IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.1)
    image = tf.image.random_flip_left_right(image)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, bbox


def test_transform(image, bbox):
    image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, bbox


# Class labels
PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

COCO_LABELS = [  # shortened for readability
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    # ... complete the list as needed
]
