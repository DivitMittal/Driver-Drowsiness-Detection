from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class ModelArchitecture:
    IMAGE_SIZE = 105
    CONV1_FILTERS = 64
    CONV2_FILTERS = 128
    CONV3_FILTERS = 128
    CONV4_FILTERS = 256
    KERNEL_SIZE = 10
    DENSE_UNITS = 1
    FINAL_ACTIVATION = 'sigmoid'
    EMBEDDING_OUTPUT = 4096


class DataConfig:
    EYE_POSITIVE_DIR = BASE_DIR / 'dataset' / 'eyes' / 'positive'
    EYE_NEGATIVE_DIR = BASE_DIR / 'dataset' / 'eyes' / 'negative'
    EYE_ANCHOR_DIR = BASE_DIR / 'dataset' / 'eyes' / 'anchor'
    YAWN_POSITIVE_DIR = BASE_DIR / 'dataset' / 'yawn' / 'positive'
    YAWN_NEGATIVE_DIR = BASE_DIR / 'dataset' / 'yawn' / 'negative'
    YAWN_ANCHOR_DIR = BASE_DIR / 'dataset' / 'yawn' / 'anchor'
    TRAIN_SPLIT = 0.7
    TEST_SPLIT = 0.3
    EYE_DATASET_SIZE = 300
    YAWN_DATASET_SIZE = 300


class TrainingConfig:
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    LOSS = 'binary_crossentropy'
    OPTIMIZER = 'adam'
    CHECKPOINT_PREFIX = './training_checkpoints/cp-{epoch:04d}.ckpt'
    CHECKPOINT_FREQ = 4
    METRICS = ['accuracy']


class HaarCascadeConfig:
    FACE_CASCADE_PATH = BASE_DIR / 'haarcascade' / 'haarcascade_frontalface_default.xml'
    EYE_CASCADE_PATH = BASE_DIR / 'haarcascade' / 'haarcascade_eye.xml'
    SCALE_FACTOR = 1.1
    MIN_NEIGHBORS = 5


class DetectionConfig:
    THRESHOLD = 0.435
    REGION_SIZE = 250
    YAWN_REGION_X = 220
    YAWN_REGION_Y = 120
    YAWN_REGION_W = 250
    YAWN_REGION_H = 250
