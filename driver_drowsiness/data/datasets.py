import tensorflow as tf

from .preprocessing import preprocess_image_pair
from ..config import DataConfig, TrainingConfig


def create_siamese_dataset(anchor_dir, pos_dir, neg_dir, dataset_size):
    positive = tf.data.Dataset.list_files(f'{pos_dir}/*.jpg').take(dataset_size)
    negative = tf.data.Dataset.list_files(f'{neg_dir}/*.jpg').take(dataset_size)
    anchor = tf.data.Dataset.list_files(f'{anchor_dir}/*.jpg').take(dataset_size)

    data = tf.data.Dataset.zip((anchor, positive, negative))
    data = data.map(preprocess_image_pair)
    data = data.cache()
    data = data.shuffle(buffer_size=1024)

    train_count = int(dataset_size * TrainingConfig.TRAIN_SPLIT)
    train_data = data.take(train_count)
    train_data = train_data.batch(TrainingConfig.BATCH_SIZE).prefetch(8)

    test_data = data.skip(train_count)
    test_data = test_data.batch(TrainingConfig.BATCH_SIZE).prefetch(8)

    return train_data, test_data


def create_eye_dataset():
    return create_siamese_dataset(
        str(DataConfig.EYE_ANCHOR_DIR),
        str(DataConfig.EYE_POSITIVE_DIR),
        str(DataConfig.EYE_NEGATIVE_DIR),
        DataConfig.EYE_DATASET_SIZE,
    )


def create_yawn_dataset():
    return create_siamese_dataset(
        str(DataConfig.YAWN_ANCHOR_DIR),
        str(DataConfig.YAWN_POSITIVE_DIR),
        str(DataConfig.YAWN_NEGATIVE_DIR),
        DataConfig.YAWN_DATASET_SIZE,
    )
