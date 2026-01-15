import tensorflow as tf
from keras.config import image_data_format

from ..config import ModelArchitecture


def preprocess_image(path):
    byte_img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(byte_img)

    if image_data_format() == 'channels_first':
        img = tf.transpose(img, perm=[2, 0, 1])

    img = tf.image.resize(img, (ModelArchitecture.IMAGE_SIZE,
                                ModelArchitecture.IMAGE_SIZE))
    img = img / 255.0
    return img


def preprocess_image_pair(anchor, positive, negative):
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )
