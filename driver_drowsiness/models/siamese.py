import keras
from keras import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)

from ..config import ModelArchitecture
from .layers import L1Dist


def make_conv_embedding():
    inputs = Input(shape=(ModelArchitecture.IMAGE_SIZE,
                          ModelArchitecture.IMAGE_SIZE, 3),
                   name='input_image')

    x = Conv2D(ModelArchitecture.CONV1_FILTERS,
               ModelArchitecture.KERNEL_SIZE,
               activation='relu')(inputs)
    x = MaxPooling2D(64)(x)

    x = Conv2D(ModelArchitecture.CONV2_FILTERS,
               ModelArchitecture.KERNEL_SIZE,
               activation='relu')(x)
    x = MaxPooling2D(64)(x)

    x = Conv2D(ModelArchitecture.CONV3_FILTERS,
               ModelArchitecture.KERNEL_SIZE,
               activation='relu')(x)
    x = MaxPooling2D(64)(x)

    x = Conv2D(ModelArchitecture.CONV4_FILTERS,
               ModelArchitecture.KERNEL_SIZE,
               activation='relu')(x)
    x = MaxPooling2D(64)(x)

    x = Flatten()(x)
    x = Dense(ModelArchitecture.EMBEDDING_OUTPUT, activation='relu')(x)

    return Model(inputs, x, name='embedding')


def make_siamese():
    embedding = make_conv_embedding()

    input_image = Input(name='input_img',
                        shape=(ModelArchitecture.IMAGE_SIZE,
                               ModelArchitecture.IMAGE_SIZE, 3))
    validation_image = Input(name='validation_img',
                             shape=(ModelArchitecture.IMAGE_SIZE,
                                    ModelArchitecture.IMAGE_SIZE, 3))

    input_embedding = embedding(input_image)
    validation_embedding = embedding(validation_image)

    l1_dist = L1Dist()(input_embedding, validation_embedding)

    outputs = Dense(ModelArchitecture.DENSE_UNITS,
                    activation=ModelArchitecture.FINAL_ACTIVATION)(l1_dist)

    return Model(inputs=[input_image, validation_image],
                 outputs=outputs,
                 name='siamese_network')
