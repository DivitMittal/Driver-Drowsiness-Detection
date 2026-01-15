from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from ..config import TrainingConfig


def setup_training(model, checkpoint_path):
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    opt = Adam(learning_rate=TrainingConfig.LEARNING_RATE)

    checkpoint = ModelCheckpoint(
        checkpoint_path,
        save_weights_only=True,
        save_freq='epoch',
        period=TrainingConfig.CHECKPOINT_FREQ,
    )

    return binary_cross_loss, opt, checkpoint


def train_step(batch, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        yhat = model(X, training=True)
        loss = loss_fn(y, yhat)

    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    return loss


def train(model: Model, train_data, epochs: int, checkpoint_path: str):
    loss_fn, optimizer, checkpoint = setup_training(model, checkpoint_path)

    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}/{epochs}')
        epoch_loss = []

        for batch in train_data:
            loss = train_step(batch, model, loss_fn, optimizer)
            epoch_loss.append(loss)

        print(f'Loss: {tf.reduce_mean(epoch_loss):.4f}')

        if epoch % TrainingConfig.CHECKPOINT_FREQ == 0:
            checkpoint.save(file_prefix=TrainingConfig.CHECKPOINT_PREFIX)

    return model
