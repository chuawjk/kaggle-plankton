import tensorflow as tf

from src.config import TRAIN_DIR, WEIGHTS_DIR
from src.config import (
    BATCH_SIZE,
    INPUT_SHAPE,
    LR,
    DROPOUT,
    DEFAULT_CLASSES
)

NUM_CLASSES = len(DEFAULT_CLASSES)


def initialise_model(num_classes=NUM_CLASSES):    
    """
    Specifies model architecture based on hyperparameters in config.py

    Args:
        num_classes (int): Number of classes to predict. Defaults to 121.

    Returns:
        tf.keras.Model object
    """
    pretrained_model = tf.keras.applications.MobileNetV2(INPUT_SHAPE, include_top=False)

    inputs = tf.keras.Input(shape=INPUT_SHAPE)

    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = pretrained_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=["categorical_accuracy", "AUC"],
    )
    return model
