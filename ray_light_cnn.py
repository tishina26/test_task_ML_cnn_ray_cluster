# https://docs.ray.io/en/latest/tune/examples/tune_mnist_keras.html
# https://docs.ray.io/en/latest/tune/index.html

import ray
from ray import air, tune
from ray.tune.integration.keras import TuneReportCallback

# importing the required packages
import tensorflow as tf
# импорт слоев
from tensorflow.keras import layers
# импорт модели
from tensorflow.keras.models import Sequential
# Импортируем набор данных MNIST
from tensorflow.keras.datasets import mnist

from pathlib import Path

def train_mnist(config):
    num_classes = 10

    # loading the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshaping the training and testing data
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    # normalizing the values of pixels of images
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0


    if (Path("./model_cnn.h5").exists()):
        model = tf.keras.models.load_model('./model_cnn.h5')
    else:
        model = Sequential([
            layers.Conv2D(config["neiron_0"], (3, 3), activation=config["activation_0"], input_shape=(x_train[0].shape)),
            layers.MaxPool2D((2, 2)),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(config["neiron_1"], activation=config["activation_1"]),
            layers.Dense(10, activation='softmax')
        ])

    model.compile(optimizer=config["optimizer"], loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint = tf.keras.callbacks.ModelCheckpoint("./model_cnn.h5", save_best_only=True)
    model.fit(x_train, y_train, batch_size=128, epochs=config["epochs"], verbose=1, validation_split=0.1,
              callbacks=[checkpoint, TuneReportCallback({"mean_accuracy": "accuracy"})], )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')


if __name__ == "__main__":
    # 2. Define a search space.
    param_space = {
        "optimizer": tune.choice(["sgd", "adam"]),
        "epochs": tune.randint(5, 10),
        # "epochs": tune.randint(1, 2),
        "neiron_0": tune.randint(16, 64),
        "neiron_1": tune.randint(128, 256),
        "activation_0": tune.choice(["sigmoid", "relu"]),
        "activation_1": tune.choice(["sigmoid", "relu"]),
    }

    tuner = tune.Tuner(
        train_mnist,
        tune_config=tune.TuneConfig(
            num_samples=10,
            metric="mean_accuracy",
            mode="max",
        ),
        param_space=param_space,
    )


    # 3. Start a Tune run and print the best result.
    results = tuner.fit()
    print("Best hyperparametres are:")
    print(results.get_best_result().config)
