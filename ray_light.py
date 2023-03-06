# https://docs.ray.io/en/latest/tune/examples/tune_mnist_keras.html
# https://docs.ray.io/en/latest/tune/index.html

import ray
from ray import air, tune
from ray.tune.integration.keras import TuneReportCallback

import tensorflow as tf
# импорт слоев
from tensorflow.keras import layers
# импорт модели
from tensorflow.keras.models import Sequential
# Импортируем набор данных MNIST
from tensorflow.keras.datasets import mnist


def train_mnist(config):
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = Sequential([
        layers.Dense(config["neiron_0"], activation=config["activation_0"], input_shape=(x_train[0].shape)),
        layers.Dense(config["neiron_1"], activation=config["activation_1"], input_shape=(x_train[0].shape)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=config["optimizer"], loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=config["epochs"], verbose=1, validation_split=0.1,
              callbacks=[TuneReportCallback({"mean_accuracy": "accuracy"})], )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')


if __name__ == "__main__":
    # 2. Define a search space.
    param_space = {
        "optimizer": tune.choice(["sgd", "adam"]),
        "epochs": tune.randint(5, 10),
        # "epochs": tune.randint(1, 2),
        "neiron_0": tune.randint(32, 512),
        "neiron_1": tune.randint(32, 64),
        "activation_0": tune.choice(["sigmoid", "relu"]),
        "activation_1": tune.choice(["sigmoid", "relu"]),
    }

    tuner = tune.Tuner(
        tune.with_resources(train_mnist, resources={"cpu": 2, "gpu": 0}),
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
