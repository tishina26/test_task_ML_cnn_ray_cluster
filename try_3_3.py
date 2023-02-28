# original code
# https://webtort.ru/%D1%80%D0%B5%D1%88%D0%B0%D0%B5%D0%BC-%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D1%83-mnist-%D0%B2-keras-%D0%B8%D0%BB%D0%B8-%D1%83%D1%87%D0%B8%D0%BC-%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D1%81%D0%B5%D1%82%D1%8C-%D1%80/

import tensorflow as tf
import tensorflow.keras
import numpy as np
# -- Импорт для построения модели: --
# импорт слоев
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
# импорт модели
from tensorflow.keras.models import Sequential
# импорт оптимайзера
from tensorflow.keras.optimizers import Adam
# Импортируем набор данных MNIST
from tensorflow.keras.datasets import mnist


# загружаем тренировочные и тестовые данные
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Узнаем длины полученных массивов
# print(len(X_train), len(y_train), len(X_test), len(y_train))
# Проверка типа и размера данных
# print(X_train[0].shape,X_train[0].dtype)
# Выведем первый элемент массива на экран
# print(X_train[0])
# print(y_train[0])


# Преобразование данных в матрицах изображений
# X_train.max() возвращает значение 255
X_train = X_train/255
X_test = X_test/255


# Преобразуем целевые значения методом «one-hot encoding»
y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
y_test = tensorflow.keras.utils.to_categorical(y_test, 10)


# Создаем модель
# Точность 99
model = Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train[0].shape)),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Flatten(),
    layers.Dense(10, activation='sigmoid')
])

# Выведем полученную модель на экран
# model.summary()

#Компиляция модели
model.compile(loss='binary_crossentropy',
            optimizer = Adam(learning_rate=0.00024),
             metrics = ['binary_accuracy'])

# Функция ранней остановки
stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=6)
# Функция постепенного сохранения лучшего результата
checkpoint = tf.keras.callbacks.ModelCheckpoint("mnist_3_1.h5", save_best_only=True)

# Запускаем обучение модели
# validation_split = 0.2 - используем 2% данных для ускоренного обучения
history = model.fit(X_train, y_train, batch_size=500, verbose=1,
                    epochs= 50, validation_split = 0.2, callbacks=[checkpoint])
print("trained model")

# Сохранение обученной модели для дальнеййшего использования
model.save("model.h5")
print("saved model")

# Загрузить готовую модель
# model = tf.keras.models.load_model('/home/uliana/PycharmProjects/pythonProject2/model.h5')

print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(X_test[:3])
print("predictions shape:", predictions.shape)

