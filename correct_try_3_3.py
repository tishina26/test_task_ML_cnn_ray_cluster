import tensorflow as tf
# импорт слоев
from tensorflow.keras import layers
# импорт модели
from tensorflow.keras.models import Sequential
# Импортируем набор данных MNIST
from tensorflow.keras.datasets import mnist

# на сколько категорий классифицируем картинки
num_classes = 10

# загружаем тренировочные и тестовые данные
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert to one-hot vector
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# 0.206  0.941 if epochs = 10
# 0.265  0.926 if epochs = 5
model = Sequential([
    layers.Dense(28*28, activation='sigmoid', input_shape=(x_train[0].shape)),
    layers.Dense(128, activation='sigmoid'),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')

