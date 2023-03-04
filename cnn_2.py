# importing the required packages
# импорт слоев
from tensorflow.keras import layers
# импорт модели
from tensorflow.keras.models import Sequential
# Импортируем набор данных MNIST
from tensorflow.keras.datasets import mnist

# loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# reshaping the training and testing data
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

# normalizing the values of pixels of images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Test loss: 0.044
# Test accuracy: 0.986

model = Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(x_train[0].shape)),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(500, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# text Description of model
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=2, validation_split=0.1)

# evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')

# model.save("model_cnn.h5")