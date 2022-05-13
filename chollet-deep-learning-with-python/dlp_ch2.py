# %% loading the MNIST data from Keras
import tensorflow as tf
from tensorflow import keras
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# %% defining the model architecture
from keras import layers
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# %% compiling the model
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],    
)

# %% preparing the image data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# %% fitting the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)