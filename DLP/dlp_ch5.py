# %% rare features and spurious correlations
from cProfile import label
import numpy as np
from tensorflow import keras
from keras import layers

# add noisy channels and zeros channels to training images
(train_images, train_labels), _ = keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

train_images_with_noise_channels = np.concatenate(
    [train_images, np.random.random((len(train_images), 784))],
    axis=1
)
train_images_with_zero_channels = np.concatenate(
    [train_images, np.zeros((len(train_images), 784))],
    axis=1
)

# train same model on MNSIT daat with noise channels or all-zero channels
def get_model():
    model = keras.Sequential([
        layers.Dense(units=512, activation='relu'),
        layers.Dense(units=10, activation='softmax'),
    ])
    model.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

model = get_model()
history_noise = model.fit(
    x=train_images_with_noise_channels,
    y=train_labels,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
)
model = get_model()
history_zeros = model.fit(
    x=train_images_with_zero_channels,
    y=train_labels,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
)

# %% plotting a validation accuracy comparison
import matplotlib.pyplot as plt
val_acc_noise = history_noise.history['val_accuracy']
val_acc_zeros = history_zeros.history['val_accuracy']
epochs = range(1, 11)
plt.plot(epochs, val_acc_noise, 'b-', label='Validation Accuracy with Noise Channels')
plt.plot(epochs, val_acc_zeros, 'b--', label='Validation Accuracy with Zeros Channels')
plt.title('Effect of Noise Channels on Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %% Fitting an MNIST model with randomly shuffled labels
(train_images, train_labels), _ = keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

random_train_labels = train_labels[:]
np.random.shuffle(random_train_labels)

model = keras.Sequential([
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=10, activation='softmax'),
])
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.fit(
    x=train_images,
    y=train_labels,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
)

# %% training MNIST model with incorrectly high learning rate
(train_images, train_labels), _ = keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

model = keras.Sequential([
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=10, activation='softmax'),
])
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-2),  # learning_rate = 1.0 -> 23% accuracy
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(
    x=train_images,
    y=train_labels,
    epochs=10,
    batch_size=128,  # smaples in gradient calculations
    validation_split=0.2,
)


# %% a simple logistic regression on MNIST
model = keras.Sequential([
    layers.Dense(units=96, activation='relu'),  # added two layers to increase
    layers.Dense(units=96, activation='relu'),  # model complextity
    layers.Dense(units=10, activation='softmax'),
])
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
history_small_model = model.fit(
    x=train_images,
    y=train_labels,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
)
val_loss = history_small_model.history['val_loss']
epochs = range(1, 21)
plt.plot(epochs, val_loss, 'b--', label='Validation Loss')
plt.title('Effect of Insufficient Model Capacity on Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# %% original movie-review classification model
(train_data, train_labels), _ = keras.datasets.imdb.load_data(num_words=10000)
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.0
    return results
train_data = vectorize_sequences(train_data)
model = keras.Sequential([
    layers.Dense(
        units=16,  # adjust model capacity with units
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.002)  # L2 regularization
    ), 
    layers.Dense(
        units=16, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.002)
    ),
    layers.Dense(
        units=1, 
        activation='sigmoid'
    ),
])
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)
history_original = model.fit(
    x=train_data,
    y=train_labels,
    epochs=20,
    batch_size=512,
    validation_split=0.4,
)

# %%
