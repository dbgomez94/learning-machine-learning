# ------------- BINARY CLASSIFICATION -------------
# %% loading the IMDB dataset 
from re import X
from keras.backend import std
import tensorflow as tf
from tensorflow import keras
(train_data, train_labels), (test_data,test_labels) = keras.datasets.imdb.load_data(num_words=10000)

# %% understanding the data
print(f'There are {len(train_data)} reviews in train_data.')
print(f'The first review contains: {len(train_data[0])} words.')

# %% decoding reviews back to english
word_index = keras.datasets.imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
)
print('The first review translates to:')
print(decoded_review)

# %% encoding the integer sequences via multi-hot encoding
import numpy as np
def vectorized_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1
    return results

# vectorize train / test data
x_train = vectorized_sequences(train_data)
x_test = vectorized_sequences(test_data)

# vectorize train / test labels
y_train = np.asanyarray(train_labels).astype('float32')
y_test = np.asanyarray(test_labels).astype('float32')


# %% model definition
from tensorflow import keras
from keras import layers
model = keras.Sequential([
    layers.Dense(units=16, activation='relu'),
    layers.Dense(units=16, activation='relu'),
    layers.Dense(units=1, activation='sigmoid')
])

# %% compiling the model
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

# %% validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# %% training model
history = model.fit(
    x=partial_x_train,
    y=partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
)

# %% ploting the training and validation loss
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% plotting the training and validation accuracy
plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %% retraining a model with only 4 epochs
model = keras.Sequential([
    layers.Dense(units=16, activation='relu'),
    layers.Dense(units=16, activation='relu'),
    layers.Dense(units=1, activation='sigmoid'),
])
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(
    x =x_train,
    y=y_train,
    epochs=4,
    batch_size=512,    
)
results = model.evaluate(x_test, y_test)

# using model to predict on test data
model.predict(x_test)

# ------------- SINGLE-LABEL MULTICLASS CLASSIFICATION -------------
# %% loading the reuters dataset
from tensorflow import keras
(train_data, train_labels), (test_data, test_labels) = keras.datasets.reuters.load_data(num_words=10000)

# %% decoding newswires back to text
word_index = keras.datasets.reuters.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)
decoded_newswire = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
)

# %% encoding the input data
x_train = vectorized_sequences(train_data)
x_test = vectorized_sequences(test_data)

# %% encoding the labels
from tensorflow import keras
y_train = keras.utils.to_categorical(train_labels)
y_test = keras.utils.to_categorical(test_labels)

# %% model definition
import tensorflow as tf
from tensorflow import keras
from keras import layers
model = keras.Sequential([
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=46, activation='softmax'),
])

# %% model compilation
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# %% validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# %% training the model
history = model.fit(
    x=partial_x_train,
    y=partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
)
# %% plotting training and validation loss
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% plotting the training and validation accuracy
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# -------------- REGRESSION -----------------
# %% loading the housing data
from tensorflow import keras
(train_data, train_targets), (test_data, test_targets) = keras.datasets.boston_housing.load_data()

# %% feature-wise normalization
# for each feature, subtract the mean, divide by stdev
mean = train_data.mean(axis=0)
train_data -= mean
stdev = train_data.std(axis=0)
train_data /= stdev
test_data -= mean
test_data /= stdev

# %% model definition
from keras import layers
def build_model():
    model = keras.Sequential([
        layers.Dense(units=64, activation='relu'),
        layers.Dense(units=64, activation='relu'),
        layers.Dense(units=1), # no activation?
    ])
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=['mae'],
    )
    return model

# %% k-fold cross-validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_scores = []
all_mae_histories = []
for i in range(k):
    print(f'Processing fold # {i}')
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]]
    )
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]]
    )
    model = build_model()
    history = model.fit(
        x=partial_train_data,
        y=partial_train_targets,
        epochs=num_epochs,
        batch_size=16,
        verbose=0,
        validation_data=(val_data, val_targets),
    )
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)



# %% building the history of successive mean k-fold validation scores
avg_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)

]
# %% plotting the validation scores
import matplotlib.pyplot as plt
plt.plot(range(1, len(avg_mae_history) + 1), avg_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# %% excluding first ten data points
import matplotlib.pyplot as plt
truncated_mae_history = avg_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# %% training the final model
model = build_model()
model.fit(
    x=train_data,
    y=train_targets,
    epochs=130,
    batch_size=15,
    verbose=0,
)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

# %%
