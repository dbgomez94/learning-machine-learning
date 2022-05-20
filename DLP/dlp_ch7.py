# %% the Sequential class
from tensorflow import keras
from keras import layers
model = keras.Sequential([
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=10, activation='softmax'),
])

# %% incrementally building a Sequential model
model = keras.Sequential()
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))

# %% calling a model for the first time to build it
model.build(input_shape=(None, 3))
model.weights

# %% the summary() method
model.summary()

# %% naming models and layers with the name argument
model = keras.Sequential(name='my_model')
model.add(layers.Dense(units=64, activation='relu', name='my_first_layer'))
model.add(layers.Dense(units=10, activation='softmax', name='my_second_layer'))
model.build(input_shape=(None, 3))
model.summary()

# %% specifying the input shape of your model in advance
model = keras.Sequential()
model.add(keras.Input(shape=(3,)))
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))
model.summary()


# %% a simple functional model with two Dense layers
# input object holds info about shape and dtype of the data the model will process
inputs = keras.Input(shape=(3,), name='my_input')
# create a new layer and *call* it on the input
features = layers.Dense(units=64, activation='relu')(inputs)
# add the output layer and call it on the features
outputs = layers.Dense(units=10, activation='softmax')(features)
model = keras.Model(inputs=inputs, outputs=outputs)  # model constuctor
model.summary()

# %% a multi-input, multi-output Function model
vocab_size = 10000
num_tags = 100
num_departments = 4

# define the model inputs
title = keras.Input(shape=(vocab_size,), name='title')
text_body = keras.Input(shape=(vocab_size,), name='text_body')
tags = keras.Input(shape=(num_tags,), name='tags')
inputs = [title, text_body, tags]

# combine input features into a single tensor, features, by concatenating them
features = layers.Concatenate()(inputs)

# apply an intermediate layer to recombine input features into richer representations
features = layers.Dense(units=64, activation='relu')(features)

# define model ouputs
priority = layers.Dense(units=1, activation='sigmoid', name='priority')(features)
department = layers.Dense(units=num_departments, activation='softmax', name='department')(features)
outputs = [priority, department]
# create the model by specifying its inputs and outputs
model = keras.Model(inputs=inputs, outputs=outputs)

# %% training a model by providing lists of input and target arrays
import numpy as np
num_samples = 1280

# dummy input data
title_data = np.random.randint(0, 2, size=(num_samples, vocab_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocab_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# dummy target data
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

model.compile(
    optimizer='rmsprop',
    loss=['mean_squared_error', 'categorical_crossentropy'],
    metrics=[['mean_absolute_error'], ['accuracy']]
)
model.fit(
    x=[title_data, text_body_data, tags_data],
    y=[priority_data, department_data],
    epochs=1,
)
model.evaluate(
    x=[title_data, text_body_data, tags_data],
    y=[priority_data, department_data],
)
priority_preds, deparment_preds = model.predict([title_data, text_body_data, tags_data])

# %% training a model by providing dicts of input and target arrays
model.compile(
    optimizer='rmsprop',
    loss={
        'priority': 'mean_squared_error',
        'department': 'categorical_crossentropy',
    },
    metrics={
        'priority': ['mean_absolute_error'],
        'department': ['accuracy'],
    }
)
model.fit(
    x={
        'title': title_data,
        'text_body': text_body_data,
        'tags': tags_data,
    },
    y={
        'priority': priority_data,
        'department': department_data,
    }
)
priority_preds, deparment_preds = model.predict({
    'title': title_data,
    'text_body': text_body_data,
    'tags': tags_data,
})

# %% plotting Functional model with plot_model()
keras.utils.plot_model(model, 'ticket_classifier.png', show_shapes=True)

# %% retrieving the inputs or outputs of a layer in a Functional model
model.layers
model.layers[3].input
model.layers[3].output

# %% creating a new model by reusing intermediate layer outputs
features = model.layers[4].output
difficulty = layers.Dense(units=3, activation='softmax', name='difficulty')(features)

new_model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department, difficulty],
)
keras.utils.plot_model(new_model, 'updated_ticket_classifier.png', show_shapes=True)

# %% a simple model subclass
class CustomerTicketModel(keras.Model):
 
    # define sublayers in constructor
    def __init__(self, num_departments):
        super().__init__()  # constructor
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(units=64, activation='relu')
        self.priority_scorer = layers.Dense(units=1, activation='sigmoid')
        self.department_classifier = layers.Dense(units=num_departments, activation='softmax')

    # define the forward pass in the call() method
    def call(self, inputs):
        title = inputs['title']
        text_body = inputs['text_body']
        tags = inputs['tags']
        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department

model = CustomerTicketModel(num_departments=4)
priority, department = model(
    {'title': title_data, 'text_body': text_body_data, 'tags': tags_data}
)

model.compile(
    optimizer='rmsprop',
    loss=['mean_squared_error', 'categorical_crossentropy'],
    metrics=[['mean_absolute_error'], ['accuracy']],
)
model.fit(
    x={'title': title_data, 'text_body': text_body_data, 'tags': tags_data},
    y=[priority_data, department_data],
    epochs=1
)
model.evaluate(
    x={'title': title_data, 'text_body': text_body_data, 'tags': tags_data},
    y=[priority_data, department_data],
)
priority_preds, deparment_preds = model.predict(
    x={'title': title_data, 'text_body': text_body_data, 'tags': tags_data},
)

# %% creating a functional model that includes a subclassed model (compare to listing 7.8 on p.177)
class CLassifier(keras.Model):

    def __init__(self, num_classes=2):
        super().__init__()
        if num_classes == 2:
            num_units = 1
            activation = 'sigmoid'
        else:
            num_units = num_classes
            activation = 'softmax'
        self.dense = layers.Dense(units=num_units, activation=activation)

    def call(self, inputs):
        return self.dense(inputs)

inputs = keras.Input(shape=(3,))
features = layers.Dense(units=64, activation='relu')
outputs = CLassifier(num_classes=10)(features)
model = keras.Model(inputs=inputs, outputs=outputs)

# %% creating a subclassed model that includes a Functional model
inputs = keras.Input(shape=(3,))
outputs = layers.Dense(units=1, activation='sigmoid')(inputs)
binary_classifier = keras.Model(inputs=inputs, outputs=outputs)

class MyModel(keras.Model):

    def __init__(self, num_classes=2):
        super().__init__()
        self.dense = layers.Dense(units=64, activation='relu')
        self.classifier = binary_classifier

    def call(self, inputs):
        features = self.dense(inputs)
        return self.classifier(features)

model = MyModel()

# %% the standard workflow: compile(), fit(), evaluate(), predict()
def get_mnist_model():
    inputs = keras.Input(shape=(28*28,))
    features = layers.Dense(units=512, activation='relu')(inputs)
    features = layers.Dropout(rate=0.5)(features)
    outputs = layers.Dense(units=10, activation='softmax')(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

(images, labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
images = images.reshape((60000, 28*28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28*28)).astype('float32') / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

model = get_mnist_model()
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(
    x=train_images,
    y=train_labels,
    epochs=3,
    validation_data=(val_images, val_labels),
)
test_metrics = model.evaluate(
    x=test_images,
    y=test_labels,
)
predictions = model.predict(
    x=test_images,
)

# %% implementing a custom metric by subclassing the Metric class
import tensorflow as tf
class RootMeanSquareError(keras.metrics.Metric):

    # define the sate variables in the constructor
    # like for layers, you access to the add_weight() method
    def __init__(self, name='rmse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name='mse_sum', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros', dtype='int32')

    # implement the state update logic in update_state()
    # the y_true argument is the targets (or labels) for one batch,
    # while y_pred represents the corresponding predictions from the model
    def update_state(self, y_true, y_pred, sample_weight=None):
        # to match our mnsit model, we expect categorical preidictions and integer labels
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = tf.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)

    # use the result() methods to return the current value of the metric
    def result(self):
        return tf.sqrt(self.mse_sum / tf.cast(self.total_samples, tf.float32))


    # use reset_state() method to reinstantiate the metric
    def reset_state(self):
        self.mse_sum.assign(0.0)
        self.total_samples.assign(0)


model = get_mnist_model()
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', RootMeanSquareError()],
)
model.fit(
    x=train_images,
    y=train_labels,
    epochs=3,
    validation_data=(val_images, val_labels),
)
test_metrics = model.evaluate(test_images, test_labels)

# %% the callbacks argument in the fit() method

# callbacks are passed to the model via the callbacks argument in fit(), 
# which takes a list of callbacks
callbacks_list = [
    # interupts training when improvement stops
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # monitors the model's validation accuracy
        patience=2,  # interrupts training when accuracy has stopped improving for 2 epochs
    ),
    # saves the current weights after every epoch
    keras.callbacks.ModelCheckpoint(
        filepath='checkpoint_path.keras',  # path to destination model file
        monitor='val_loss',
        # previous two args mean won't overwrite model unless val_loss has improved,
        # which allows you to keep the best model seen during training
        save_best_only=True,
    )
]

model = get_mnist_model()
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(
    x=train_images,
    y=train_labels,
    epochs=10,
    validation_data=(val_images, val_labels),
    callbacks=callbacks_list,
)
# %% creating a custom callback by subclassing the Callback class
import matplotlib.pyplot as plt

class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs):
        self.per_batch_losses = []
    
    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get('loss'))
    
    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses, label='Training loss for each batch')
        plt.xlabel(f'Batch (epoch {epoch})')
        plt.ylabel('Loss')
        plt.legend()
        self.per_batch_losses = []


model = get_mnist_model()
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(
    x=train_images,
    y=train_labels,
    epochs=10,
    callbacks=[LossHistory()],
    validation_data=(val_images, val_labels),
)

# %% um, is TensorBoard the greatest thing ever?
model = get_mnist_model()
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
tensorboard = keras.callbacks.TensorBoard(
    log_dir='/Users/gomez/learning-deep-learning/DLP'
)
model.fit(
    x=train_images,
    y=train_labels,
    epochs=10,
    callbacks=[tensorboard],
    validation_data=(val_images, val_labels),
)

# %% writing a step-by-step training loop: the training step function
model = get_mnist_model()

loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.RMSprop()
metrics = [keras.metrics.SparseCategoricalAccuracy()]
loss_tracking_metric = keras.metrics.Mean()

def train_step(inputs, targets):
    
    # run the forward pass with training=True inside a GradientTape
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    # run the backward pass, note the use of only the trainable weights
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    
    # keep track of metrics
    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs[metric.name] = metric.result()

    # keep train of the loss average
    loss_tracking_metric.update_state(loss)
    logs['loss'] = loss_tracking_metric.result()

    # return the current values of the metrics and the loss
    return logs

# %% writing a step-by-step training loop: resetting the metrics
def reset_metrics():
    for metric in metrics:
        metric.reset_state()
    loss_tracking_metric.reset_state()

# %% writing a step-by-step training loop: the loop itself
training_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
training_dataset = training_dataset.batch(32)
epochs = 3
for epoch in range(epochs):
    reset_metrics()
    for inputs_batch, targets_batch in training_dataset:
        logs = train_step(inputs_batch, targets_batch)
        print(f'Results at the end of epoch {epoch}')
        for key, value in logs.items():
            print(f'...{key}: {value:.4f}')

# %% writing a step-by-step evaluation loop
@tf.function
def test_step(inputs, targets):
    predictions = model(inputs, training=True)
    loss = loss_fn(targets, predictions)

    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs['val_' + metric.name] = metric.result()
    loss_tracking_metric.update_state(loss)
    logs['val_loss'] = loss_tracking_metric.result()
    return logs

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.batch(32)
reset_metrics()
for inputs_batch, targets_batch in val_dataset:
    logs = test_step(inputs_batch, targets_batch)
print('Evaluation results:')
for key, value in logs.items():
    print(f'...{key}: {value:.4f}')


# %% implementing a custom training step to use with fit()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
loss_tracker = keras.metrics.Mean(name='loss')  # metric object to track avg per-batch losses

class CustomModel(keras.Model):

    # override the train_step method
    # @tf.function - don't need to add this, framework does it for you
    def train_step(self, data):
        inputs, targets = data
        
        # forward pass
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)  # use self (not model) since out model is the class itself
            loss = loss_fn(targets, predictions)
        # backward pass
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # update loss tracker that tracks avg per-batch loss
        loss_tracker.update_state(loss)

        # return average loss
        return {'loss': loss_tracker.result()}

    # metrics to resuts across epochs
    @property
    def metrics(self):
        return {loss_tracker}

# instantiate our custom model
inputs = keras.Input(shape=(28*28,))
features = layers.Dense(units=512, activation='relu')(inputs)
features = layers.Dropout(rate=0.5)(features)
outputs = layers.Dense(units=10, activation='softmax')(features)
model = CustomModel(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(),
    metrics=['accuracy']
)
model.fit(
    x=train_images, 
    y=train_labels, 
    epochs=3,
)


# %%
