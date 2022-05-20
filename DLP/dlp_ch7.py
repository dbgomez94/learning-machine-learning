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
# %%
