from tensorflow import keras
from keras import layers

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
ouputs = [priority, department]
# create the model by specifying its inputs and outputs
model = keras.Model(inputs=inputs, outputs=outputs)
