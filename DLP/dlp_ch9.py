# %% preparing list of input fil paths and list of corresponding mask file paths
import os

input_dir = 'images/'
target_dir = 'annotations/trimaps/'

input_img_paths = sorted([
    os.path.join(input_dir, fname) 
    for fname in os.listdir(input_dir) 
    if fname.endswith('.jpg')
])

target_paths = sorted([
    os.path.join(target_dir, fname)
    for fname in os.listdir(target_dir)
    if fname.endswith('png') and not fname.startswith('.')
])

# %% plotting the imput image and target mask
import matplotlib.pyplot as plt
from tensorflow import keras

plt.axis('off')
plt.imshow(keras.utils.load_img(input_img_paths[9])
)

# for the target, orig labels are 1, 2, and 3 (this is why all the labels appear black). 
# substract 1 and multiply by 127 so labels become 0, 127, and 254

def display_target(target_array):
    normalized_array = (target_array.astype('uint8') - 1) * 127
    plt.axis('off')
    plt.imshow(normalized_array[:, :, 0])

img = keras.utils.img_to_array(keras.utils.load_img(target_paths[9], color_mode='grayscale'))
display_target(img)

# %% load images into numpy arrays and split data into training and validation sets
import numpy as np
import random

img_size = (200, 200)
num_imgs = len(input_img_paths)

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)

def path_to_input_image(path):
    return keras.utils.img_to_array(keras.utils.load_img(path, target_size=img_size))

def path_to_target(path):
    img = keras.utils.img_to_array(
        keras.utils.load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1
    return img

input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")
for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i])

num_val_samples = 1000
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]

# %% model definition
from tensorflow import keras
from keras import layers

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Rescaling(1./255)(inputs)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(filters=256, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(filters=256, kernel_size=3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(filters=128, kernel_size=3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=3, activation="relu", padding="same", strides=2)(x)

    outputs = layers.Conv2D(filters=num_classes, kernel_size=3, activation="softmax", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model

model = get_model(img_size=img_size, num_classes=3)
model.summary()

# %% model compilation
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
callbacks = keras.callbacks.ModelCheckpoint(
    filepath='oxford_segmentation.keras',
    save_best_only=True,
    monitor='val_loss',
)
history = model.fit(
    x=train_input_imgs,
    y=train_targets,
    epochs=50,
    callbacks=callbacks,
    batch_size=64,
    validation_data=(val_input_imgs, val_targets),
)

# %% plotting results
loss = history.history['loss']
epochs = range(1, len(loss) + 1)
val_loss = history.history['val_loss']
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.legend()

# %% reload best model and use it to predicted a image segmentation mask from val set
model = keras.models.load_model('oxford_segmentation.keras')

i = 4
test_image = val_input_imgs[i]
plt.axis('off')
plt.imshow(keras.utils.array_to_img(test_image))

mask = model.predict(np.expand_dims(test_image, 0))[0]

def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis('off')
    plt.imshow(mask)

display_mask(mask)


# %% convnet visualizations: visualizing intermediate activations
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers

model = keras.models.load_model('convnet_from_scratch_with_augmentation.keras')
model.summary()

# %% preprocessing single image
img_path = keras.utils.get_file(
    fname='cat.jpg',
    origin='https://img-datasets.s3.amazonaws.com/cat.jpg',
)

def get_img_array(img_path, target_size):
    img = keras.utils.load_img(img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

img_tensor = get_img_array(img_path, target_size=(180, 180))

# %% displaying the test picture
import matplotlib.pyplot as plt

plt.axis('off')
plt.imshow(img_tensor[0].astype('uint8'))
plt.show()

# %% instantiating a model that returns layer activations
from keras import layers

layer_outputs = []
layer_names = []
for layer in model.layers:
    if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

# %% using the model to compute layer activations
activations = activation_model.predict(img_tensor)

# %% visualizing the fifth channel
first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 5], cmap='viridis')
plt.axis('off')

# %% visualizaing every channel in every intermediate activation
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros(((size + 1) * n_cols - 1,
                             images_per_row * (size + 1) - 1))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            channel_image = layer_activation[0, :, :, channel_index].copy()
            if channel_image.sum() != 0:
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            display_grid[
                col * (size + 1): (col + 1) * size + col,
                row * (size + 1) : (row + 1) * size + row] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(display_grid, aspect="auto", cmap="viridis")


# %% convnet visualization: visualizing convnet filters
from tensorflow import keras
import numpy as np

model = keras.applications.xception.Xception(
    weights='imagenet',
    include_top=False,  # classification layers are irrelevant
)
for layer in model.layers:
    if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
        print(layer.name)
        
# %% creating a feature extractor model
layer_name = 'block3_sepconv1'
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

# %% using the feature extractor
import tensorflow as tf
activation = feature_extractor(keras.applications.xception.preprocess_input(img_tensor))

def compute_loss(image, filter_index):
    activation = feature_extractor(image)
    filter_activation = activation[:, 2:-2, 2:-2]
    return tf.reduce_mean(filter_activation)

# %% loss maximizatio nvia stochastic gradient ascent
@tf.function
def gradient_ascent_step(image, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image, filter_index)
    grads = tape.gradient(loss, image)
    grads = tf.math.l2_normalize(grads)
    image += learning_rate * grads
    return image

# %% function to generate filter visualizations
img_width = 200
img_height = 200

def generate_filter_pattern(filter_index):
    iterations = 30
    learning_rate = 10.
    image = tf.random.uniform(
        minval=0.4,
        maxval=0.6,
        shape=(1, img_width, img_height, 3))
    for i in range(iterations):
        image = gradient_ascent_step(image, filter_index, learning_rate)
    return image[0].numpy()


# %% utility function to convert a tensor to a valid image
def deprocess_image(image):
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    image = np.clip(image, 0, 255).astype("uint8")
    image = image[25:-25, 25:-25, :]
    return image
    

# %% printing patterns
plt.axis('off')
plt.imshow(deprocess_image(generate_filter_pattern(filter_index=2)))

# %% generating a grid of all filter reponse patterns in a layer
all_images = []
for filter_index in range(64):
    print(f"Processing filter {filter_index}")
    image = deprocess_image(
        generate_filter_pattern(filter_index)
    )
    all_images.append(image)

margin = 5
n = 8
cropped_width = img_width - 25 * 2
cropped_height = img_height - 25 * 2
width = n * cropped_width + (n - 1) * margin
height = n * cropped_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

for i in range(n):
    for j in range(n):
        image = all_images[i * n + j]
        stitched_filters[
            (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
            (cropped_height + margin) * j : (cropped_height + margin) * j
            + cropped_height,
            :,
        ] = image

keras.utils.save_img(
    f"filters_for_layer_{layer_name}.png", stitched_filters)


# %% convnet visualizations: visualizing heatmaps of class activation
model = keras.applications.xception.Xception(weights='imagenet')
img_path = keras.utils.get_file(
    fname='elephant.jpg',
    origin='https://img-datasets.s3.amazonaws.com/elephant.jpg',
)

def get_img_array(img_path, target_size):
    img = keras.utils.load_img(img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = keras.applications.xception.preprocess_input(array)
    return array

img_array = get_img_array(img_path, target_size=(299, 299))

# %% run pretrained network on image and decode its prediction
preds = model.predict(img_array)
print(keras.applications.xception.decode_predictions(preds, top=3)[0])

# %% setting up a model that returns the last convolutional output
last_conv_layer_name = 'block14_sepconv2_act'
classifier_layer_names = [
    'avg_pool',
    'predictions',
]
last_conv_layer = model.get_layer(last_conv_layer_name)
last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

# %% reapplying the classifier on top of the last convolutional output
classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in classifier_layer_names:
    x = model.get_layer(layer_name)(x)
classifier_model = keras.Model(classifier_input, x)

# %% retrieving the gradient of the top predicted class
import tensorflow as tf

with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_array)
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]

grads = tape.gradient(top_class_channel, last_conv_layer_output)

# %% gradient pooling and channel-importance weighting
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
last_conv_layer_output = last_conv_layer_output.numpy()[0]
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]
heatmap = np.mean(last_conv_layer_output, axis=-1)

# %% heatmap post-processing
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

# %% superimposing the heatmap on the original image
import matplotlib.cm as cm

img = keras.utils.load_img(img_path)
img = keras.utils.img_to_array(img)

heatmap = np.uint8(255 * heatmap)
