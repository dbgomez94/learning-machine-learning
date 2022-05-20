import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# using GradientTape to compute the gradient of y = x^2
x = tf.Variable(initial_value=3.0)
with tf.GradientTape() as tape:
    y = tf.square(x)
gradient = tape.gradient(y, x)
print(gradient)

# linear classifier in pure TensorFlow
 
# generating two classes of random points in 2D plane
num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class,
)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class,
)

# stacking the two classes into an array with shape (2000, 2)
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

# generating the corresponding targets (0, 1)
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype='float32'), np.ones((num_samples_per_class, 1), dtype='float32')))

# plotting the two point classes
plt.scatter(
    x=inputs[:, 0],
    y=inputs[:, 1],
    c=targets[:, 0],
)

# creating the linear classifier variables
input_dim = 2
output_dim = 1
w = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

# forward pass function
def model(inputs):
    return tf.matmul(inputs, w) + b


# means square error loss function
def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)


# training step funciton
learning_rate = 0.1
def training_step(inputs, targets):
    # forward pass inside a gradient tape loop
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    # gradient of loss wrt to weights and biases
    grad_loss_wrt_w, grad_loss_wrt_b = tape.gradient(loss, [w, b])
    # updated parameters
    w.assign_sub(grad_loss_wrt_w * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss

# batch training loop
for step in range(40):
    loss = training_step(inputs, targets)
    print(f'loss at step {step}: {loss: .4f}')

# plotting the trained classifier
x = np.linspace(-1, 4, 100)
y = -w[0] / w[1] * x + (0.5 - b) / w[1]
plt.plot(x, y, '-k')
plt.scatter(
    x=inputs[:, 0],
    y=inputs[:, 1],
    c=targets[:, 0],
)
plt.show()