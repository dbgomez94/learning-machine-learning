# %% least squares linear regression
import numpy as np

# compute model: linear combination of input point
def model(x_p, w):
    mdl = w[0] + np.dot(x_p.T, w[1:])
    return mdl.T

# least squares with for loops
def least_squares_with_for_loops(w, x, y):
    # loop over points and compute cost contribution from each input / output pair
    cost = 0
    for p in range(y.size):
        # get pth input / output pair
        x_p = x[:, p][:, np.newaxis]
        y_p = y[p]
        # add to current cost
        cost += (model(x_p, w) - y_p) ** 2
    return cost / float(y.size)

# %% least squares the RIGHT way
def model(x, w):
    mdl = w[0] + np.dot(x.T, w[1:])  # alternatively can add a 1 to top of x
    return mdl.T

def least_squares(w, x, y):
    cost = np.sum((model(x, w) - y) ** 2)
    return cost / float(y.size)

# %%
