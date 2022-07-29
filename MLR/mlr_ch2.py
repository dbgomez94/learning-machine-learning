# %% P2.1 Minimizing a quadratic function and the curse of dimensionality
import numpy as np
import matplotlib.pyplot as plt

# consider the quadratic function g = np.dot(w, w)
# create a range of these quadratic function for n = 1 to 32
# sample the inputs space of each quadratic p = 100, 1000, 10000 times uniformly on the hypercube
# plot min val attained for each quadratic vs input dimension
dimensions = np.arange(1, 32, 1)  # dimension of w
plt.figure()
for samples in [100, 1000, 10000]:
    min_g = []
    for n in dimensions:
        m = int(samples ** (1/n))
        w = np.linspace(-1, 1, m)
        meshgrids = np.meshgrid(*[w for i in range(n)])
        m_tuple = tuple([m for i in range(n)])
        g = np.zeros(shape=m_tuple)
        for meshgrid in meshgrids:
            g += meshgrid ** 2
        min_g.append(np.min(g))
    plt.plot(dimensions, min_g)
plt.xlabel('Number of Dimensions')
plt.ylabel('Minimun value of quadradic')
plt.title('The Curse of Dimensionality\n(the minimum value is 0)')
plt.show()

   
# %% P2.2 Implementing a random search in python
# implement random search from Example 2.4

def random_search_1d(g_func, w0, steps, alpha):
    w = w0
    g = g_func(w0)
    w_search = [w]
    g_search = [g]
    for step in range(steps):
        g_left = g_func(w - alpha)
        g_right = g_func(w + alpha)
        next_step = np.argmin([g_left, g_right])
        if next_step == 0:
            w -= alpha
            w_search.append(w)
            g_search.append(g_left)
        else:
            w += alpha
            w_search.append(w)
            g_search.append(g_right)

    return w_search, g_search


def g(w):
    return np.sin(3 * (w)) + 0.3 * ((w) ** 2)

w = np.arange(start=-5, stop=5, step=0.01)
g_curve = g(w)


w_rs, g_rs = random_search_1d(g_func=g, w0=4.5, steps=10, alpha=0.1)

plt.figure()
plt.plot(w, g_curve, c='k', zorder=-1)
plt.scatter(
    w_rs, g_rs, 
    marker='X', 
    c=np.arange(len(w_rs)), 
    cmap='RdYlGn_r', 
    alpha=1, 
    zorder=1,
)
plt.scatter(
    w_rs, np.zeros((len(w_rs))),
    marker='o', 
    c=np.arange(len(w_rs)), 
    cmap='RdYlGn_r', 
    alpha=1, 
    zorder=1,
)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$g(\omega)$')
plt.show()

# %% P2.3 Using random search to minimize a nonconvex function

def random_search(g_func, w0, n_dir, steps, alpha):

    # get dimension of input space
    if type(w0) is list:
        n_dim = len(w0)  # number of dimensions of input space
    else:
        n_dim = 1

    # the actual min path taken
    w = w0
    g = g_func(w)
    w_min_path = [w]
    g_min_path = [g]
    
    # check for n_dim == 1
    if n_dim == 1:
        for step in range(steps):
            g_left = g_func(w - alpha)
            g_right = g_func(w + alpha)
            next_step = np.argmin([g_left, g_right])
            if next_step == 0:
                w -= alpha
                w_min_path.append(w)
                g_min_path.append(g_left)
            else:
                w += alpha
                w_min_path.append(w)
                g_min_path.append(g_right)
    else:
        # iterate through k steps
        for step in range(steps):
            d_search = []
            w_search = []
            g_search = []
            # iterate through n_dir random directions to find the best one
            for i in range(n_dir):
                d_rand = np.random.uniform(low=-1, high=1, size=n_dim)  # pick a random n_dim vector
                d_rand /= np.linalg.norm(d_rand)  # normalize it
                w_rand = w + alpha * d_rand  # new random point to evaluate
                g_rand = g_func(w_rand)  # new random value of function
                d_search.append(d_rand)  # update the lists with computed values
                w_search.append(w_rand)
                g_search.append(g_rand)
            if np.any(g_search < g):  # if any search points lower than function at current point
                d_next = d_search[np.argmin(g_search)]  # select the best direction
                w += alpha * d_next  # take step in best direction
                g = g_func(w)  # compute the value of the function at new point
            w_min_path.append(w)
            g_min_path.append(g)

    return w_min_path, g_min_path


def g1(w):
    return np.sin(3 * (w)) + 0.3 * ((w) ** 2)

def g2(w):
    return np.tanh(4 * w[0] + 4 * w[1]) + np.max([0.4 * w[0] ** 2, 1]) + 1

w_min_path, g_min_path = random_search(g_func=g2, w0=[2, 2], n_dir=100, steps=100, alpha=1)

plt.figure()
plt.plot(np.arange(len(g_min_path)), g_min_path)
plt.show()

# %% P2.4 Random search with diminishing step-length
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

# Rosenbrock function
def g3(x1, x2):
    return 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
x_vec = np.arange(-2, 2, 0.25)
y_vec = np.arange(-2, 2, 0.25)
x_mesh, y_mesh = np.meshgrid(x_vec, y_vec)
z = g3(x_mesh, y_mesh)

# Plot the surface.
surf = ax.plot_surface(
    x_mesh, y_mesh, z, 
    cmap=cm.coolwarm,
    linewidth=0, 
    antialiased=False,
)

ax.contour(x_mesh, y_mesh, z)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# %% 
