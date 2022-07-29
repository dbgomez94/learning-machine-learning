# %% Practicing plotting surfaces in 3D and corresponding contour plots in 2D
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 11, 1)
y = np.arange(-10, 11, 1)

x, y = np.meshgrid(x, y)
z = x**2 + y**2 + 0

fig1 = plt.figure(figsize=(10, 10))
ax1 = fig1.add_subplot(211, projection='3d')
ax1.plot_surface(x, y, z)
ax1.contour(x, y, z, zdir='z', offset=0)  # *** there must be an offset

fig2 = plt.figure(figsize=(4, 4))
ax2 = fig2.add_subplot(111)
ax2.contour(x, y, z)

plt.show()

# %%
np.random.rand()
np.random.uniform()

# %% Gradient Descent

def g(w): return (1/50) * (w**4 + w**2 + 10*w)
def dg(w): return (1/50) * (4*w**3 + 2*w + 10)

w = 2
alpha = 1
steps = 50

w_history = []
g_history = []

for step in range(steps):
    # step 1: compute negative gradient
    neg_grad = -dg(w)
    # step 2: take step in that direction
    w += alpha * neg_grad
    # step 3: repeat
    w_history.append(w)
    g_history.append(g(w))


plt.figure()
x = np.linspace(-3, 3, 50)
y = g(x)
plt.plot(x, y, 'k', zorder=-1)
plt.scatter(
    w_history, g_history, 
    marker='X', 
    c=np.arange(steps), 
    cmap='RdYlGn_r', 
    alpha=1, 
    zorder=1,
)





