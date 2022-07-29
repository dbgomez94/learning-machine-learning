# %% python implementation of newton's method
import numpy as np
import autograd
from autograd import grad
from autograd import hessian

# a function
def tanh(w):                 # Define a function
    y = np.exp(-2.0 * w)
    return (1.0 - y) / (1.0 + y)

# newton's method
def newtons_method(g, max_iterations, w, epsilon=1e-7):
    gradient = grad(g)
    hess = hessian(g)       
    w_history = [w]
    g_history = [g(w)]
    for k in range(max_iterations):
        
        # evaluate the gradient and hessian
        grad_eval = gradient(w)
        hess_eval = hess(w)

        # reshape hessian to square matrix
        hess_eval.shape = (int((np.size(hess_eval))**(0.5)), int((np.size(hess_eval))**(0.5)))

        # solve second-order system for weight update (see eq. (4.17) on pg 84)
        a = hess_eval + epsilon * np.eye(w.size)
        b = np.dot(a, w) - grad_eval
        w = np.linalg.solve(a, b)

        # record weight and cost
        w_history.append(w)
        g_history.append(g(w))
    
    return w_history, g_history


