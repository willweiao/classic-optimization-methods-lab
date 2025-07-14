import numpy as np

# --- Rosenbrock Function (needs gradient) ---
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    grad = np.zeros(2)
    grad[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    grad[1] = 200 * (x[1] - x[0]**2)
    return grad

def rosenbrock_hess(x):
    hess = np.zeros((2, 2))
    hess[0, 0] = 2 - 400 * x[1] + 1200 * x[0]**2
    hess[0, 1] = -400 * x[0]
    hess[1, 0] = -400 * x[0]
    hess[1, 1] = 200
    return hess

# --- A simple quadratic function (good for testing Newton's) ---
def quadratic_fn(x):
    # f(x,y) = x^2 + 2y^2
    return x[0]**2 + 2 * x[1]**2

def quadratic_grad(x):
    return np.array([2 * x[0], 4 * x[1]])

def quadratic_hess(x):
    return np.array([[2, 0], [0, 4]])

# --- Bundle them for easy access ---
PROBLEMS = {
    'rosenbrock': {
        'func': rosenbrock,
        'grad': rosenbrock_grad,
        'hess': rosenbrock_hess,
        'x0': np.array([0.0, 0.0]) # Starting point
    },
    'quadratic': {
        'func': quadratic_fn,
        'grad': quadratic_grad,
        'hess': quadratic_hess,
        'x0': np.array([3.0, 4.0])
    }
}