# TODO: implement the BFGS method
#########################################################################################
import numpy as np

def solve(problem, x0, tol=1e-6, max_iter=100):
    """
    Solves an optimization problem using BFGS.
    
    Args:
        problem (dict): A dictionary containing 'func' and 'grad'.
        x0 (np.array): The starting point.
    
    Returns:
        tuple: (final point, history of points, history of objective values)
    """
    f = problem['func']
    grad_f = problem['grad']
    
    x = x0.copy()
    n=len(x0)
    H=np.eye(n)

    history = [x]
    obj_values = [f(x)]

    for k in range(max_iter):
        g =grad_f(x)

        if np.linalg.norm(g) < tol:
            print(f"BFGS converged after {k} iterations.")
            break

        p = np.dot(-H, g)
        alpha = 1.0
        while f(x + alpha * p) > f(x) + 0.1 * alpha * np.dot(g, p):
            alpha *= 0.8

        x_new = x + alpha * p
        s = x_new - x
        y = grad_f(x_new) - g
        yT = y.T
        yT_s = np.dot(yT, s)

        if yT_s <= 1e-10:
            print("Curvature condition failed.")
            break

        rho = 1.0 / yT_s
        I = np.eye(n)
        H = np.dot(I - rho * np.outer(s, y), np.dot(H, I - rho * np.outer(y, s))) + rho * np.outer(s, s)

        history.append(x_new)
        obj_values.append(f(x_new))

        x=x_new
    
    return x, np.array(history), obj_values