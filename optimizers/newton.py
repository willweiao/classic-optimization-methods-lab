import numpy as np

def solve(problem, x0, tol=1e-6, max_iter=100): # Newton converges fast, needs fewer iter
    """
    Solves an optimization problem using Newton's Method.
    
    Args:
        problem (dict): A dictionary containing 'func', 'grad', and 'hess'.
        x0 (np.array): The starting point.
    
    Returns:
        tuple: (final point, history of points, history of objective values)
    """
    f = problem['func']
    grad_f = problem['grad']
    hess_f = problem['hess']

    x = x0.copy()
    history = [x]
    obj_values = [f(x)]
    
    # TODO:
    ########################################################################################
    for i in range(max_iter):
        gradient = grad_f(x)
        hessian = hess_f(x)

        if np.linalg.norm(gradient) < tol:
            print(f"Newton converged after {i} iterations.")
            break
        
        # Key step: Solve the linear system H*p = -g
        # We use np.linalg.solve for numerical stability instead of inverting
        p = np.linalg.solve(hessian, -gradient)
        
        # For simplicity, we use a fixed step size of 1 (the "pure" Newton step)
        # A more robust version would add a line search here as well!
        alpha = 1.0 
        
        x = x + alpha * p
        history.append(x)
        obj_values.append(f(x))
    #########################################################################################

    return x, np.array(history), obj_values