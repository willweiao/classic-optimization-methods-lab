import numpy as np

def backtracking_line_search(f, grad_f, x, p, c=0.1, tau=0.8, alpha_init=1.0):
    """
    Performs backtracking line search to find a step size alpha.
    f: The objective function
    grad_f: The gradient of the objective function
    x: The current point (numpy array)
    p: The search direction (numpy array)
    c: Armijo condition control parameter
    tau: Reduction factor for alpha
    alpha_init: Initial step size
    """
    # TODO:
    ##################################################################################
    alpha = alpha_init
    # The term grad_f(x).T @ p is the directional derivative
    while f(x + alpha * p) > f(x) + c * alpha * np.dot(grad_f(x), p):
        alpha *= tau
    return alpha
    ###################################################################################

def solve(problem, x0, tol=1e-6, max_iter=1000):
    """
    Solves an optimization problem using Gradient Descent.
    
    Args:
        problem (dict): A dictionary containing 'func' and 'grad'.
        x0 (np.array): The starting point.
    
    Returns:
        tuple: (final point, history of points, history of objective values)
    """
    f = problem['func']
    grad_f = problem['grad']
    
    x = x0.copy()
    history = [x]
    obj_values = [f(x)]
    
    # TODO:
    ######################################################################################
    for i in range(max_iter):                                                            
        gradient = grad_f(x)
        
        # 1. Convergence Check
        if np.linalg.norm(gradient) < tol:
            print(f"Converged after {i} iterations.")
            break
            
        # 2. Determine Search Direction
        p = -gradient
        
        # 3. Find Step Size using Backtracking
        alpha = backtracking_line_search(f, grad_f, x, p)
        
        # 4. Update the point
        x = x + alpha * p
        
        history.append(x)
        obj_values.append(f(x))
    #########################################################################################

    else: # This 'else' belongs to the 'for' loop, runs if loop finishes without break
        print("Did not converge within max_iter.")

    return x, np.array(history), obj_values