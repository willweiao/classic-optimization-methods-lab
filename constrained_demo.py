import numpy as np
from optimizers import gradient_descent
import matplotlib.pyplot as plt

# == Problem setting==
def objective_function(x):
    #f(x,y) = (x-2)^2 + (y-2)^2
    return (x[0] - 2)**2 + (x[1] - 2)**2

def objective_gradient(x):
    # Gradient of f(x,y)
    return np.array([2 * (x[0] - 2), 2 * (x[1] - 2)])

def constraint_function(x):
    # g(x,y) = x^2 + y^2 - 1 <= 0
    return x[0]**2 + x[1]**2 - 1

def constraint_gradient(x):
    # Gradient of g(x,y)
    return np.array([2 * x[0], 2 * x[1]])

# create the peanlty method
def create_penalty_problem(f, grad_f, g, grad_g, rho):
    """
    Creates a new, unconstrained problem using the penalty method.
    The new objective is P(x) = f(x) + rho * max(0, g(x))^2
    """
    def penalty_objective(x):
        # The max(0, g(x)) term ensures we only penalize constraint violations
        return f(x) + rho * max(0, g(x))**2

    def penalty_gradient(x):

        # Using the chain rule to calculate Gradient of P(x).The derivative of max(0, z)^2 is 2 * max(0, z) * z'
        g_val = g(x)
        if g_val <= 0:
            # Inside the feasible region, penalty has no effect
            return grad_f(x)
        else:
            # Outside the feasible region, add the penalty's gradient
            return grad_f(x) + 2 * rho * g_val * grad_g(x)
            
    # Return a problem dictionary in the format our solvers expect
    return {'func': penalty_objective, 'grad': penalty_gradient}

# == verify with kkt condition and visualize ==
def run_kkt_demo():
    print("="*60)
    print("KKT Conditions & Penalty Method Demonstration")
    print("="*60)

    # The true solution we found analytically which I show the process on my blog
    analytical_solution = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    print(f"Analytical KKT Solution: {analytical_solution}")
    print("-" * 60)

    # A starting point outside the feasible region
    x0 = np.array([2.0, 0.0])
    last_solution = x0
  
    # List of penalty parameters to test
    rho_values = [1, 10, 100, 1000, 10000]
    
    numerical_solutions = []

    for rho in rho_values:
        print(f"Solving with penalty rho = {rho}...")
        
        # Create the unconstrained problem for the current rho
        penalty_problem = create_penalty_problem(
            objective_function, objective_gradient,
            constraint_function, constraint_gradient,
            rho
        )
        
        # Solve using our existing Gradient Descent optimizer
        # Use the solution from the previous, easier problem as the starting point, fuctioning as a warm start
        solution, history, _ = gradient_descent.solve(penalty_problem, last_solution, max_iter=2000)
        last_solution = solution
        
        numerical_solutions.append(history) # Store the full path for plotting
        
        # Calculate the error (distance from the true solution)
        error = np.linalg.norm(solution - analytical_solution)
        
        print(f"  Numerical solution: {solution}")
        print(f"  Error (L2 norm):   {error:.6f}\n")

    # Visualize the Results
    plot_kkt_demo(numerical_solutions, rho_values)

def plot_kkt_demo(solutions, rhos):
    """Visualizes the convergence of the penalty method."""
    plt.figure(figsize=(10, 10))
    
    # Plot the constraint boundary (the unit circle)
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Constraint Boundary (g(x)=0)')

    # Plot the path for each rho
    for i, history in enumerate(solutions):
        plt.plot(history[:, 0], history[:, 1], '-o', markersize=3, label=f'Path for rho={rhos[i]}')

    # Mark key points
    plt.plot(2, 2, 'g*', markersize=15, label='Original Objective Minimum')
    analytical_sol = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    plt.plot(analytical_sol[0], analytical_sol[1], 'rX', markersize=15, label='Analytical KKT Solution')

    plt.title('Penalty Method Convergence to KKT Point', fontsize=16)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# == run the demonstration ==
if __name__ == "__main__":
    run_kkt_demo()
