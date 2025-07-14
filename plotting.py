import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_paths(problem, results_dict, title):
    """
    Creates a contour plot of the problem and overlays the convergence paths of solvers.

    Args:
        problem (dict): The problem dictionary containing 'func' and 'x0'.
        results_dict (dict): A dictionary where keys are solver names and values are
                             dictionaries containing the 'history' of points.
        title (str): The title for the plot.
    """
    # 1. Create a grid of points to evaluate the function
    # We create a plotting boundary around all points visited by the optimizers
    all_x = np.concatenate([res['history'][:, 0] for res in results_dict.values()])
    all_y = np.concatenate([res['history'][:, 1] for res in results_dict.values()])
    x_min, x_max = all_x.min() - 0.5, all_x.max() + 0.5
    y_min, y_max = all_y.min() - 0.5, all_y.max() + 0.5
    
    x_grid = np.linspace(x_min, x_max, 400)
    y_grid = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = problem['func'](np.array([X, Y]))

    # 2. Create the contour plot
    plt.figure(figsize=(12, 9))
    # Use logspace for levels to better visualize steep functions like Rosenbrock
    levels = np.logspace(0, 5, 35)
    plt.contour(X, Y, Z, levels=levels, cmap='viridis')
    plt.colorbar(label='Objective Function Value')

    # 3. Plot the path for each solver
    for name, result in results_dict.items():
        history = result['history']
        plt.plot(history[:, 0], history[:, 1], '-o', markersize=3, label=name)

    # 4. Mark the start and true minimum (if known)
    plt.plot(problem['x0'][0], problem['x0'][1], 'kx', markersize=12, label='Start Point')
    # Assuming minimum is at (1,1) for Rosenbrock for this example
    if 'rosenbrock' in problem['func'].__name__:
        plt.plot(1, 1, 'g*', markersize=15, label='True Minimum')

    # 5. Final plot adjustments
    plt.title(title, fontsize=16)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_objective_values(results_dict, title):
    """
    Plots the objective function value versus the iteration number for each solver.

    Args:
        results_dict (dict): A dictionary where keys are solver names and values are
                             dictionaries containing the 'obj_values' list.
        title (str): The title for the plot.
    """
    plt.figure(figsize=(12, 8))

    # Plot the objective values for each solver
    for name, result in results_dict.items():
        # The x-axis is simply the iteration number (the index of the list)
        plt.plot(result['obj_values'], label=name)

    # Use a logarithmic scale for the y-axis to see convergence details
    plt.yscale('log')
    
    # Final plot adjustments
    plt.title(title, fontsize=16)
    plt.xlabel('Iteration Number')
    plt.ylabel('Objective Function Value (log scale)')
    plt.legend()
    plt.grid(True)
    plt.show()