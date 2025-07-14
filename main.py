import numpy as np
from problems import PROBLEMS
from optimizers import gradient_descent, newton, bfgs
from plotting import plot_convergence_paths, plot_objective_values

# --- CONFIGURATION ---
PROBLEM_NAME = 'quadratic'  # Options: 'rosenbrock', 'quadratic'

# Define which solvers to run and compare.
SOLVERS = {
    'Gradient Descent': gradient_descent,
    'Newton\'s Method': newton,
    'BFGS': bfgs
}


def main():
    """
    Main execution function to run the optimization comparison.
    """
    # 1. Load the selected problem from our library
    try:
        problem = PROBLEMS[PROBLEM_NAME]
    except KeyError:
        print(f"Error: Problem '{PROBLEM_NAME}' is not defined in problems.py.")
        print(f"Available problems are: {list(PROBLEMS.keys())}")
        return

    x0 = problem['x0']
    
    print("="*50)
    print(f"Optimizing Problem: '{PROBLEM_NAME.title()}'")
    print(f"Starting Point x0 = {x0}")
    print("="*50)

    # 2. Run each solver and store its results
    results = {}
    for name, solver_module in SOLVERS.items():
        print(f"\n--- Running: {name} ---")
        try:
            # All solvers share the same 'solve' interface, making this simple
            solution, history, obj_values = solver_module.solve(problem, x0)
            
            # Store results for plotting
            results[name] = {'history': history, 'obj_values': obj_values}
            
            # Print summary
            print(f"  Status: Converged")
            print(f"  Found Minimum: {solution}")
            print(f"  Objective Value: {obj_values[-1]:.6f}")
            print(f"  Iterations: {len(obj_values) - 1}")

        except np.linalg.LinAlgError:
            print(f"  Status: Failed")
            print("  Reason: Newton's Method failed. The Hessian matrix was singular and could not be inverted.")
        except Exception as e:
            print(f"  Status: Failed")
            print(f"  An unexpected error occurred: {e}")

    # 3. Generate comparison plots if any solvers succeeded
    if not results:
        print("\nNo solvers completed successfully. No plots will be generated.")
        return

    print("\n--- Generating Plots ---")
    
    # Generate a plot showing the convergence paths on a contour map
    plot_convergence_paths(
        problem=problem,
        results_dict=results,
        title=f"Convergence Paths on '{PROBLEM_NAME.title()}'"
    )

    # Generate a plot comparing the objective function decrease over iterations
    plot_objective_values(
        results_dict=results,
        title=f"Objective Value vs. Iteration for '{PROBLEM_NAME.title()}'"
    )

    print("Plots displayed. Close plot windows to terminate the program.")


if __name__ == "__main__":
    main()