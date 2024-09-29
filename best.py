###############################################################################
# EvoMan FrameWork - Test Best Solution                                       #
# Test file to evaluate the best solution from one run using `evaluate()`     #
###############################################################################

import sys
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import pandas as pd
import os
import EA1

# Choose the enemy
enemy = 3
loc = 149

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = f'EA1_enemy{enemy}'
n_hidden_neurons = 10

# Load the best solution from a previous run (adjust file path as needed)
best_solution_file = f'{experiment_name}/{experiment_name}.csv'

# Read the best solution for a specific run (you can modify this to select the run you want)
best_solutions_df = pd.read_csv(best_solution_file)
best_solution_str = best_solutions_df.loc[loc, 'BEST SOL']  # Change the index to select a different run
print(f'Best solution location: {best_solutions_df.loc[loc]}')
best_solution = np.array(eval(best_solution_str))  # Convert string back to a numpy array

# initializes simulation in individual evolution mode, for single static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# uses evaluate function to test one individual
def evaluate_best_solution(env, best_sol):
    # Evaluate the individual (reshape into 2D array to pass as population of one)
    best_sol = np.array([best_sol])  # Convert the solution into a 2D array (1xN)
    return EA1.evaluate(best_sol)

# Test the best solution by running it 5 times and taking the average gain
n_tests = 5
gains = []

for i in range(n_tests):
    result = evaluate_best_solution(env, best_solution)
    gain = result[0][1]  # Get the gain from the evaluation result
    gains.append(gain)
    print(f'Test {i+1}: Gain = {gain}')

# Calculate the average gain
average_gain = np.mean(gains)
print(f'Average gain over {n_tests} tests = {average_gain}')

# Save results to a file
test_results_df = pd.DataFrame({"Test Run": list(range(1, n_tests + 1)), "Gain": gains})
test_results_df.to_csv(f'{experiment_name}/test_best_solution.csv', index=False)

print(f'Test results saved to {experiment_name}/test_best_solution.csv')