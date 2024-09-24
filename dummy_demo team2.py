################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys
from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs, sqrt
import glob, os
import itertools

# ATTENTION: To train change headless to true, visuals(within env) to false and run_mode to train job

# choose this for not using visuals and thus making experiments faster
headless = False
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'team_2_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
# Generate all possible combinations of enemies and levels
enemies_list = [1]
levels_list = [2]
enemy_level_combinations = list(itertools.product(enemies_list, levels_list))

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemymode="static",
                  speed="fastest",
                  sound="off",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  visuals=True)

# genetic algorithm params
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

run_mode = 'test'

dom_u = 1
dom_l = -1
npop = 100
gens = 50
mutation_weight = 0.1
n_parents = 2
k = 3  # Tournament size
num_offspring = 50

# Island Model parameters
num_islands = 5  # Number of islands
migration_interval = 25  # Migrate every 25 generations
migration_size = 3  # Number of individuals to migrate
mutation_rate = 0.2


# Evaluate fitness
def evaluate_population(population):
    fitness_scores = []
    for individual in population:
        f, p, e, t = env.play(pcont=individual)
        fitness_scores.append(f)
    return fitness_scores


# Tournament selection
def tournament_selection(population, fitness_scores, k):
    selected_parents = []
    for _ in range(0, len(population), 2):
        tournament_indices = np.random.choice(len(population), k, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected_parents.append(population[winner_index])
    return selected_parents



def select_parents(population, fitness_scores, k=3):
    return tournament_selection(population, fitness_scores, k)


# Multi-parent recombination type 2
def multi_parent_recombination(parents):
    child = np.zeros(n_vars)
    for i in range(n_vars):
        child[i] = np.mean([parent[i] for parent in parents])
    return child


# Mutatie
def mutate(child, mutation_rate, mutation_weight):
    for i in range(n_vars):
        if np.random.rand() < mutation_rate:
            child[i] += np.random.uniform(-mutation_weight, mutation_weight)
    return child


# Evolution of a single population (island)
def evolve_single_population(population, fitness_scores, mutation_rate, num_offspring=50, mutation_weight=0.1, k=3, n_parents=2):
    offspring = []
    # Gebruik toernooiselectie voor het selecteren van ouders
    selected_parents = select_parents(population, fitness_scores, k)
    for _ in range(num_offspring):
        parent_indices = np.random.choice(len(selected_parents), n_parents, replace=False)
        parents = [selected_parents[i] for i in parent_indices]
        child = multi_parent_recombination(parents)
        mutated_child = mutate(child, mutation_rate, mutation_weight)
        offspring.append(mutated_child)

    new_population = population + offspring
    new_fitness_scores = evaluate_population(new_population)

    combined = list(zip(new_population, new_fitness_scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    population = [ind for ind, fitness in combined[:npop]]
    fitness_scores = [fitness for ind, fitness in combined[:npop]]

    return population, fitness_scores


# Migration between islands
def migrate(populations, fitness_scores_list, migration_size):
    for i in range(num_islands):
        source_island = i
        target_island = (i + 1) % num_islands  # Ring topology

        combined = list(zip(populations[source_island], fitness_scores_list[source_island]))
        combined.sort(key=lambda x: x[1], reverse=True)
        migrants = [ind for ind, fit in combined[:migration_size]]

        combined_target = list(zip(populations[target_island], fitness_scores_list[target_island]))
        combined_target.sort(key=lambda x: x[1])  # Sort ascending (worst first)
        for j in range(migration_size):
            populations[target_island][j] = migrants[j]
            fitness_scores_list[target_island][j] = fitness_scores_list[source_island][j]

    return populations, fitness_scores_list


# Main loop for training
if run_mode == 'train':
    for enemy, level in enemy_level_combinations:
        env.update_parameter('enemies', [enemy])
        env.update_parameter('level', level)

        if not os.path.exists(experiment_name + '/evoman_solstate'):
            print('\nNEW EVOLUTION\n')

            # Initialize populations for each island
            populations = [
                [np.random.uniform(dom_l, dom_u, n_vars) for _ in range(npop)]
                for _ in range(num_islands)
            ]

            # Evaluate fitness for each population
            fitness_scores_list = [evaluate_population(population) for population in populations]
            ini_g = 0

        else:
            print('\nCONTINUING EVOLUTION\n')

            env.load_state()
            populations = env.solutions[0]
            fitness_scores_list = env.solutions[1]

            with open(experiment_name + '/gen.txt', 'r') as file_aux:
                ini_g = int(file_aux.readline().strip())

        for generation in range(ini_g, gens):
            print(f"\nGeneration {generation}")

            for i in range(num_islands):
                print(f"  Evolving Island {i + 1}")
                populations[i], fitness_scores_list[i] = evolve_single_population(
                    populations[i],
                    fitness_scores_list[i],
                    mutation_rate,
                    num_offspring,
                    mutation_weight,
                    k,
                    n_parents
                )

            if generation % migration_interval == 0:
                print("  Migration between islands")
                populations, fitness_scores_list = migrate(populations, fitness_scores_list, migration_size)

            # Collect statistics
            all_fitness = [fitness for fitness_scores in fitness_scores_list for fitness in fitness_scores]
            all_individuals = [ind for population in populations for ind in population]
            best_index = np.argmax(all_fitness)
            best_fitness = all_fitness[best_index]
            best_individual = all_individuals[best_index]
            mean_fitness = np.mean(all_fitness)
            std_fitness = np.std(all_fitness)

            # Save the generation results
            with open(experiment_name + '/results.txt', 'a') as file_aux:
                print(f' GENERATION {generation} {round(best_fitness, 6)} {round(mean_fitness, 6)} {round(std_fitness, 6)}')
                file_aux.write(f'\n{generation} {round(best_fitness, 6)} {round(mean_fitness, 6)} {round(std_fitness, 6)}')

            # Save the best individual for this generation
            np.save(f"{experiment_name}/best_individual_gen_{generation}.npy", best_individual)
            if best_fitness > fitness_scores_list[0][0]:  # Save if improvement
                np.savetxt(experiment_name + '/best.txt', best_individual)

            # Save the current state
            solutions = [populations, fitness_scores_list]
            env.update_solutions(solutions)

            with open(experiment_name + '/gen.txt', 'w') as file_aux:
                file_aux.write(str(generation))


# Test the best solution
elif run_mode == 'test':
    for enemy, level in enemy_level_combinations:
        try:
            # Load the best solution
            best_sol = np.load(experiment_name + '/best_individual_gen_49.npy')  # Adjust generation as needed

            print('\nRUNNING SAVED BEST SOLUTION\n')

            # Update environment parameters
            env.update_parameter('enemies', [enemy])
            env.update_parameter('level', level)
            env.update_parameter('speed', 'normal')

            # Evaluate the best solution
            f, p, e, t = env.play(pcont=best_sol)
            print(f"Fitness: {f}, Player Life: {p}, Enemy Life: {e}, Time: {t}")

        except Exception as e:
            print(f"Error loading best solution: {e}")
            sys.exit(1)
