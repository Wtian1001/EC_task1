##################################################################################
# This file could contain the main methods for doing simulation, evolution, 
# the genetic algorithm, etc.
##################################################################################

import numpy as np

def initialize_generation(population_size, num_genes):
	"""
	Randomly initializes a generation.
	:param population_size: number of individuals in population
	:param num_genes: total number of weights in neural network controller
	"""
	return np.random.uniform(-1, 1, (population_size, num_genes))


def generate_next_generation(population_fitness):
	"""
	Generates next generation from current population.
	:param population_fitness: array containing each individual in population and their fitness score
	"""
	# should normalize each new vector between [-1, 1]
	return

def parent_selection(population_fitness):
	"""
	Returns a list of pairs with size |population| containing the selected parents that will reproduce.
	:param populaton_fitness: array containing each individual in population and their fitness score
	"""
	return

def recombine(x, y):
	"""
	Generate one offspring from individuals x and y using crossover.
	:param x: first individual ; numpy vector with 265 weights
	:param y: second individual ; numpy vector with 265 weights
	"""
	return

def mutate(x):
	"""
	Applies random mutation to individual.
	:param x: numpy vector with 265 weights
	"""

def compute_fitness(env, x):
	"""
	Evaluate the fitness of individual x.
	:param x: individual x
	"""
	fitness, player_life, enemy_life, time = env.play(pcont=x)
	return fitness