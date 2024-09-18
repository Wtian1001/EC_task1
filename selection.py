import numpy as np
from specialist_arco import Specialist

# dummy values for testing
lowerbound = -1
upperbound = 1
population_size = 100
n_vars = 265
generation_num=100

# population is (100, 265) with real values ranging from -1 to 1
population = np.random.uniform(lowerbound, upperbound, (population_size, n_vars))
fitness_population = np.random.uniform(50, 100, population_size)
# end dummy values for testing



def roulette_wheel_selection(population, fitnesses):
    total_fitness = np.sum(fitnesses)
    selected_individuals = []
    
    for _ in range(population.shape[0]): 
        alpha = np.random.uniform(0, total_fitness)
        cumulative_sum = 0
        j = 0
        
        # selection
        while cumulative_sum < alpha and j < population.shape[0]:
            cumulative_sum += fitnesses[j]
            j += 1
            
        selected_individuals.append(population[j-1])
    
    return np.array(selected_individuals)



def stochastic_universal_sampling(population, fitnesses):
    mean_fitness = np.mean(fitnesses)
    alpha = np.random.rand()
    selected_individuals = []

    cumulative_sum = fitnesses[1]
    delta = alpha * mean_fitness
    j = 0

    while j < population.shape[0] - 1:
        if delta < cumulative_sum:
            selected_individuals.append(population[j])
            delta += cumulative_sum
        else:
            j += 1
            cumulative_sum += fitnesses[j]
    return np.array(selected_individuals)




def linear_rank_selection(population, fitnesses):
    sorted_indices = np.argsort(fitnesses)[::-1]
    ranks = np.zeros_like(sorted_indices)

    for rank, index in enumerate(sorted_indices, start=1):
        ranks[index] = rank

    probs = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        probs[i] = ranks[i] / (population.shape[0] * (population.shape[0] - 1))

    value = 1 / (population.shape[0] - 2.001)
    selected_individuals = []

    for i in range(population.shape[0]):
        alpha = np.random.uniform(0, value)
        for j in range(population.shape[0]):
            if probs[j] < alpha:
                selected_individuals.append(population[j])
                break
    return np.array(selected_individuals)



def selection_score(population, generation, fitness_population):
    ''' 
    Selection based on the dynamic approach from
    'Parent Selection Operators for Genetic Algorithms'
    Input: current population, current generation number, fitness of current population
    Output: selection criterion
    '''
    best_idx = np.argmax(fitness_population)
    best = population[best_idx]
    criteria1 = 0
    pop_size = population.shape[0]

    # Hamming distance is binary so we use Euclidean distance instead
    for individual in population:
        criteria1 += np.linalg.norm(best - individual)
    criteria1 /=  pop_size # normalize
    criteria1 = np.exp(- criteria1 / generation) # decrease over generations

    max_fitness = np.max(fitness_population)
    min_fitness = np.min(fitness_population)
    criteria2 = max_fitness / (max_fitness **2 + min_fitness**2) # maximisation problem
    
    criterion = 1/generation * criteria1 + ((generation-1)/generation) * criteria2

    return criterion



def dynamic_selection(population, fitnesses, generation):
    specialist = Specialist(
        experiment_name='optimization_test_arco', 
        population_size=50
    )
    fitnesses = specialist.fitness_eval(population)

    rws_population = roulette_wheel_selection(population, fitnesses)
    rws_fitness = specialist.fitness_eval(rws_population)
    rws_score = selection_score(rws_population, generation, rws_fitness)

    sus_population = stochastic_universal_sampling(population, fitnesses)
    sus_fitness = specialist.fitness_eval(sus_population)
    sus_score = selection_score(sus_population, generation, sus_fitness)

    lrs_population = linear_rank_selection(population, fitnesses)
    lrs_fitness = specialist.fitness_eval(lrs_population)
    lrs_score = selection_score(lrs_population, generation, lrs_fitness)

    scores = [rws_score, sus_score, lrs_score]
    new_populations = [rws_population, sus_population, lrs_population]
    new_fitnesses = [rws_population, sus_population, lrs_population]
    best = np.argmax(scores)

    return new_populations[best], new_fitnesses[best]


# print(parent_selection(population, generation_num, fitness_population))
# print(roulette_wheel_selection(population, fitness_population).shape)
print(dynamic_selection(population, fitness_population, generation_num))
