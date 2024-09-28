###############################################################################
# EvoMan FrameWork 			                                                  #
# DEMO : Neuroevolution - Genetic Algorithm                                   #
# Author: Group 18		                                                      #
###############################################################################

# Import framwork and other libs
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import os
import random
import pandas as pd
import csv

# Choose the enemy
enemy = 3

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = f'EA1_enemy{enemy}'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# default environment fitness is assumed for experiment
env.state_to_log() # checks environment state

ini = time.time()  # sets time marker

# genetic algorithm params
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5 # number of weights for multilayer with 10 hidden neurons

dom_u = 1
dom_l = -1
npop = 100
gens = 30
mutation_rate = 0.1

n_runs = 10
tournament_size = 5

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    gain = p - e
    return f, gain

# normalizes
def norm(x, pfit_pop):

    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

# tournament selection in 10 random individuals
def tournament(pop, fit_pop , tournament_size=10):
    
    # Randomly select individuals for the tournament
    indices = np.random.choice(len(pop), tournament_size, replace=False)
    tournament_fits = fit_pop[indices]

    # Select the individual with the highest fitness
    winner_index = indices[np.argmax(tournament_fits)]
    return pop[winner_index]

# limits
def limits(x):

    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x
    
# crossover
def crossover(pop, fit_pop, tournament_size, alpha=0.5):

    total_offspring = np.zeros((0,n_vars))

    for p in range(0, pop.shape[0], 2):
        p1 = tournament(pop, fit_pop, tournament_size)
        p2 = tournament(pop, fit_pop, tournament_size)
        
        n_offspring =   np.random.randint(1,3+1, 1)[0]
        offspring =  np.zeros( (n_offspring, n_vars) )

        for f in range(0,n_offspring):

            cross_prop = np.random.uniform(0,1)
            offspring[f] = p1*cross_prop+p2*(1-cross_prop)

            # mutation
            for i in range(0,len(offspring[f])):
                if np.random.uniform(0 ,1)<=mutation_rate:
                    offspring[f][i] =   offspring[f][i]+np.random.normal(0, 1)

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring

# nonuniform mutation
def mutation(offspring, mutation_rate, dom_u, dom_l):
    # apply non-uniform mutation to each offspring
    for i in range(len(offspring)):
        if np.random.uniform(0, 1) <= mutation_rate:
            offspring[i] += np.random.normal(0, 1)
            offspring[i] = limits(offspring[i])
    return offspring


if __name__ == "__main__": 
    indices_run     = []
    indices_gen     = []
    
    best_gain       = []
    best_fit        = []
    mean_fitness    = []
    std_fitness     = []
    best_solutions  = []
    game_lostwon    = []

    result_matrix_max=np.zeros((n_runs,gens))
    result_matrix_mean=np.zeros((n_runs,gens))
    
    # run the algorithm n_runs times
    for r in range(n_runs):
        i = 0

        pop = np.random.uniform(dom_u, dom_l, (npop, n_vars))
        fit_pop_gain = evaluate(pop)
        fit_pop = fit_pop_gain[:,0]
        pop_gain = fit_pop_gain[:,1]
        
        best = np.argmax(fit_pop)
        best_solution = pop[best].tolist()
        mean = np.mean(fit_pop)
        std  = np.std(fit_pop)
        
        # write result
        print('\n RUN '+str(r)+ ' GENERATION '+str(i)+'  '+str(round(pop_gain[best],6))+'  '+str(round(fit_pop[best],6))+'  '+str(round(mean,6))+'  '+str(round(std,6)))    
        experiment_data  = open(experiment_name+'/results.txt','a')
        experiment_data.write('\n RUN '+str(r)+ ' GENERATION '+str(i)+'  '+str(round(pop_gain[best],6))+'  '+str(round(fit_pop[best],6))+'  '+str(round(mean,6))+'  '+str(round(std,6)))
        experiment_data.close()

        result_matrix_max[r,i]=np.max(fit_pop)
        result_matrix_mean[r,i]=np.mean(fit_pop)
        
        indices_run.append(r)
        indices_gen.append(i)
        
        best_gain.append(pop_gain[best])
        best_fit.append(fit_pop[best])
        mean_fitness.append(mean)
        std_fitness.append(std)
        best_solutions.append(best_solution)

        # evolution
        for i in range(1,gens):
            # Create offspring applying crossover and mutation
            offspring = crossover(pop, fit_pop, tournament_size)
            offspring = [mutation(gene, mutation_rate, dom_u, dom_l) for gene in offspring]

            # Evaluate offspring
            fit_offspring_gain = evaluate(offspring)
            fit_offspring = fit_offspring_gain[:,0]
            offspring_gain = fit_offspring_gain[:,1]

            # Combine population and offspring
            pop = np.vstack((pop, offspring))
            fit_pop = np.append(fit_pop, fit_offspring)
            pop_gain = np.append(pop_gain, offspring_gain)

            # Select the best individuals
            sorted_indices = np.argsort(fit_pop)[::-1]
            pop = pop[sorted_indices[:npop]]
            fit_pop = fit_pop[sorted_indices[:npop]]
            pop_gain = pop_gain[sorted_indices[:npop]]
            
            best = np.argmax(fit_pop)
            best_solution = pop[best].tolist()
            mean = np.mean(fit_pop)
            std  =  np.std(fit_pop) 

            # Saves result
            print('\n RUN '+str(r)+ ' GENERATION '+str(i)+'  '+str(round(pop_gain[best],6))+'  '+str(round(fit_pop[best],6))+'  '+str(round(mean,6))+'  '+str(round(std,6)))    
            experiment_data  = open(experiment_name+'/results.txt','a')
            experiment_data.write('\n RUN '+str(r)+ ' GENERATION '+str(i)+'  '+str(round(pop_gain[best],6))+'  '+str(round(fit_pop[best],6))+'  '+str(round(mean,6))+'  '+str(round(std,6)))
            experiment_data.close()

            result_matrix_max[r,i]=np.max(fit_pop)
            result_matrix_mean[r,i]=np.mean(fit_pop)

            indices_run.append(r)
            indices_gen.append(i)
            
            best_gain.append(pop_gain[best])
            best_fit.append(fit_pop[best])
            mean_fitness.append(mean)
            std_fitness.append(std)
            best_solutions.append(best_solution)

    d = {"Run": indices_run, "Gen": indices_gen, "gain": best_gain, "Best fit": best_fit, "Mean": mean_fitness, "STD": std_fitness, "BEST SOL": best_solutions}
    df = pd.DataFrame(data=d)
    print(df)
    #makes csv file
    df.to_csv(f'{experiment_name}/{experiment_name}.csv', index=False)
    