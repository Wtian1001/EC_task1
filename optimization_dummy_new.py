###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
import time
from math import fabs,sqrt,exp
import random

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
from scipy.special import softmax
import os

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

ini = time.time()  # sets time marker
run_mode = 'train'
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'optimization_test_task'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                enemies=[2],
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)


# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

ratio = 0.33
dom_u = 1
dom_l = -1
npop = 100
gens = 200
mutation = 0.2
num_step_size = 1
learning_rate = 1/sqrt(n_vars)
init_step_size = 0.5
u_boundary_step_size = 1
l_boundary_step_size = 0

# start writing your own code from here
def survival_selection(pop, fit_pop, num_selected):
    min_fitness = np.min(fit_pop)
    # # shifting the fitness values to make sure it contains no negative
    # if min_fitness < 0:
    #     fit_pop = fit_pop - min_fitness
    normalized_fitness = fit_pop
    if min_fitness < 0:
        normalized_fitness = fit_pop - min_fitness
        

    
    probs = normalized_fitness / np.sum(normalized_fitness)
    
    best_idx = np.argmax(fit_pop)
    best_nn = pop[best_idx]
    best_fit = fit_pop[best_idx]

    num_selected = num_selected - 1
    
    if num_selected > (pop.shape[0] - 1):
        num_selected = (pop.shape[0] - 1)
    print(pop.shape[0])
    print(probs.shape[0])
    indices = np.random.choice(pop.shape[0], num_selected, p = probs, replace=False)
    print("test 3", probs.shape[0])
    selected_pop = pop[indices]
    selected_fit = fit_pop[indices]
    print("test 4", selected_fit.shape[0])
    # print(selected_pop, best_nn)
    # print(selected_fit, best_fit)
    print(selected_fit, best_fit)
    selected_pop = np.vstack([selected_pop, best_nn])
    selected_fit = np.append(selected_fit, best_fit)
    print("test 5", selected_fit.shape[0])
    
    return selected_pop, selected_fit

# ensures that the step_size stays within these boundaries
def boundaries_step_size(x):
    if x>u_boundary_step_size:
        return u_boundary_step_size
    elif x<l_boundary_step_size:
        return l_boundary_step_size
    else:
        return x

# limits
def limits(x):

    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x

# weighted average crossover function
def crossover(pop, fit_pop, parents):

    # initialize output array 
    total_offspring = np.zeros((0,n_vars + 1))
    
    for p in range(0,len(parents), 2):

        # select parents
        p1 = pop[parents[p]]
        p2 = pop[parents[p+1]]
        
        n_offspring = 1
        offspring =  np.zeros((n_offspring, n_vars + num_step_size))

        # TODO: add mutation
        for f in range(1):
            # get weights for weighted average by using softmax
            w1, w2 = softmax([fit_pop[parents[p]], fit_pop[parents[p+1]]])
            # cross_prob = np.random.uniform(0,1)
            offspring[f] = w1 * p1 + p2 * w2
            
            #TODO: here 
            # my idea for incorporating the guided fitness mutation alongside the adaptive stepsize 
            # is to compare the fitness of the current offspring to the rest of the population. If it is below
            # a certain treshold of the population add an additional value to the step-size or it is probably better/neater
            # to change the mean of new_step_size = step_size * exp(learning_rate * np.random.normal(0, 1)) normal distribution a bit
            # maybe 50/50 pos/neg

            #probably need to initialize the stepsize randomly


            # obtain the step_size from the offspring 
            step_size = offspring[f][n_vars]

            # mutate the step_size
            new_step_size = step_size * exp(learning_rate * np.random.normal(0, 1))

            # ensures step size stays within boundaries
            new_step_size = boundaries_step_size(new_step_size)

            # set the new stepsize for the offspring
            offspring[f][n_vars] = new_step_size

            # mutation using updated step size
            for i in range(0,len(offspring[f] - num_step_size)):
                if np.random.uniform(0 ,1)<=mutation:
                    offspring[f][i] =   offspring[f][i]+np.random.normal(0, new_step_size)

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring

def truncation_random_hybrid_selection(pop,fit, nparents, ratio):
# 
# Returns a list of parents to be used in the crossover. The indices to the parents are provided.
# Uses a mix of Truncation Selection and random selection to generate a list of parents.
# Through truncation selection a certain number of best individuals are selected as parents.
# The parent list is then filled up by randomly sampling from the remaining individuals.
# pop: list that contains the population which will be sampled
# fit: list that contains the corresponding fitness values of the population
# nparents: the amount of parents to be generated
# ratio: the ratio between the parents chosen through truncation and random selection.
# For example a ratio of 3 will yield 1/3 selected through truncation and 2/3 selected randomly.
# 
    popsize = len(pop)
    nparents = nparents + 1 if nparents % 2 == 1 else nparents
    indices = list(range(popsize))
    ratio = int(1//ratio)
    slicesize = nparents//ratio
    combined = list(zip(fit,indices))

    sorted_combined = sorted(combined, key = lambda x:x[0], reverse=True) #highest first
    sorted_population = [ind for _, ind in sorted_combined]
    best_individuals = sorted_population[:slicesize]
    other_ind = sorted_population[slicesize:]
    random_ind = list(np.random.choice(other_ind, nparents - slicesize, False))

    return best_individuals + random_ind

# # tournament
# def truncation_random_hybrid_selection(pop, fit_pop, ratio):
#     popsize = pop.shape[0]
#     indices = list(range(popsize))
#     ratio = int(1//ratio)
#     slicesize = popsize//ratio
#     combined = list(zip(fit_pop,pop))
#     #print(pop[0], fit_pop[0])

#     sorted_combined = sorted(combined, key = lambda x:x[0], reverse=True) #highest first
#     sorted_population = [ind for _, ind in sorted_combined]
#     #print(sorted_population[0])
#     best_individuals = sorted_population[:slicesize]
#     other_individuals = sorted_population[slicesize:]
#     #print(best_individuals[0])
#     if np.random.random() < 0.33:
#         c1 =  np.random.randint(0,slicesize)
#         #print(c1)
#         return best_individuals[c1][0]
#     else:
#         c1 = np.random.randint(0, popsize - slicesize)
#         return other_individuals[c1][0]


# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars)) # creating npop size nn's with weights in between -1 and 1
    pop = np.hstack([pop, np.full((pop.shape[0], 1), init_step_size)])
    fit_pop = evaluate(env, pop) #returns an array that stores the fitness of each nn
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()




# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()


# evolution

last_sol = fit_pop[best]
notimproved = 0

for i in range(ini_g+1, gens):
    parents = truncation_random_hybrid_selection(pop, fit_pop, pop.shape[0] // 2, ratio)
    random.shuffle(parents)
    offspring = crossover(pop, fit_pop, parents)  # crossover
    fit_offspring = evaluate(env, offspring)   # evaluation
    pop = np.vstack((pop,offspring))
    fit_pop = np.append(fit_pop,fit_offspring)
    print("test 1 ", pop.shape[0])

    best = np.argmax(fit_pop) #best solution in generation
    fit_pop[best] = float(evaluate(env, np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    pop, fit_pop = survival_selection(pop, fit_pop, npop)
    print("test 2 ", pop.shape[0])



    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)


    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',pop[best])

    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()




fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state

    