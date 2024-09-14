import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

class Specialist():
    def __init__(self, n_hidden_neurons=10,
                 experiment_name = 'optimization_test_arco',
                 upperbound = 1,
                 lowerbound = -1,
                 population_size = 100,
                 ) -> None:
        self.headless = True
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.experiment_name = experiment_name
        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        self.n_hidden_neurons = n_hidden_neurons

        self.env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
        self.n_vars = (self.env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.population_size = population_size

    def simulation(self, neuron_values):
        f, p, e, t = self.env.play(pcont=neuron_values)
        return f
    
    def fitness_eval(self, population):
        return np.array([simulation(self.env, individual) for individual in population])
    
    def initialize(self):
        return np.random.uniform(self.lowerbound,
                                self.upperbound,
                                (self.population_size, self.n_vars))
    
    def mutation(self, child, p_mutation):
        
        if np.random.uniform() > p_mutation:
            #no mutation
            return child
        else:
            child = np.array(child)
            swap1, swap2 = np.random.choice(np.arange(1,10), size=2)
            child[[swap1, swap2]] = child[[swap2, swap1]]
            child_mutated = list(child)
        return child_mutated
    
    
    def limits(self, x):

        if x>self.upperbound:
            return self.upperbound
        elif x<self.lowerbound:
            return self.lowerbound
        else:
            return x
        
    def tournament(self, pop):
        c1 =  np.random.randint(0,pop.shape[0], 1)
        # c2 =  np.random.randint(0,pop.shape[0], 1)
        return c1

        # if fit_pop[c1] > fit_pop[c2]:
        #     return pop[c1][0]
        # else:
        #     return pop[c2][0]
        
    def crossover(self, pop, p_mutation):
        total_offspring = np.zeros((0,self.n_vars))


        for p in range(0,pop.shape[0], 2):
            p1 = self.tournament(pop)
            p2 = self.tournament(pop)

            n_offspring =   np.random.randint(1,3+1, 1)[0]
            offspring =  np.zeros( (n_offspring, self.n_vars) )

            for f in range(0,n_offspring):

                cross_prop = np.random.uniform(0,1)
                offspring[f] = p1*cross_prop+p2*(1-cross_prop)

                # mutation
                for i in range(0,len(offspring[f])):
                    if np.random.uniform(0 ,1)<=p_mutation:
                        offspring[f][i] =   offspring[f][i]+np.random.normal(0, 1)

                offspring[f] = np.array(list(map(lambda y: self.limits(y), offspring[f])))

                total_offspring = np.vstack((total_offspring, offspring[f]))

        return total_offspring

        
    
    def train(self, total_generations=100):
        ### run_mode = 'train'
        # if no earlier training is done:
        if not os.path.exists(self.experiment_name+'/evoman_solstate'):
            population = self.initialize()
            generation_number = 0
        else:
            print("Found earlier state")
            self.env.load_state()
            population = self.env.solutions[0]
        
        # Log mean, best, std
        fitness_population = self.fitness_eval(population)
        mean = np.mean(fitness_population)
        best = np.argmax(fitness_population)
        std = np.std(fitness_population)
        
        with open(self.experiment_name + '/gen.txt', 'r') as file_aux:
            generation_number = int(file_aux.readline())
        
        for gen_idx in range(generation_number + 1, total_generations):
            offspring = self.crossover(population, p_mutation=0.2)# PLACEHOLDER
            mutated_offspring = [self.mutation(springie) for springie in offspring]
            
            new_population = np.vstack((population, mutated_offspring))
            
            fitness_population = self.fitness_eval(population)
            mean = np.mean(fitness_population)
            best = np.argmax(fitness_population)
            std = np.std(fitness_population)
            Continue 233
            

if __name__ == '__main__':
    