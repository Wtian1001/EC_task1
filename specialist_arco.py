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
    
    def train(self):
        run_mode = 'train'
        
        # if no earlier training is done:
        if not os.path.exists(self.experiment_name+'/evoman_solstate'):
            print( '\nInitializing training\n')
            population = np.random.uniform(self.lowerbound,
                                           self.upperbound,
                                           (self.population_size, self.n_vars))
            fitness_population = fitness_eval(population)
        continue line 186

if __name__ == '__main__':
    