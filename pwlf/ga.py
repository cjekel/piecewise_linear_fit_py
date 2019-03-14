import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random


def genetic_algorithm(total_set, nvar, objective_function, ngen=50, mu=20,
                      lam=40, cxpb=0.7, mutpb=0.2, tournsize=3,
                      verbose=False):
    # Creates a new class name set_mod which is based on the standard python
    # set. This means set_mod is just like set, with the addion of a fitness
    # attribute.
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", set, fitness=creator.Fitness)
    set_mod = creator.Individual

    def random_samp(size):
        """Function to initlize individual in the population."""
        return set_mod(random.sample(total_set, size))

    def evaluation(individual):
        """Evaluate the objective function."""
        return objective_function(list(individual)),

    def cxSet(ind1, ind2):
        """Apply a crossover operation on two sets."""
        full_set = list(ind1 | ind2)
        ind1 = set_mod(random.sample(full_set, nvar))
        ind2 = set_mod(random.sample(full_set, nvar))
        return ind1, ind2

    def mutSet(individual):
        """Mutation that randomly removes and item and randomly adds an item.
        """
        temp_set = set_mod(random.sample(individual, nvar-1))
        set_to_choose = np.array(list(temp_set ^ total_set))
        new = random.choice(set_to_choose)
        temp_set.add(new)
        return temp_set,

    toolbox = base.Toolbox()

    # set up the population
    toolbox.register("individual", random_samp, nvar)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # set up the GA functions
    toolbox.register("evaluate", evaluation)
    toolbox.register("mate", cxSet)
    toolbox.register("mutate", mutSet)
    # toolbox.register("select", tools.selNSGA2)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)

    # initialize the population
    pop = toolbox.population(n=mu)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    # run the GA
    algorithms.eaMuPlusLambda(pop, toolbox, mu, lam, cxpb, mutpb, ngen, stats,
                              halloffame=hof, verbose=verbose)
    return pop, hof, stats
