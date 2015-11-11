from __future__ import division
import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0],"../../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from jmoo_properties import *
from jmoo_core import *

problems =[
    dtlz1(9, 5),
    dtlz2(14, 5),
    dtlz3(14, 5),
    dtlz4(14, 5),
    dtlz5(14, 5),
    dtlz6(14, 5),
    dtlz1(7, 3),
    dtlz2(12, 3),
    dtlz3(12, 3),
    dtlz4(12, 3),
    dtlz5(12, 3),
    dtlz6(12, 3),
    dtlz1(12, 8),
    dtlz2(17, 8),
    dtlz3(17, 8),
    dtlz4(17, 8),
    dtlz5(17, 8),
    dtlz6(17, 8),
    dtlz1(14, 10),
    dtlz2(19, 10),
    dtlz3(19, 10),
    dtlz4(19, 10),
    dtlz5(19, 10),
    dtlz6(19, 10),
    dtlz1(19, 15),
    dtlz2(24, 15),
    dtlz3(24, 15),
    dtlz4(24, 15),
    dtlz5(24, 15),
    dtlz6(24, 15),
]

Configurations = {
    "Universal": {
        "Repeats" : 5,
        "Population_Size" : 100,
        "No_of_Generations" : 20
    },
    "NSGAIII": {
        "SBX_Probability": 1,
        "ETA_C_DEFAULT_" : 30,
        "ETA_M_DEFAULT_" : 20
    },
    "GALE": {
        "GAMMA" : 0.15,  #Constrained Mutation Parameter
        "EPSILON" : 1.00,  #Continuous Domination Parameter
        "LAMBDA" :  3,     #Number of lives for bstop
        "DELTA"  : 1       # Accelerator that increases mutation size
    },
    "DE": {
        "F" : 0.75, # extrapolate amount
        "CF" : 0.3, # prob of cross over
    },
    "MOEAD" : {
        "niche" : 20,  # Neighbourhood size
        "SBX_Probability": 1,
        "ETA_C_DEFAULT_" : 20,
        "ETA_M_DEFAULT_" : 20,
        "Theta" : 5
    },
    "STORM": {
        "STORM_EXPLOSION" : 5,
        "STORM_POLES" : 20,  # number of actual poles is 2 * ANYWHERE_POLES
        "F" : 0.75, # extrapolate amount
        "CF" : 0.3, # prob of cross over
        "STORM_SPLIT": 6,  # Break and split into pieces
        "GAMMA" : 0.15,
    }
}


def fastmap(problem, true_population):
    """
    Fastmap function that projects all the points on the principal component
    :param problem: Instance of the problem
    :param population: Set of points in the cluster population
    :return:
    """

    def list_equality(lista, listb):
        for a, b in zip(lista, listb):
            if a != b: return False
        return True

    from random import choice
    from euclidean_distance import euclidean_distance

    decision_population = [pop.decisionValues for pop in true_population]
    one = choice(decision_population)
    west = furthest(one, decision_population)
    east = furthest(west, decision_population)
    c = euclidean_distance(east, west)
    tpopulation = []
    for one in decision_population:
        a = euclidean_distance(one, west)
        b = euclidean_distance(one, east)
        tpopulation.append([one, projection(a, b, c)])

    for tpop in tpopulation:
        for true_pop in true_population:
            if list_equality(tpop[0], true_pop.decisionValues):
                true_pop.x = tpop[-1]
    temp_list = sorted(true_population, key=lambda pop: pop.x)
    return true_population, temp_list[0], temp_list[-1]


def furthest(individual, population):
    from euclidean_distance import euclidean_distance
    distances = sorted([[euclidean_distance(individual, pop), pop] for pop in population], key=lambda x: x[0], reverse=True)
    return distances[0][-1]


def projection(a, b, c):
    """
    Fastmap projection distance
    :param a: Distance from West
    :param b: Distance from East
    :param c: Distance between West and East
    :return: FastMap projection distance(float)
    """
    return (a**2 + c**2 - b**2) / (2*c+0.00001)


def loss_function(mean_scores_before, mean_scores_after):
    weights = [1 if obj.lismore else -1 for obj in problem.objectives]
    weightedWest = [c * w for c, w in zip(mean_scores_before, weights)]
    weightedEast = [c * w for c, w in zip(mean_scores_after, weights)]
    westLoss = loss(weightedWest, weightedEast, mins=[obj.low for obj in problem.objectives],
                    maxs=[obj.up for obj in problem.objectives])
    eastLoss = loss(weightedEast, weightedWest, mins=[obj.low for obj in problem.objectives],
                    maxs=[obj.up for obj in problem.objectives])

    return westLoss, eastLoss

repeat = 10
print Configurations["Universal"], Configurations["GALE"]
for problem in problems:
    print problem.name, " | ",
    pass_count = 0
    count = 0
    success_count = 0
    for _ in xrange(repeat):
        initialPopulation(problem, Configurations["Universal"]["Population_Size"], "unittesting")
        population = problem.loadInitialPopulation(Configurations["Universal"]["Population_Size"], "unittesting")

        for individual in population:
            if not individual.valid: individual.evaluate()

        # find median of the scores
        mean_scores_before = [mean([pop.fitness.fitness[i] for pop in population]) for i in xrange(len(problem.objectives))]
        # print "Before: ", mean_scores_before

        population, east, west = fastmap(problem, population)


        weights = [1 if obj.lismore else -1 for obj in problem.objectives]
        weightedWest = [c * w for c, w in zip(west.fitness.fitness, weights)]
        weightedEast = [c * w for c, w in zip(east.fitness.fitness, weights)]
        westLoss = loss(weightedWest, weightedEast, mins=[obj.low for obj in problem.objectives],
                        maxs=[obj.up for obj in problem.objectives])
        eastLoss = loss(weightedEast, weightedWest, mins=[obj.low for obj in problem.objectives],
                        maxs=[obj.up for obj in problem.objectives])

        # Determine better Pole
        if eastLoss < westLoss:   SouthPole, NorthPole = east, west
        else:  SouthPole, NorthPole = west, east

        # Magnitude of the mutations
        g = abs(SouthPole.x - NorthPole.x)


        mutants = []

        # Iterate over the individuals of the leaf
        for i, row in enumerate(population):

            # Make a copy of the row in case we reject it
            from copy import deepcopy
            copy = deepcopy(row)
            cx = row.x

            for attr in range(0, len(problem.decisions)):

                # just some naming shortcuts
                me = row.decisionValues[attr]
                good = SouthPole.decisionValues[attr]
                bad = NorthPole.decisionValues[attr]
                dec = problem.decisions[attr]

                # Find direction to mutate (Want to mutate towards good pole)
                if me > good:  d = -1
                if me < good:  d = +1
                if me == good: d = 0

                row.decisionValues[attr] = min(dec.up, max(dec.low, (me + me * g * d) * Configurations["GALE"]["DELTA"]))

            # Project the Mutant
            a = euclidean_distance(NorthPole.decisionValues, row.decisionValues)
            b = euclidean_distance(SouthPole.decisionValues, row.decisionValues)
            c = euclidean_distance(SouthPole.decisionValues, NorthPole.decisionValues)
            x = (a ** 2 + c ** 2 - b ** 2) / (2 * c + 0.00001)

            # print abs(cx-x), (cx + (g * configuration["GALE"]["GAMMA"]))
            if abs(x - cx) > (g * Configurations["GALE"]["GAMMA"]) or problem.evalConstraints(row.decisionValues):  # reject it
                # print "reject it", i
                row = copy
                row.x = x
            else:
                # print "Changed Number: ", count, g * Configurations["GALE"]["GAMMA"]
                count = count + 1
                # print "Before: ", copy.decisionValues
                # print "After : ", row.decisionValues
                # print "Difference: ", euclidean_distance(copy.decisionValues, row.decisionValues)
                row.evaluate()
                score, _ = loss_function(copy.fitness.fitness, row.fitness.fitness)
                if score < 1: success_count += 1
                mutants.append(row)
                # print

                pass

        # mean_scores_after = [mean([pop.fitness.fitness[i] for pop in population]) for i in xrange(len(problem.objectives))]
        # # print "After : ", mean_scores_after
        #
        # score, _ = loss_function(mean_scores_before, mean_scores_after)
        #
        # mean_scores_mutants = [mean([pop.fitness.fitness[i] for pop in mutants]) for i in xrange(len(problem.objectives))]
        # print "Mutants : ", mean_scores_mutants

        # print loss_function(mean_scores_before, mean_scores_mutants)

        # print "Number of changes: ", count, " out of : ", Configurations["Universal"]["Population_Size"], " | ", score
        # print " score: ", score, " "
        # if score < 1: pass_count += 1
    # print pass_count, repeat
    print round((success_count/count) * 100, 3), "%"
