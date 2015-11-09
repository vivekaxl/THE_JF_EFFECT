import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0],"../../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from jmoo_properties import *
from jmoo_core import *

problems =[
    # dtlz1(9, 5),
    # dtlz2(14, 5),
    # dtlz3(14, 5),
    # dtlz4(14, 5),
    # dtlz1(7, 3),
    # dtlz2(12, 3),
    # dtlz3(12, 3),
    # dtlz4(12, 3),
    # dtlz1(12, 8),
    # dtlz2(17, 8),
    # dtlz3(17, 8),
    # dtlz4(17, 8),
    # dtlz1(14, 10), dtlz2(19, 10),
    # dtlz3(19, 10),
    # dtlz4(19, 10),
    # dtlz1(19, 15),
    # dtlz2(24, 15),
    # dtlz3(24, 15),
    # dtlz4(24, 15)
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
        "DELTA"  : 3       # Accelerator that increases mutation size
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

for problem in problems:
    initialPopulation(problem, Configurations["Universal"]["Population_Size"])
    population = problem.loadInitialPopulation(Configurations["Universal"]["Population_Size"])

    # Pull out the Poles
        east = leaf.table.rows[0]
        west = leaf.table.rows[-1]

        # Evaluate those poles if needed
        if not east.evaluated:
            for o, objScore in enumerate(problem.evaluate(east.cells)):
                east.cells[-(len(problem.objectives) - o)] = objScore
            east.evaluated = True
            numEval += 1
        if not west.evaluated:
            for o, objScore in enumerate(problem.evaluate(west.cells)):
                west.cells[-(len(problem.objectives) - o)] = objScore
            west.evaluated = True
            numEval += 1

        # Score the poles
        n = len(problem.decisions)
        weights = []
        for obj in problem.objectives:
            # w is negative when we are maximizing that objective
            if obj.lismore:
                weights.append(+1)
            else:
                weights.append(-1)
        weightedWest = [c * w for c, w in zip(west.cells[n:], weights)]
        weightedEast = [c * w for c, w in zip(east.cells[n:], weights)]
        westLoss = loss(weightedWest, weightedEast, mins=[obj.low for obj in problem.objectives],
                        maxs=[obj.up for obj in problem.objectives])
        eastLoss = loss(weightedEast, weightedWest, mins=[obj.low for obj in problem.objectives],
                        maxs=[obj.up for obj in problem.objectives])

        # Determine better Pole
        if eastLoss < westLoss:
            SouthPole, NorthPole = east, west
        else:
            SouthPole, NorthPole = west, east

        # Magnitude of the mutations
        g = abs(SouthPole.x - NorthPole.x)

        # Iterate over the individuals of the leaf
        for row in leaf.table.rows:

            # Make a copy of the row in case we reject it
            copy = [item for item in row.cells]
            gen_num_xyz = get_previous_generation_number(actual_population, copy)
            print "Generation Number: ", gen_num_xyz
            temp_generation_number = get_previous_generation_number(actual_population, copy)
            cx = row.x

            for attr in range(0, len(problem.decisions)):

                # just some naming shortcuts
                me = row.cells[attr]
                good = SouthPole.cells[attr]
                bad = NorthPole.cells[attr]
                dec = problem.decisions[attr]

                # Find direction to mutate (Want to mutate towards good pole)
                if me > good:  d = -1
                if me < good:  d = +1
                if me == good: d = 0

                row.cells[attr] = min(dec.up, max(dec.low, (me + me * g * d) * configuration["GALE"]["DELTA"]))

            # Project the Mutant
            a = row.distance(NorthPole)
            b = row.distance(SouthPole)
            c = NorthPole.distance(SouthPole)
            x = (a ** 2 + row.c ** 2 - b ** 2) / (2 * row.c + 0.00001)

            # Test Mutant for Acceptance
            # confGAMMA = 0.15 #note: make this a property

            # print abs(cx-x), (cx + (g * configuration["GALE"]["GAMMA"]))
            if abs(x - cx) > (g * configuration["GALE"]["GAMMA"]) or problem.evalConstraints(row.cells[:n]):  # reject it
                row.cells = copy
                row.x = x
            row.generation = temp_generation_number + [gen]