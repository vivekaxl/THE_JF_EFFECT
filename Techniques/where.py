from __future__ import division


class Node:
    def __init__(self, level, members):
        self.members = members
        self.level = level


def where_wrapper(problem, population):
    """Where the population and return the leaves"""

    def divide(problem, population, lvl,  where_configuration):


        def check_depth(): return lvl > where_configuration["depthMax"]  # depthMax is not updated

        def check_elements() : return len(population) < where_configuration["minSize"]

        if check_depth() or check_elements(): return Node(lvl, population)

        print "Length of population: ", len(population), " Level: ", lvl
        population, east, west = fastmap(problem, population)
        mid = int(len(population) / 2)
        wests = population[:mid]
        easts = population[mid+1:]

        # print divide(problem, wests, lvl + 1, where_configuration), len(divide(problem, wests, lvl + 1, where_configuration))
        raw_input()
        r_population = []
        r_population += divide(problem, wests, lvl + 1, where_configuration)
        r_population += divide(problem, easts, lvl + 1, where_configuration)

        return r_population



    decisions = [pop.decisionValues for pop in population]
    where_configuration = {
        "minSize": 10,    # min leaf size
        "depthMin": 2,      # no pruning till this depth
        "depthMax": 10,     # max tree depth
        "wriggle": 0.2,    # min difference of 'better'
        "prune": True,   # pruning enabled?
        "b4": '|.. ', # indent string
        "verbose": False,  # show trace info?
    }
    leaves = divide(problem, decisions, 0, where_configuration)
    clusters = []
    print "Length of leaves: ", len(leaves)
    import pdb
    pdb.set_trace()
    for leaf in leaves[-1]:
        cluster = []
        print "Length of the leaf is: ", len(leaf), leaf
        for member in leaf:
            for pop in population:
                assert(len(member) == len(pop.decisionValues)), "Something is wrong"
                # compare list
                result = reduce(lambda x, y: x and y, [True if i == j else False for i, j in zip(member, pop.decisionValues)])
                if result is True:
                    cluster.append(pop)
                    break
        clusters.append(cluster)

    print clusters

    return clusters


# def get_leaves(listoflist):
#     if isinstance(listoflist[0], float):
#         return listoflist
#     else:


def maybePrune(problem, lvl, west, east, Configuration):
    """Usually, go left then right, unless dominated."""
    go_left, go_right = True, True  # default
    if Configuration["prune"] and lvl >= Configuration["depthMin"]:
        sw = problem.evaluate(west)
        se = problem.evaluate(east)
        if abs(sw - se) > Configuration["wriggle"]:  # big enough to consider
            if se > sw: go_left = False  # no left
            if sw > se: go_right = False  # no right

    return go_left, go_right


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


def fastmap(problem, population):
    """
    Fastmap function that projects all the points on the principal component
    :param problem: Instance of the problem
    :param population: Set of points in the cluster population
    :return:
    """
    from random import choice
    from euclidean_distance import euclidean_distance
    one = choice(population)
    west = furthest(one, population)
    east = furthest(west, population)
    c = euclidean_distance(east, west)
    tpopulation = []
    for one in population:
        a = euclidean_distance(one, west)
        b = euclidean_distance(one, east)
        tpopulation.append([one, projection(a, b, c)])
    population = [pop[0] for pop in sorted(tpopulation, key=lambda l: l[-1])]
    return population, east, west
