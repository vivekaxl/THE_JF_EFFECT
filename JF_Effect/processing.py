class DataContainer():
    def __init__(self, contents):
        self.problem_name = None
        self.algorithm_name = None
        self.repeat_number = None
        self.contents = contents
        self.generations = None
        self.get_info()
        self.generation_split()

    def get_info(self):
        line = self.contents[0]
        names = line.split("|")[1].replace(" ", "")
        self.problem_name, self.algorithm_name, self.repeat_number = names.split("+")
        self.contents = self.contents[1:]

    def generation_split(self):
        indexes = [i for i, line in enumerate(self.contents) if "|" in line] + [len(self.contents)]
        self.generations = [self.contents[indexes[i] + 1:indexes[i + 1]] for i in xrange(len(indexes) - 1)]
        self.remove_fluf()

    def remove_fluf(self):
        temp = [[line.replace("Generation Number:  ", "").replace("\n", "")[1:-1] for line in slice] for slice in self.generations]
        self.generations = [max([len(t.split(",")) for t in t_temp if t != "##################"]) for t_temp in temp]


def median_list(data):
    """
    Find the median of list of lists
    :param data: A DataContainer
    :return: list
    """
    return_list = []
    max_generation = max(len(l.generations) for l in data)
    from numpy import median
    for generation in xrange(max_generation):
        # print "> ", [l.generations[generation] for l in data if len(l.generations) - 1 >= generation]
        return_list.append(median([l.generations[generation] for l in data if len(l.generations) - 1 >= generation]))
    return return_list


def algorithm_problem_combo(contents):
    # find the indexes which has the content : ' # Finished: Celebrate! # \n'
    problem_name, algorithm_name, _ = contents[0].split("|")[1].replace(" ", "").split("+")
    indexes = [0] + [count+1 for count, content in enumerate(contents) if ' # Finished: Celebrate! #' in content]
    slices = [contents[indexes[i]:indexes[i+1]] for i in xrange(len(indexes) - 1)]
    data_slices = [DataContainer(sliced) for sliced in slices]
    return problem_name, algorithm_name, median_list(data_slices)


def splitting_into_problems(contents):
    indexes = [count for count, content in enumerate(contents) if '# ' * 30 in content] + [len(contents)]
    problem_slices = [contents[indexes[i]+1:indexes[i+1]] for i in xrange(len(indexes) - 1)]
    return problem_slices


def splitting_into_algorithms(contents):
    indexes = [count for count, content in enumerate(contents) if '+ ' * 30 in content] + [len(contents)]
    algorithms_slices = [contents[indexes[i]+1:indexes[i+1]] for i in xrange(len(indexes) - 1)]
    return algorithms_slices


def drawing(problem_name, gale, gale2, x):
    import numpy as np
    import matplotlib.pyplot as plt
    n= 20
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    ax1.bar(x, gale, 0.5) # thickness=0.5
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Survival Rate (Median)')

    ax2.bar(x, gale2, 0.5)
    ax2.set_xlabel('Generations')
    ax2.set_ylabel('Survival Rate (Median)')

    plt.ylim(0, max([max(gale), max(gale2)]) + 2)
    plt.xlim(0, 21)
    f.suptitle(problem_name, fontsize="x-large")
    plt.tight_layout()
    plt.savefig(problem_name + ".png", dpi=100)




if __name__ == "__main__":
    contents = open("jfresult.txt").readlines()
    problem_slices = splitting_into_problems(contents)
    result = {}
    for problem_slice in problem_slices:
        algorithm_slices = splitting_into_algorithms(problem_slice)
        for algorithm_slice in algorithm_slices:
            problem_name, algorithm_name, median_values =  algorithm_problem_combo(algorithm_slice)
            if problem_name not in result.keys():
                result[problem_name] = {}
            result[problem_name][algorithm_name] = median_values

    for problem in result.keys():
        drawing(problem, result[problem]["GALE"], result[problem]["GALE2"], range(1, 21))