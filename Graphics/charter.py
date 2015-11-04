
"""
##########################################################
### @Author Joe Krall      ###############################
### @copyright see below   ###############################

    This file is part of JMOO,
    Copyright Joe Krall, 2014.

    JMOO is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    JMOO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with JMOO.  If not, see <http://www.gnu.org/licenses/>.
    
###                        ###############################
##########################################################
"""

"Brief notes"
"Objective Space Plotter"

# from pylab import *
from time import *

from pylab import *

from jmoo_properties import *
from utility import *


def read_initial_population(prob, filename):
    fd_initial_data = open(filename, 'rb')
    reader_initial_data = csv.reader(fd_initial_data, delimiter=',')
    initial = []
    row_count = sum(1 for _ in csv.reader(open(filename)))
    for i,row in enumerate(reader_initial_data):
        if i > 1 and i != row_count-1:
                row = map(float, row)
                try: initial.append(prob.evaluate(row)[-1])
                except: pass
    return initial


def charter_reporter(problems, algorithms, Configurations, tag=""):
    date_folder_prefix = strftime("%m-%d-%Y")
    MU = Configurations["Universal"]["Population_Size"]

    generation_data = []
    
    for p,prob in enumerate(problems):
        generation_data.append([])
        for a, alg in enumerate(algorithms):
            fd_statistic_file = open("Data/results_" + prob.name + "-p" + str(MU) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "_" + alg.name + ".datatable", 'rb')
            reader_statistic_file = csv.reader(fd_statistic_file, delimiter=',')

            generation_data[p].append([])
            for i,row in enumerate(reader_statistic_file):
                # if not str(row[0]) == "0":
                    for j, col in enumerate(row):
                        if i == 0: generation_data[p][a].append([])
                        else:
                            if not col == "": generation_data[p][a][j].append(float(col.strip("%)(")))

        # For the ax1.text
        left, width = .55, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        from matplotlib.font_manager import FontProperties
        from matplotlib import rc
        font = {'family': 'sans-serif', 'weight': 'normal', 'size': 8}
        rc('font', **font)
        fontP = FontProperties()
        fontP.set_size('x-small')

        subplots_adjust(hspace=0)
        number_of_subplots = len(prob.objectives)

        for v in xrange(number_of_subplots):
            ax1 = subplot(number_of_subplots, 1, v+1)
            ax1.set_xscale("log")
            # to make sure the xaxis.xtick is only in the last objective
            if v != (number_of_subplots - 1):
                ax1.get_xaxis().set_visible(False)
            all_objective_values = []
            all_generation_numbers = []
            for a, alg in enumerate(algorithms):
                generation_numbers = generation_data[-1][a][0]
                objective_values = generation_data[-1][a][3 * v + 2]

                # for scale values
                all_objective_values += objective_values
                all_generation_numbers += generation_numbers

                # since generation_number cannot be 0 we change 0 to 1
                generation_data[-1][a][0][0] = 1

                # To generate the minimum line for each algorithm
                min_point = []
                temp_min = 1e32
                for gn, ov in zip(generation_numbers, objective_values):
                    if ov <= temp_min:
                        min_point.append([gn, ov])
                        temp_min = ov

                print alg.name
                print generation_numbers
                print objective_values

                ax1.scatter(generation_numbers,objective_values, marker=alg.type, color=alg.color)
                ax1.plot([item[0] for item in min_point], [item[-1] for item in min_point], marker=alg.type, color=alg.color)

            ax1.set_ylim(min(all_objective_values) * 0.99, max(all_objective_values) * 1.1)
            ax1.set_xlim(min(all_generation_numbers), max(all_generation_numbers) * 1.1)
            ax1.set_ylabel("% change\nmedian values")
            ax1.text(right, 0.5*(bottom+top), prob.objectives[v].name,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=270,
                    fontsize=11,
                    transform=ax1.transAxes)


        plt.xlabel("Number of evaluations in log scale")

        # House keeping
        # if the directory for today is not created then create it
        if not os.path.isdir('Charts/' + date_folder_prefix): os.makedirs('Charts/' + date_folder_prefix)

        # to assign a figure number so that there is no conflict
        figure_number = len([name for name in os.listdir('Charts/' + date_folder_prefix)]) + 1
        plt.tight_layout()

        # to save the figure as .png file
        plt.savefig('Charts/' + date_folder_prefix + '/figure' + str("%02d" % figure_number) + "_" + prob.name + "_" + tag + '.png', dpi=100)
        cla()
        clf()
        close()
