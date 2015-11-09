
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
from Algorithms.DEAP.tools.support import ParetoFront


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


# def charter_reporter(problems, algorithms, Configurations, tag=""):
#     date_folder_prefix = strftime("%m-%d-%Y")
#     MU = Configurations["Universal"]["Population_Size"]
#
#     generation_data = []
#
#     for p,prob in enumerate(problems):
#         generation_data.append([])
#         for a, alg in enumerate(algorithms):
#             fd_statistic_file = open("Data/results_" + prob.name + "-p" + str(MU) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "_" + alg.name + ".datatable", 'rb')
#             reader_statistic_file = csv.reader(fd_statistic_file, delimiter=',')
#
#             generation_data[p].append([])
#             for i,row in enumerate(reader_statistic_file):
#                 # if not str(row[0]) == "0":
#                     for j, col in enumerate(row):
#                         if i == 0: generation_data[p][a].append([])
#                         else:
#                             if not col == "": generation_data[p][a][j].append(float(col.strip("%)(")))
#
#         # For the ax1.text
#         left, width = .55, .5
#         bottom, height = .25, .5
#         right = left + width
#         top = bottom + height
#         from matplotlib.font_manager import FontProperties
#         from matplotlib import rc
#         font = {'family': 'sans-serif', 'weight': 'normal', 'size': 8}
#         rc('font', **font)
#         fontP = FontProperties()
#         fontP.set_size('x-small')
#
#         subplots_adjust(hspace=0)
#         number_of_subplots = len(prob.objectives)
#
#         for v in xrange(number_of_subplots):
#             ax1 = subplot(number_of_subplots, 1, v+1)
#             ax1.set_xscale("log")
#             # to make sure the xaxis.xtick is only in the last objective
#             if v != (number_of_subplots - 1):
#                 ax1.get_xaxis().set_visible(False)
#             all_objective_values = []
#             all_generation_numbers = []
#             for a, alg in enumerate(algorithms):
#                 generation_numbers = generation_data[-1][a][0]
#                 objective_values = generation_data[-1][a][3 * v + 2]
#
#                 # for scale values
#                 all_objective_values += objective_values
#                 all_generation_numbers += generation_numbers
#
#                 # since generation_number cannot be 0 we change 0 to 1
#                 generation_data[-1][a][0][0] = 1
#
#                 # To generate the minimum line for each algorithm
#                 min_point = []
#                 temp_min = 1e32
#
#                 scores = {}
#
#                 for gn, ov in zip(generation_numbers, objective_values):
#                     eval = int(round(gn/5.0)*5.0)
#                     if eval in scores: scores[eval].append(ov)
#                     else: scores[eval] = [ov]
#
#
#                 key_list = [1]
#                 score_list = [100]
#                 smallslist = [100]
#                 for eval in sorted(scores.keys()):
#                     lq = getPercentile(scores[eval], 25)
#                     uq = getPercentile(scores[eval], 75)
#                     scores[eval] = [score for score in scores[eval] if lq <= score <= uq]
#                     # import pdb
#                     # pdb.set_trace()
#                     for item in scores[eval]:
#                         key_list.append(eval)
#                         score_list.append(item)
#                         if len(smallslist) == 0:
#                             smallslist.append(min(scores[eval]))
#                         else:
#                             smallslist.append(    min(min(scores[eval]), min(smallslist))  )
#
#                 # print alg.name, v
#                 # print key_list
#                 # print score_list
#                 # print smallslist
#                 # print " = " * 30
#
#                 ax1.scatter(key_list, score_list, marker=alg.type, color=alg.color)
#                 ax1.plot(key_list, smallslist, marker=alg.type, color=alg.color)
#                 # ax1.plot([item[0] for item in min_point], [item[-1] for item in min_point], marker=alg.type, color=alg.color)
#
#             ax1.set_ylim(min(all_objective_values) * 0.9, max(all_objective_values) * 1.1)
#             ax1.set_xlim(min(all_generation_numbers), max(all_generation_numbers) * 1.1)
#             ax1.set_ylabel("% change\nbest values")
#             ax1.text(right, 0.5*(bottom+top), prob.objectives[v].name,
#                     horizontalalignment='center',
#                     verticalalignment='center',
#                     rotation=270,
#                     fontsize=11,
#                     transform=ax1.transAxes)
#
#
#         plt.xlabel("Number of evaluations in log scale")
#
#         # House keeping
#         # if the directory for today is not created then create it
#         if not os.path.isdir('Charts/' + date_folder_prefix): os.makedirs('Charts/' + date_folder_prefix)
#
#         # to assign a figure number so that there is no conflict
#         figure_number = len([name for name in os.listdir('Charts/' + date_folder_prefix)]) + 1
#         plt.tight_layout()
#
#         # to save the figure as .png file
#         plt.savefig('Charts/' + date_folder_prefix + '/figure' + str("%02d" % figure_number) + "_" + prob.name + "_" + tag + '.png', dpi=150)
#         cla()
#         clf()
#         close()

def charter_reporter(problems, algorithms, Configurations, tag=""):
    date_folder_prefix = strftime("%m-%d-%Y")

    fignum = 0


    base = []
    final = []
    RRS = []
    data = []
    foam = []

    for p,prob in enumerate(problems):
        base.append([])
        final.append([])
        RRS.append([])
        data.append([])
        foam.append([])

        for a,alg in enumerate(algorithms):
            finput = open("data/" + prob.name + "-p" + str(Configurations["Universal"]["Population_Size"]) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "-dataset.txt", 'rb')
            f2input = open(DATA_PREFIX + RRS_TABLE + "_" + prob.name + "-p" + str(Configurations["Universal"]["Population_Size"]) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "_" + alg.name + DATA_SUFFIX, 'rb')
            f3input = open("data/results_" + prob.name + "-p" + str(Configurations["Universal"]["Population_Size"]) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "_" + alg.name + ".datatable", 'rb')
            f4input = open(DATA_PREFIX + "decision_bin_table" + "_" + prob.name+ "-p" + str(Configurations["Universal"]["Population_Size"]) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives))  + "_" + alg.name + DATA_SUFFIX, 'rb')
            reader = csv.reader(finput, delimiter=',')
            reader2 = csv.reader(f2input, delimiter=',')
            reader3 = csv.reader(f3input, delimiter=',')
            reader4 = csv.reader(f4input, delimiter=',')
            base[p].append( [] )
            final[p].append( [] )
            RRS[p].append( [] )
            data[p].append( [] )
            foam[p].append( [] )

            for i,row in enumerate(reader):
                if i <= 100 and i > 0:
                    candidate = [float(col) for col in row]
                    fitness = prob.evaluate(candidate)
                    base[p][a].append(candidate+fitness)

            for i,row in enumerate(reader4):
                n = len(prob.decisions)
                candidate = [float(col) for col in row[:n]]
                fitness = prob.evaluate(candidate)
                final[p][a].append(candidate+fitness)

            for o,obj in enumerate(prob.objectives):
                RRS[p][a].append([])
                RRS[p][a][o] = {}
                foam[p][a].append([])
                foam[p][a][o] = {}


            for i,row in enumerate(reader2):
                k = len(prob.objectives)
                fitness = [float(col) for col in row[-k-1:-1]]
                for o,fit in enumerate(fitness):
                    n = int(row[k])
                    n = (int(round(n/5.0)*5.0))
                    if n in RRS[p][a][o]: RRS[p][a][o][n].append(fit)
                    else: RRS[p][a][o][n] = [fit]

            for i,row in enumerate(reader3):
                if not str(row[0]) == "0":
                    for j,col in enumerate(row):
                        if i == 0:
                            data[p][a].append([])
                        else:
                            if not col == "":
                                data[p][a][j].append(float(col.strip("%)(")))
                    # row is now read
                    if i > 0:
                        for o,obj in enumerate(prob.objectives):
                            n = data[p][a][0][-1]
                            n = (int(round(n/20.0)*20.0))
                            if n in foam[p][a][o]: foam[p][a][o][n].append(float(data[p][a][o*3+2][-1]))
                            else: foam[p][a][o][n] = [float(data[p][a][o*3+2][-1])]




    fignum = 0
    colors = ['r', 'b', 'g']
    from matplotlib.font_manager import FontProperties
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 8}

    matplotlib.rc('font', **font)
    fontP = FontProperties()
    fontP.set_size('x-small')


    codes = ["b*", "r.", "g*"]

    line =  "-"
    dotted= "--"
    algnames = [alg.name for alg in algorithms]
    axy = [0,1,2,3]
    axx = [0,0,0,0]
    codes2= ["b-", "r-", "g-"]
    colors= ["b", "r", "g"]
    ms = 8
    from mpl_toolkits.mplot3d import Axes3D
    #fig  = plt.figure()
    #ax = fig.gca(projection='3d')




    for p,prob in enumerate(problems):
                f, axarr = plt.subplots(len(prob.objectives))#+1, len(prob.objectives))

                for o, obj in enumerate(prob.objectives):
                    maxEvals = 0
                    for a,alg in enumerate(algorithms):
                        maxEvals = max(maxEvals, max(data[p][a][0]))
                    for a,alg in enumerate(algorithms):

                        scores = {}
                        for score,eval in zip(data[p][a][o*3+2], data[p][a][0]):
                            eval = int(round(eval/5.0)*5.0)
                            if eval in scores: scores[eval].append(score)
                            else: scores[eval] = [score]

                        keylist = [1]
                        scorelist = [100]
                        smallslist = [100]
                        for eval in sorted(scores.keys()):
                            lq = getPercentile(scores[eval], 25)
                            uq = getPercentile(scores[eval], 75)
                            scores[eval] = [score for score in scores[eval] if score >= lq and score <= uq ]
                            for item in scores[eval]:
                                keylist.append(eval)
                                scorelist.append(item)
                                if len(smallslist) == 0:
                                    smallslist.append(min(scores[eval]))
                                else:
                                    smallslist.append(    min(min(scores[eval]), min(smallslist))  )

                        axarr[o].plot(keylist, scorelist, linestyle='None', marker=alg.type, color=alg.color, markersize=8, markeredgecolor='none')
                        axarr[o].plot(keylist, smallslist, color=alg.color)
                        axarr[o].set_ylim(0, 130)
                        # axarr[o].set_autoscale_on(True)
                        axarr[o].set_xlim([-10, 10000])
                        axarr[o].set_xscale('log', nonposx='clip')


                if not os.path.isdir('charts/' + date_folder_prefix):
                    os.makedirs('charts/' + date_folder_prefix)

                f.suptitle(prob.name)
                fignum = len([name for name in os.listdir('charts/' + date_folder_prefix)]) + 1
                plt.savefig('charts/' + date_folder_prefix + '/figure' + str("%02d" % fignum) + "_" + prob.name + "_" + tag + '.png', dpi=100)
                cla()
    #show()

