# used from https://github.com/ai-se/SPL/blob/master/parser/parser.py

import os,sys,random
galedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/GALE/'
sys.path.insert(0,galedir)

import pdb,re
import xml.etree.ElementTree as ET
import numpy as np
from Feature_tree import *
# from model import *
from jmoo_objective import *
from jmoo_decision import *
from jmoo_problem import jmoo_problem


def load_ft_url(url):
    # load the feature tree and constraints
    tree = ET.parse(url)
    root = tree.getroot()

    for child in root:
        print child
        if child.tag == 'feature_tree':
            feature_tree_text = child.text
        if child.tag == 'constraints':
            constraints = child.text

    # initialize the feature tree
    feature_tree = FeatureTree()

    # parse the feature tree text
    features = feature_tree_text.split("\n")
    features = filter(bool, features)  # Fastest way to remove empty strings
    common_feature_pattern = re.compile('(\t*):([romg]?).*\W(\w+)\W.*')
    group_pattern = re.compile('\t*:g \W(\w+)\W \W(\d),([\d\*])\W.*')
    layer_dict = dict()
    for feature in features:
        m = common_feature_pattern.match(feature)
        """
        m.group(1) layer
        m.group(2) type
        m.group(3) id
        """
        layer = len(m.group(1))
        t = m.group(2)
        if t == 'r':
            tree_root = Node(id = m.group(3), node_type = 'r')
            layer_dict[layer] = tree_root
            feature_tree.set_root(tree_root)
        elif t== 'g':
            mg = group_pattern.match(feature)
            """
            mg.group(1) id
            mg.group(2) down_count
            mg.group(3) up_count
            """
            gNode = Node(id = mg.group(1), parent = layer_dict[layer-1], node_type = 'g')
            layer_dict[layer] = gNode
            if mg.group(3) == '*':
                gNode.g_u = np.inf
            else:
                gNode.g_u = mg.group(3)
            gNode.g_d = mg.group(2)
            layer_dict[layer] = gNode
            gNode.parent.add_child(gNode)
        else:
            treeNode = Node(id = m.group(3), parent = layer_dict[layer-1], node_type = t)
            layer_dict[layer] = treeNode
            treeNode.parent.add_child(treeNode)

    # parse the constraints
    cons = constraints.split('\n')
    cons = filter(bool, cons)
    common_con_pattern = re.compile('(\w+):(~?)(\w+)(.*)\s*')
    common_more_con_pattern = re.compile('\s+(or) (~?)(\w+)(.*)\s*')

    for cc in cons:
        literal = []
        li_pos = []
        m = common_con_pattern.match(cc)
        con_id = m.group(1)
        li_pos.append(not bool(m.group(2)))
        literal.append(m.group(3))
        while m.group(4):
            cc = m.group(4)
            m = common_more_con_pattern.match(cc)
            li_pos.append(not bool(m.group(2)))
            literal.append(m.group(3))
        """
         con_id: constraint identifier
         literal: literals
         li_pos: whether is positive or each literals
        """
        con_stmt = Constraint(id = con_id, literals = literal, literals_pos = li_pos)
        feature_tree.add_constraint(con_stmt)

    feature_tree.set_features_list()

    return feature_tree

def if_exists(file_name):
    folder_name = "./Problems/Feature_Models/References/"
    from os.path import isfile
    from os import getcwd
    print getcwd()
    print folder_name + file_name + ".xml"
    print isfile(folder_name + file_name + ".xml")
    return isfile(folder_name + file_name + ".xml")

# three objectives at this time
class FeatureTreeModel(jmoo_problem):
    def __init__(self, name,  objnum = 3):

        self.name = name
        assert(if_exists(name) is True), "Check the filename"
        self.url = "./Problems/Feature_Models/References/" + name + ".xml"
        spl_cost_data = "./Problems/Feature_Models/Cost/" + name + ".cost"
        self.ft = load_ft_url(self.url)
        self.ft.load_cost(spl_cost_data)
        lows = [0 for _ in xrange(len(self.ft.leaves))]
        ups = [1 for _ in xrange(len(self.ft.leaves))]
        names = ["x"+str(i) for i in xrange(len(self.ft.leaves))]
        self.decisions = [jmoo_decision(names[i], lows[i], ups[i]) for i in xrange(len(self.ft.leaves))]
        self.objectives = [jmoo_objective("fea", True), jmoo_objective("conVio", True), jmoo_objective("Cost", True)]

    def evaluate(self, input = None):
        t = self.ft
        if input:
            # print input
            input = [int(round(inp, 0)) for inp in input[:len(self.decisions)]]
            # obj1: features numbers
            # initialize the fulfill list
            fulfill = [-1] * t.featureNum
            for x in range(len(input)): fulfill[t.features.index(t.leaves[x])] = input[x]

            # fill other tree elements
            t.fill_form_4_all_features(fulfill)

            # here group should not count as feature
            obj1 = t.featureNum - sum(fulfill) - sum([fulfill[t.features.index(g)] for g in t.groups])

            # obj2: constraint violation
            obj2 = len(t.constraints) - sum([1 for cc in t.constraints if cc.iscorrect(t, fulfill) is True])

            # obj3: total cost
            obj3 = sum([t.cost[i] for i,f in enumerate(t.features) if (fulfill[i] == 1 and f.node_type != 'g')])

            return [obj1, obj2, obj3]
        else:
            assert False, "BOOM"
            exit()

    def evalConstraints(prob, input=None):
        return False

    """
    checking whether the candidate meets ALL constraints
    """
    def ok(self,c):
        try:
            if c.scores == []:
                self.eval(c)
        except:
            self.eval(c)
        return c.scores[1] == 0

    def genRandomCan(self,guranteeOK = False):
        while True:
            randBinList = lambda n: [random.randint(0,1) for b in range(1,n+1)]
            can = candidate(decs=randBinList(len(self.dec)),scores=[])
            if not guranteeOK or self.ok(can): break
        return can

    def printModelInfo(self):
        print '---Information for SPL--------'
        print 'Name:', self.name
        print 'Leaves #:', len(self.ft.leaves)
        print 'Total Features#:', self.ft.featureNum-len(self.ft.groups)
        print 'Constraints#:', len(self.ft.constraints)
        print '-'*30

def main():
    m = FeatureTreeModel('../feature_tree_data/cellphone.xml')
    can = m.genRandomCan()
    m.eval(can,doNorm=False)
    pdb.set_trace()

if __name__ == '__main__':
    main()