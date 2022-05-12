import pyomo.environ as pyo
from math import ceil

from time import time
from copy import deepcopy
import numpy as np

instance_name = "Instances/bin_pack_500_0.dat"

time_limit = 10

BRANCH = {
    "DEPTH_FIRST" : 0,
    "BEST_FIRST" : 1,
}

VARIABLE = {
    "ROUNDED" : 0,
    "HALF" : 1,
    "FULL" : 2
}

INEQUALITIES = {
    "SOLUTION" : 0,
    "PROBLEM" : 1,
}

def solve_bp_lp(instance_name):
    """
    Solves the LP relaxation based on the instance_name.
    It reports in the standart input/output the time it took and the gab between the repaired heuristic and the
    optimal objective value of the LP relaxation.
    :param instance_name: (String) The name of the .dat file to run.
    :return: (Tuple) a tuple containing :
        - obj: the optimal objective value of the LP relaxation.
        - x: The optimal solution x (i.e. which object is assigned to which bag and in which proportion)
        - y: the optimal solution y (i.e. which bag are used and in which proportion
    """
    size, cap, weight = extract_data(instance_name)
    calculator = Calculator(instance_name)
    start = time()
    calculator.run()
    print("time to solve the relaxed problem:", time() - start)
    x = calculator.get_relaxed_solution(size)
    y = calculator.get_relaxed_bags(size)
    repaired_objective = calculator.compute_int_solution(size, cap, weight, scheme=0)
    obj = calculator.get_objective()
    print("gap between objective and heuristic reparation : gap = repaired - obj =", repaired_objective - obj)
    return (obj, x, y)

def branch_and_bound(instance_name, branching_scheme=1, variable_selection_scheme=2, valid_inequalities=1,
                     time_limit=time_limit*60):
    """
    Applies the branch and bound procedure to try to solve the LP.
    It will output progresse as it goes to the tree in the STDIO, and will make a comparaison between the result
    obtained by the branch and bound and the best result obtained by the heuristics.

    :param instance_name: (String) The name of the .dat file to run.
    :param branching_scheme: (Int) The branching methode you want to use.
        - 0 : DEPTH_FIRST
        - 1 : BEST_FIRST
    :param variable_selection_scheme: (Int) The variable (to branch at each node) selection methode you want to use.
        - 0 : ROUNDED (takes the variable whose fractional value is closest to an integer (in our case : 0 or 1))
        - 1 : HALF (takes the variable whose fractional value is closest to 1/2.)
        - 2 : FULL (takes the variable whose fractional value is closest to 1.)
    :param valid_inequalities: (Int) The cutting planes generation methode you want to use.
        - 0 : SOLUTION (generates cutting planes based on the solution of the LP relaxation of the root node.)
        - 1 : PROBLEM (generates cutting planes based on the problem.)
    :param time_limit: (Int) time in seconds available for the b&b to solve the problem.
    """
    start = time()
    size, cap, weight = extract_data(instance_name)
    heuristic = get_best_heuristic(size, cap, weight)
    root_node = build_root_node(heuristic[1], instance_name, variable_selection_scheme, valid_inequalities, size, cap,
                                weight)

    iteration = 0
    privious = 9999

    while root_node.get_lowerbound() != root_node.get_upperbound() and not root_node.get_is_done() and iteration <10000:
        if not iteration % 1: # you can choose every how many iteration it will print the evolution of the b&b.
            print(sum(weight)/cap, "current lowerbound:", root_node.get_lowerbound(), "  current upperbound:",
                  root_node.get_upperbound(), "  number of iteration:", iteration, " Time ", round(time() - start, 2))
           
        selected = select_node_to_expand(root_node, branching_scheme)
        expand_tree(selected, instance_name, variable_selection_scheme, size, cap, weight)
        iteration += 1

        if privious > root_node.get_upperbound():
            privious = root_node.get_upperbound()

        if time() - start > time_limit:
            print("the algo took to long best solution found by b&b is :", privious, "the best solution found by "
                                                                                     "heuristics is:", heuristic[1])
            break
    print("the algo is done, the objective value is :", root_node.get_upperbound(), " and initial lowerbound is :",
          ceil(sum(weight)/cap))

def extract_data(instance_name):
    """
    Opens the instance_name file and extracts the weight, size and cap of the problem.
    :param instance_name: (String) The name of the .dat file to run.
    :return: (Int) size: the size of the problem, (Int) cap: the capacity of each bag, (List) weight: list of the
    weights of the objects.
    """
    empty_list = []
    with open(instance_name) as datFile:
        for data in datFile:
            empty_list.append(data.split())

    size = int(empty_list[0][-1].split(';')[0]) + 1
    cap = int(empty_list[2][-1].split(';')[0])
    weight = []

    for i in range(5, len(empty_list)):
        weight.append(int(empty_list[i][1].split(';')[0]))

    weight = np.array(weight, dtype=int)

    return size, cap, weight

def get_best_heuristic(size, cap, weight):
    """
    Runs all the heuristics implemented and selects the one with the best solution (i.e. the best objective value).
    ex : best = [sol, obj]
    :param size: (Int) the size of the problem.
    :param cap: (Int) the capacity of each bag.
    :param weight: (List) a list of the weights of each objects.
    :return: (List) a list containing first
    """
    heur_list = []

    sol_best_fit = build_best_fit_solution(size, cap, weight)
    print("best packing : ", get_obj(sol_best_fit, size, cap, weight, True))
    heur_list.append([sol_best_fit, get_obj(sol_best_fit, size, cap, weight, True) ])

    sol_greedy = build_greedy_solution(size, cap, weight)
    print("greedy packing : ", get_obj(sol_greedy, size, cap, weight, True))
    heur_list.append([sol_greedy , get_obj(sol_greedy, size, cap, weight, True)])

    sol_evenly_fill = build_evenly_fill_solution(size, cap, weight)
    print("evenly packing : ", get_obj(sol_evenly_fill, size, cap, weight, True))
    heur_list.append([sol_evenly_fill, get_obj(sol_evenly_fill, size, cap, weight, True)])

    sol_full_packing = build_full_packing_solution(size, cap, weight)
    print("full packing : ", get_obj(sol_full_packing, size, cap, weight, True))
    heur_list.append([sol_full_packing, get_obj(sol_full_packing, size, cap, weight, True)])
    
    best = 9999
    elem = []
    for i in range(len(heur_list)):
        if heur_list[i][1] < best: 
            elem = heur_list[i]
            best = heur_list[i][1]
    return elem

def build_greedy_solution(size, cap, weight):
    """
    Greedy heuristic algorithm that will fill the first bag that can accept an object.
    It will return the first solution for the root node.
    As the objects are already sorted by weights this algorithm is equivalent to "First-Fit decreasing".
    :param size:
    :param cap:
    :param weight:
    :return:
    """
    bag = [cap for i in range(size)]
    solution = np.zeros((size, size))
    for obj in range(size):
        for sac in range(size):
            if bag[sac] >= weight[obj]:
                solution[obj][sac] = 1
                bag[sac]-=weight[obj]
                break
    return solution

def build_best_fit_solution(size, cap, weight):
    """
    Algorithm that will try to fill a bag such that it will be the closest of being full. 
    It will return the first solution for the root node.
    :param size:
    :param cap:
    :param weight:
    :return:
    """
    bag = [cap for i in range(size)]
    solution = np.zeros((size, size))
    for obj in range(size):
        rest = 1000
        index = 0
        w = -1
        for sac in range(size):
            if bag[sac] >= weight[obj] and bag[sac]-weight[obj]<rest:
                rest = bag[sac] - weight[obj]
                w = weight[obj]
                index = sac
        if w != -1:
            solution[obj][index] = 1
            bag[index]-=weight[obj]
    return solution   

def build_evenly_fill_solution(size, cap, weight):
    """
    Algorithm that will try to fill a bag such that it will be the closest of being full. 
    It will return the first solution for the root node.
    :param size:
    :param cap:
    :param weight:
    :return:
    """
    bag = [cap for i in range(size)]
    solution = np.zeros((size, size))
    for obj in range(size):
        ratio = 1
        index = 0
        for sac in range(size):
            if ratio > weight[obj]/bag[sac] and bag[sac]>=weight[obj]:
                ratio = weight[obj]/bag[sac]
                index = sac
        solution[obj][index] = 1
        bag[index]-=weight[obj]
    return solution

def build_full_packing_solution(size, cap, weight):
    """
    Algorithm that will try to find combination of objects based on the sum of their weights (s.t it is equal to the cap) to try and
    fill a bag as much as possible. 
    It will return the first solution for the root node.
    :param size:
    :param cap:
    :param weight:
    :return:
    """
    if np.amax(weight)<=cap: # check if the problem is feasible.
        bag = 0
        temp = []
        for i in range(size):
            temp.append([weight[i], i])
        weight = temp
        solution = np.zeros((size, size))
        while len(weight)>0:
            work_cap = weight[0][0]
            best_index = [0]
            for i in range(1, len(weight)):
                work_bag = np.array([weight[0][0]], dtype=int)
                index = [0]
                for j in range(i, len(weight)):
                    if sum(work_bag)+weight[j][0] < cap:
                        work_bag = np.append(work_bag,weight[j][0])
                        index.append(j)
                    elif sum(work_bag)+weight[j][0] == cap:
                        work_bag = np.append(work_bag,weight[j][0])
                        index.append(j)
                        break
                if work_cap < sum(work_bag):
                    work_cap = sum(work_bag)
                    best_index = deepcopy(index)
                    if work_cap == cap:
                        break
            a = 0
            for i in best_index:
                solution[weight[i-a][1]][bag] = 1
                weight.pop(i-a)
                a+=1
            bag +=1
        return solution  
    else :
        return np.zeros((size, size))

def get_obj(solution, size, cap, weight, rounded=False):
    """
    Computes the objective value of a given solution of a problem.
    :param solution:
    :param size:
    :param cap:
    :param weight:
    :param rounded:
    :return: (Int) the Objective value
    """
    if rounded:
        value = np.count_nonzero(sum(np.asarray(solution))) 
    else:
        bag = np.zeros(size)
        for col in range(size):
            for rows in range(size):
                bag[col] += solution[rows][col] * weight[rows] / cap
        value = sum(bag)
    return value

def build_root_node(upperbound, instance_name, variable_selection_scheme, valid_inequalities, size, cap, weight):
    """

    :param upperbound:
    :param instance_name:
    :param variable_selection_scheme:
    :param valid_inequalities:
    :param size:
    :param cap:
    :param weight:
    :return:
    """
    calculator = Calculator(instance_name)
    calculator.run()
    non_int = calculator.get_non_int(variable_selection_scheme)
    root_node = Node([], upperbound, ceil(calculator.get_objective()), None, None, non_int, [], 0, False)
    constraint = []
    if valid_inequalities == 0:
        constraint = cutting_planes_1(calculator.get_relaxed_solution(size), size, cap, weight)
    elif valid_inequalities == 1:
        constraint = cutting_planes_2(size, cap, weight)
    root_node.set_root(root_node)
    root_node.set_cutting_planes(constraint)
    return root_node

def cutting_planes_2(size, cap, weight):
    """

    :param size:
    :param cap:
    :param weight:
    :return:
    """
    constraint = []
    for i in range(size):
        val = weight[i]
        selected = [i]
        for j in range(i+1,size): 
            if val < 150:
                val += weight[j]
                selected.append(j)
            else:
                break
        if val > 150: #To consider only interesting cuts.
            sol = np.zeros(size+2, dtype=int)
            b = 0
            for elem in selected:
                sol[elem] = weight[elem]
                b += weight[elem]
            c = cap
            k = ceil(b/c)
            g = b-(k-1)*c
            sol[-2] = g  # constraint is x <= b - g*(k-y)
            sol[-1] = b-g*k
            constraint.append(np.rot90(np.tile(sol,(size,1)), -1))
    return constraint


def cutting_planes_1(solution, size, cap, weight):
    """

    :param solution:
    :param size:
    :param cap:
    :param weight:
    :return:
    """
    solution = solution
    bag_constraint = np.full(size, cap, dtype=int)
    prod_constraint = np.rot90(np.tile(weight,(size,1)), -1)
    maximals = np.ones(size, dtype=int)
    for box in range(size):
        index_max = 0
        value = 0
        for prod in range(size):
            if solution[prod][box] != 1 and solution[prod][box] > value:
                value = solution[prod][box]
                index_max = prod
        maximals[box] = prod_constraint[index_max][box]
    prod_constraint = np.floor_divide(prod_constraint, maximals)
    bag_constraint = np.floor_divide(bag_constraint, maximals)
    blank = np.zeros_like(bag_constraint, dtype=int)
    constraint = np.vstack((prod_constraint, bag_constraint))
    constraint = np.vstack((constraint, blank))
    return [constraint]

def select_node_to_expand(node, branching_scheme):
    """

    :param node:
    :param branching_scheme:
    :return:
    """
    selected = None

    if branching_scheme == BRANCH["DEPTH_FIRST"]:
        selected = select_node_to_expand_depth_first(node)
    elif branching_scheme == BRANCH["BEST_FIRST"]:
        selected = select_node_to_expand_best_first(node)
    else :
        print("Sorry, the branching scheme selected has not been implemented yet.")
        exit(0)
    return selected
    
def select_node_to_expand_depth_first(node):
    """
    Select the left most node to expand if it has not been visited yet.
    :param node:
    :return:
    """
    childs = node.get_childs()
    if len(childs):
        for i in range(len(childs)):
            if not childs[i].get_is_done():
                return select_node_to_expand_depth_first(childs[i])
    return node

def select_node_to_expand_best_first(node):
    """
    Select the best node such that the node with the lowest upperbound
    will be expanded if it has not been visited yet.
    :param node:
    :return:
    """
    childs = node.get_childs()
    counter = 0
    while len(childs) and not node.get_is_done():
        u = 9999
        index = -1
        for i in range(len(childs)):
            if (not childs[i].get_is_done()) and (childs[i].get_upperbound() < u ):
                u = deepcopy(childs[i].get_upperbound())
                index = i
        if index != -1:
            node = childs[i]
            childs = node.get_childs()
        counter +=1
        if counter >10000:
            node.set_is_done(True)
            if node.get_parent() is not None:
                update_parent(node.get_parent())
            node = node.get_root()
    return node

def expand_tree(node, instance_name, variable_selection_scheme, size, cap, weight):
    """

    :param node:
    :param instance_name:
    :param variable_selection_scheme:
    :param size:
    :param cap:
    :param weight:
    :return:
    """
    if len(node.get_non_int()):
        constr = node.get_non_int() #return a list with in first the position and the value [(1,2), 1], [(1,2), 0]
        for i in range(2):
            constr[1] = abs(constr[1]-i)
            constraints = deepcopy(node.get_constraints())
            constraints.append(constr)
            calculator = Calculator(instance_name)
            calculator.add_constraint_model(constraints)
            calculator.add_cutting_planes(node.get_cutting_planes())
            calculator.run()
            lowerbound = ceil(calculator.get_objective())
            if lowerbound and lowerbound <= node.get_root().get_upperbound():
                non_int = calculator.get_non_int(variable_selection_scheme)
                upperbound = None
                is_done = False
                if calculator.checkFinishedProduct():
                    upperbound = deepcopy(calculator.get_int_objective(size))
                    is_done = True
                    print("current solution found with upperbound value :", upperbound, "the lowerbound is :", lowerbound, "my depth =", node.get_depth()+1)  
                else :
                    upperbound = deepcopy(calculator.compute_int_solution(size, cap, weight))
                node.add_child(Node(deepcopy(constraints), deepcopy(upperbound), deepcopy(lowerbound), node, node.get_root(), deepcopy(non_int), node.get_cutting_planes(),deepcopy(node.get_depth()+1) ,deepcopy(is_done)))
            else:
                if lowerbound > node.get_root().get_upperbound():
                    print("The solution has a lowerbound that is higher than the upperbound. It has thus not been added in the tree")
                else :
                    print("not feasible solution")  

        if len(node.get_childs()) == 0: #Case where none of the childs are possible. 
            node.set_is_done(True)
            if node.get_parent() is not None:
                update_parent(node.get_parent())  
        else:
            update_parent(node)
    else :
        node.set_is_done(True)
        if node.get_parent() is not None:
            update_parent(node.get_parent())

def update_bounds(node):
    """
    The node received by argument will always be a parent node, meaning it will always have childs.
    But the number of childs is not fixed, it can be 1 or 2.
    :param node:
    :return:
    """
    ub=9999
    lb=9999
    childs = node.get_childs()   
    for i in range(len(childs)):   
        if childs[i].get_lowerbound()<lb: #Comparer les 2 lowerbound et la donner à parent. 
            lb = deepcopy(childs[i].get_lowerbound())
        if childs[i].get_upperbound() is not None and childs[i].get_upperbound() < ub:
            ub = deepcopy(childs[i].get_upperbound())
    flag = False
    
    if ub != 9999 and ub != node.get_upperbound():
        node.set_upperbound(ub)  
        flag = True

    if lb != 9999 and lb != node.get_lowerbound():
        node.set_lowerbound(lb)
        flag = True

    if flag :
        parent = node.get_parent()
        if parent is not None:
            update_bounds(parent)

def update_state(node):
    """
    If the node has both its childs as 'is_done=True', then we will set it to 'is_done=True'.
    If it is set to True, we apply the function to the parent node in a recursive manner.
    :param node:
    :return:
    """
    flag = True
    childs = node.get_childs() 
    for i in range(len(childs)):   
        if not childs[i].get_is_done():
            flag = False
    if flag:
        node.set_is_done(True)
        parent = node.get_parent()
        if parent is not None:
            update_state(parent)

def update_parent(node):
    """

    :param node:
    :return:
    """
    update_bounds(node)
    update_state(node)

class Calculator:
    def __init__(self, filename):  # elements
        """
        Initiator of the class. 
        :param filename:
        """

        self.filename = filename
        self.model = pyo.AbstractModel()

        """ 2. LISTS and CONSTANTS DECLARATION """

        self.model.I = pyo.Set()
        self.model.cap = pyo.Param()
        self.model.size = pyo.Param(self.model.I, within=pyo.NonNegativeIntegers)

        self.model.p = pyo.Set(initialize=self.model.I)
        self.model.b = pyo.Set(initialize=self.model.I)

        """ 3. VARIABLE DECLARATION """

        self.model.y = pyo.Var(self.model.b, domain=pyo.NonNegativeReals, initialize=0,
                               bounds=(0, 1))  # 1 if box b is used
        self.model.x = pyo.Var(self.model.p, self.model.b, domain=pyo.NonNegativeReals, initialize=0,
                               bounds=(0, 1))  # 1 if product p is in box b

        """ 4. OBJECTIVE FUNCTION DECLARATION """

        @self.model.Objective()
        def obj_expression(m):
            return pyo.summation(m.y)

        """ 5. CONSTRAINTS DECLARATION """

        @self.model.Constraint(self.model.b)
        def xcy_constraint_rule(m, b):
            return sum(m.x[p, b] * m.size[p] for p in m.p) <= m.cap * m.y[b]

        @self.model.Constraint(self.model.p)
        def x_constraint_rule(m, p):
            return sum(m.x[p, b] for b in m.b) == 1

        # This constraint is to remove the symmetry from the problem.
        @self.model.Constraint(self.model.b)
        def y_constraint_rule(m, b):
            if b+1 in m.b:
                return m.y[b] >= m.y[b+1]
            else:
                return m.y[1] >= 0

        """ 6. SOLVER SETTINGS """

        self.solver = pyo.SolverFactory('glpk')
        self.solver.options['tmlim'] = time_limit

        """ 7. DATA GATHERING """

        self.data = pyo.DataPortal(model=self.model)
        self.data.load(filename=self.filename, model=self.model)
        self.instance = self.model.create_instance(self.data)
        self.instance.constraint_list = pyo.ConstraintList()

    def run(self):
        """ 8. RUNNING THE SOLVER """
        result = self.solver.solve(self.instance)  # , tee=True).write()

    def affichage_result(self):
        """

        """
        """ 9. RESULTS RECUPERATION """
        # instance.display() # usefull command to show to full result but quite heavy in the output stream

        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) != 0:
                print(self.instance.x[j], " of value ", pyo.value(self.instance.x[j]))

        for i in self.instance.y:
            if pyo.value(self.instance.y[i]) != 0:
                print(self.instance.y[i], " of value ", pyo.value(self.instance.y[i]))

        print(pyo.value(self.instance.obj_expression))

    def add_constraint_model(self, corrected):
        """

        :param corrected:
        :return:
        """
        for elem in corrected:
            if elem[1]:
                self.instance.constraint_list.add(self.instance.x[elem[0]] >= elem[1])
            else:
                self.instance.constraint_list.add(self.instance.x[elem[0]] <= elem[1])

    def add_cutting_planes(self, constraint):
        for m in range(len(constraint)):
            for j in range(len(constraint[m][0])):
                self.instance.constraint_list.add(sum(self.instance.x[(i,j)]*constraint[m][i][j] for i in range(len(constraint[m])-2)) <= self.instance.y[j]*constraint[m][-2][j]+constraint[m][-1][j])

    def get_non_int(self, variable_selection_scheme):
        """
        It will return the variable that will be branched on. To do so it first gathers all fractional value in the
        solution computed by the solver (i.e. GLPK). The position of the fractional values and their actual value are
        stored in the list : list_non_int. Once we gathered them all, we send this list to the compute dist methode
        which will return the variable selected to branch.

        :param variable_selection_scheme: (Int) The scheme used to select the variable (cf. compute_dist())
        :return: (List) [pos, value] the position of the variable and the integer it was closest to.
        """
        list_non_int = []
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) < 0.99 and pyo.value(self.instance.x[j]) > 0.01:
                pos = j
                value = pyo.value(self.instance.x[j])
                list_non_int.append([pos, value])
        return self.compute_dist(list_non_int, variable_selection_scheme)

    def compute_dist(self, all_frac, variable_selection_scheme):
        """
        It will select the variable x[i,j] that will be branched. To select it, a distance is computed (depending on the
        variable_selection_scheme) for each fractionnal vraible and the one with the minimal distance is selected.


        :param all_frac: (List) of all fractional value.

        Example of all_frac:
            - all_frac = [elem_1, elem_2, ..., elem_n].
            - elem_i = [pos, value] for all i = 1, ..., n.
            - pos = (x,y) tuple of coordinates of the fractional value in the solution matrix.
            - value = value of the fractional value at the given position.

        :param variable_selection_scheme: (Int) The scheme used.

        The different available schemes are
            - 0 -> variable whose fractional value is closest to int (either 1 or 0).
            - 1 -> variable whose fractional value is closest to 1/2.
            - 2 -> variable whose fractional value is closest to 1.
        By testing, the one who works best is variable_selection_scheme=2.

        :return: (List) [pos, value] the position of the variable and the integer it was closest to.
        """
        best = 2
        coord = []
        for i in all_frac:
            if variable_selection_scheme == 0:
                dist = 0.5 - abs(i[1] - 0.5)
            elif variable_selection_scheme == 1:
                dist = abs(i[1] - 0.5)
            elif variable_selection_scheme == 2:
                dist = abs(i[1] - 1)
            if dist < best:
                best = dist
                coord = i
        if len(coord):
            coord[1] = round(coord[1])  # If the value is = 1/2 the chance of it being a 1 or a 0 is equiprobable.
        return coord

    def get_relaxed_solution(self, size):
        solution = np.zeros((size,size))
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]):
                solution[j] = pyo.value(self.instance.x[j])
        return solution

    def get_relaxed_bags(self, size):
        solution = np.zeros(size)
        for j in self.instance.y:
            if pyo.value(self.instance.y[j]):
                solution[j] = pyo.value(self.instance.y[j])
        return solution

    def get_one_values(self):
        """
        Will gather in a list (list_int) all the variable of the solution which value is equal to 1.

        Example of the list :
            - list_int = [pos_1, pos_2, ..., pos_n]
            - pos_i = (x,y) tuple of coordinates of the fractional value in the solution matrix.
        :return: (List) list_int the position of the variables which equal 1.
        """
        list_int = []
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) == 1:
                pos = j
                list_int.append(pos)
        return list_int

    def compute_int_solution(self, size, cap, weight, scheme=0):
        """

        :param size:
        :param cap:
        :param weight:
        :param scheme:
        :return:
        """
        fixed_values = self.get_one_values()
        bag = [cap for i in range(size)]
        solution = np.zeros((size, size))
        obj = [i for i in range(size)]

        for pos in fixed_values:
            solution[pos] = 1
            bag[pos[1]] -= weight[pos[0]]
            obj.remove(pos[0])

        up = 0

        if scheme == 0:
            up = self.rebuild_best_fit(size, bag, solution, obj, weight)
        elif scheme == 1:
            up = self.rebuild_first_fit(size, bag, solution, obj, weight)
        return up

    def rebuild_best_fit(self, size, bag, solution, obj, weight):
        """

        :param size:
        :param bag:
        :param solution:
        :param obj:
        :param weight:
        :return:
        """
        for o in obj:
            rest = 1000
            index = 0
            w = -1
            for sac in range(size):
                if bag[sac] >= weight[o] and bag[sac] - weight[o] < rest:
                    rest = bag[sac] - weight[o]
                    w = weight[o]
                    index = sac
            if w != -1:
                solution[o][index] = 1
                bag[index] -= weight[o]
            else:
                print("solution impossible to build")
        used = sum(solution)
        up = 0
        for i in used:
            if i > 0:
                up += 1
        return up

    def rebuild_first_fit(self, size, bag, solution, obj, weight):
        """

        :param size:
        :param bag:
        :param solution:
        :param obj:
        :param weight:
        :return:
        """
        for o in obj:
            for sac in range(size):
                if bag[sac] >= weight[o]:
                    solution[o][sac] = 1
                    bag[sac] -= weight[o]
                    break

        used = sum(solution)
        up = 0
        for i in used:
            if i > 0:
                up += 1
        return up

    def get_objective(self):
        """

        :return:
        """
        return pyo.value(self.instance.obj_expression)
    
    def get_int_objective(self, size):
        solution = np.zeros((size,size))
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]):
                solution[j] = 1
        return np.count_nonzero(sum(solution)) 

    def checkFinishedProduct(self):
        """

        :return:
        """
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) != 0 and pyo.value(self.instance.x[j]) != 1:
                return False
        return True

class Node:
    def __init__(self, constraints, upperbound, lowerbound, parent, root, non_int, cutting_planes, depth, is_done):
        """

        :param constraints:
        :param upperbound:
        :param lowerbound:
        :param parent:
        :param root:
        :param non_int:
        :param depth:
        :param is_done:
        """
        self.cutting_planes = cutting_planes
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.constraints = constraints
        self.parent = parent
        self.root = root
        self.childs = []
        self.non_int = non_int
        self.is_done = is_done
        self.depth = depth

    def get_cutting_planes(self):
        return self.cutting_planes

    def set_cutting_planes(self, cutting_planes):
        self.cutting_planes = cutting_planes

    def get_depth(self):
        return deepcopy(self.depth)

    def set_depth(self, depth):
        self.depth = depth

    def get_non_int(self):
        return self.non_int

    def set_non_int(self, non_int):
        self.non_int = non_int

    def get_root(self):
        return self.root

    def set_root(self, root):
        self.root = root

    def get_is_done(self):
        return self.is_done

    def toggle_is_done(self):
        self.is_done = not self.is_done

    def set_is_done(self, is_done):
        self.is_done = is_done

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_childs(self):
        return self.childs

    def add_child(self, child):
        self.childs.append(child)

    def set_childs(self, childs):
        self.childs = childs

    def get_upperbound(self):
        return self.upperbound

    def set_upperbound(self, upperbound):
        self.upperbound = upperbound

    def get_lowerbound(self):
        return self.lowerbound

    def set_lowerbound(self, lowerbound):
        self.lowerbound = lowerbound

    def get_constraints(self):
        if self.constraints is None:
            return []
        return deepcopy(self.constraints)

    def set_constraints(self, constraints):
        self.constraints = constraints

solve_bp_lp(instance_name)
branch_and_bound(instance_name, branching_scheme=BRANCH["BEST_FIRST"], variable_selection_scheme=VARIABLE["FULL"], valid_inequalities=2)#INEQUALITIES["PROBLEM"])