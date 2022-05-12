import pyomo.environ as pyo
import numpy as np

from math import ceil
from time import time
from copy import deepcopy

instance_name = "Instances/bin_pack_20_0.dat"

time_limit = 1

BRANCH = {
    "DEPTH_FIRST": 0,
    "BEST_FIRST": 1,
}

VARIABLE = {
    "ROUNDED": 0,
    "HALF": 1,
    "FULL": 2
}

INEQUALITIES = {
    "SOLUTION": 0,
    "PROBLEM": 1,
    "NO": 2
}


def solve_bp_lp(instance_name):
    """
    Solves the LP relaxation based on the instance_name.
    It reports in the standart input/output the time it took and the gap between the repaired heuristic and the
    optimal objective value of the LP relaxation.
    :param instance_name: (String) The name of the .dat file to run.
    :return: (Tuple) a tuple containing :
        - obj: the optimal objective value of the LP relaxation.
        - x: The optimal solution x (i.e. which object is assigned to which bag and in which proportion)
        - y: the optimal solution y (i.e. which bag are used and in which proportion)
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
                     time_limit=time_limit * 60):
    """
    Applies the branch and bound procedure to solve the LP.
    It will output the progress as it goes through the tree in the STDIO, and it will make a comparison between the result
    obtained by the branch and bound and the best result obtained by the heuristics.

    :param instance_name: (String) The name of the .dat file to run.
    :param branching_scheme: (Int) The branching method to use.
        - 0 : DEPTH_FIRST
        - 1 : BEST_FIRST
    :param variable_selection_scheme: (Int) The variable (to branch at each node) selection method to use.
        - 0 : ROUNDED (takes the variable whose fractional value is closest to an integer (in this case : 0 or 1))
        - 1 : HALF (takes the variable whose fractional value is closest to 1/2.)
        - 2 : FULL (takes the variable whose fractional value is closest to 1.)
    :param valid_inequalities: (Int) The cutting plane generation method you want to use.
        - 0 : SOLUTION (generates cutting planes based on the solution of the LP relaxation of the root node.)
        - 1 : PROBLEM (generates cutting planes based on the problem.)
        - 2 : NO (doesn't generate cutting planes.)
    :param time_limit: (Int) time in seconds available for the b&b to solve the problem.
    """
    start = time()
    size, cap, weight = extract_data(instance_name)
    heuristic = get_best_heuristic(size, cap, weight)
    root_node = build_root_node(heuristic[1], instance_name, variable_selection_scheme, valid_inequalities, size, cap,
                                weight)

    iteration = 0
    privious = 9999

    while root_node.get_lowerbound() != root_node.get_upperbound() and not root_node.get_is_done() and iteration < 10000:
        if not iteration % 1:  # you can choose every how many iteration it will print during the evolution of the b&b.
            print(sum(weight) / cap, "current lowerbound:", root_node.get_lowerbound(), "  current upperbound:",
                  root_node.get_upperbound(), "  number of iteration:", iteration, " Time ", round(time() - start, 2))

        selected = select_node_to_expand(root_node, branching_scheme)
        expand_tree(selected, instance_name, variable_selection_scheme, size, cap, weight)
        iteration += 1

        if privious > root_node.get_upperbound():
            privious = root_node.get_upperbound()

        if time() - start > time_limit:
            print("the algo took too long, the best solution found by the b&b is :", privious, "the best solution found by "
                                                                                     "heuristics is:", heuristic[1])
            break
    print("the algo is done, the objective value is :", root_node.get_upperbound(), " and initial lowerbound is :",
          ceil(sum(weight) / cap))


def extract_data(instance_name):
    """
    Open the instance_name file and extracts the weight, size and cap of the problem.
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
    :param weight: (List) a list of the weights of each object.
    :return: (List) a list containing first the solution x_pb representing if an object p is in box b, and second the
    objective value of the solution.
    """
    heur_list = []

    sol_best_fit = build_best_fit_solution(size, cap, weight)
    print("best packing : ", get_obj(sol_best_fit, size, cap, weight, True))
    heur_list.append([sol_best_fit, get_obj(sol_best_fit, size, cap, weight, True)])

    sol_greedy = build_greedy_solution(size, cap, weight)
    print("greedy packing : ", get_obj(sol_greedy, size, cap, weight, True))
    heur_list.append([sol_greedy, get_obj(sol_greedy, size, cap, weight, True)])

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
    As the objects are already sorted by weights this algorithm is equivalent to "First-Fit decreasing".
    :param size: (Int) the size of the problem.
    :param cap: (Int) the capacity of each bag.
    :param weight: (List) a list of the weights of each object.
    :return: (List) the solution x_pb representing if an object p is in box b.
    """
    bag = [cap for i in range(size)]
    solution = np.zeros((size, size))
    for obj in range(size):
        for sac in range(size):
            if bag[sac] >= weight[obj]:
                solution[obj][sac] = 1
                bag[sac] -= weight[obj]
                break
    return solution


def build_best_fit_solution(size, cap, weight):
    """
    Algorithm that will try to fill a bag such that the remaining capacity of the bag is minimal.
    :param size: (Int) the size of the problem.
    :param cap: (Int) the capacity of each bag.
    :param weight: (List) a list of the weights of each object.
    :return: (List) the solution x_pb representing if an object p is in box b.
    """
    bag = [cap for i in range(size)]
    solution = np.zeros((size, size))
    for obj in range(size):
        rest = 1000
        index = 0
        w = -1
        for sac in range(size):
            if bag[sac] >= weight[obj] and bag[sac] - weight[obj] < rest:
                rest = bag[sac] - weight[obj]
                w = weight[obj]
                index = sac
        if w != -1:
            solution[obj][index] = 1
            bag[index] -= weight[obj]
    return solution


def build_evenly_fill_solution(size, cap, weight):
    """
    Algorithm that will try to fill evenly all the bags. This result in having one object per bag.
    :param size: (Int) the size of the problem.
    :param cap: (Int) the capacity of each bag.
    :param weight: (List) a list of the weights of each object.
    :return: (List) the solution x_pb representing if an object p is in box b.
    """
    bag = [cap for i in range(size)]
    solution = np.zeros((size, size))
    for obj in range(size):
        ratio = 1
        index = 0
        for sac in range(size):
            if ratio > weight[obj] / bag[sac] and bag[sac] >= weight[obj]:
                ratio = weight[obj] / bag[sac]
                index = sac
        solution[obj][index] = 1
        bag[index] -= weight[obj]
    return solution


def build_full_packing_solution(size, cap, weight):
    """
    Algorithm that will try to find combination of objects based on the sum of their weights (s.t it is equal to the
    cap) to try and fill a bag as much as possible.
    :param size: (Int) the size of the problem.
    :param cap: (Int) the capacity of each bag.
    :param weight: (List) a list of the weights of each object.
    :return: (List) the solution x_pb representing if an object p is in box b.
    """
    if np.amax(weight) <= cap:  # check if the problem is feasible.
        bag = 0
        temp = []
        for i in range(size):
            temp.append([weight[i], i])
        weight = temp
        solution = np.zeros((size, size))
        while len(weight) > 0:
            work_cap = weight[0][0]
            best_index = [0]
            for i in range(1, len(weight)):
                work_bag = np.array([weight[0][0]], dtype=int)
                index = [0]
                for j in range(i, len(weight)):
                    if sum(work_bag) + weight[j][0] < cap:
                        work_bag = np.append(work_bag, weight[j][0])
                        index.append(j)
                    elif sum(work_bag) + weight[j][0] == cap:
                        work_bag = np.append(work_bag, weight[j][0])
                        index.append(j)
                        break
                if work_cap < sum(work_bag):
                    work_cap = sum(work_bag)
                    best_index = deepcopy(index)
                    if work_cap == cap:
                        break
            a = 0
            for i in best_index:
                solution[weight[i - a][1]][bag] = 1
                weight.pop(i - a)
                a += 1
            bag += 1
        return solution
    else:
        return np.zeros((size, size))


def get_obj(solution, size, cap, weight, rounded=False):
    """
    Computes the objective value of a given solution of a problem.
    :param solution: (List) the solution x_pb representing if an object p is in box b.
    :param size: (Int) the size of the problem.
    :param cap: (Int) the capacity of each bag.
    :param weight: (List) a list of the weights of each object.
    :param rounded: (Boolean) true if the objective value should be for the integer problem, false for the objective value
    of the LP relaxation.
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
    Creates the first node of the tree and generates the cutting planes inequalities.
    :param upperbound: (Int) the initial upperbound of the root_node is the objective value of the best heuristic.
    :param instance_name: (String) The name of the .dat file to run.
    :param variable_selection_scheme: (Int) The variable (to branch at each node) selection method to use.
        - 0 : ROUNDED (takes the variable whose fractional value is closest to an integer (in our case : 0 or 1))
        - 1 : HALF (takes the variable whose fractional value is closest to 1/2.)
        - 2 : FULL (takes the variable whose fractional value is closest to 1.)
    :param valid_inequalities: (Int) The cutting plane generation method to use.
        - 0 : SOLUTION (generates cutting planes based on the solution of the LP relaxation of the root node.)
        - 1 : PROBLEM (generates cutting planes based on the problem.)
        - 2 : NO (doesn't generate cutting planes.)
    :param size: (Int) the size of the problem.
    :param cap: (Int) the capacity of each bag.
    :param weight: (List) a list of the weights of each object.
    :return: (Object Node) returns the root node.
    """
    calculator = Calculator(instance_name)
    calculator.run()
    non_int = calculator.get_non_int(variable_selection_scheme)
    root_node = Node([], upperbound, ceil(calculator.get_objective()), None, None, non_int, [], 0, False)
    constraint = []
    if valid_inequalities == 0:
        constraint = cutting_planes_solution(calculator.get_relaxed_solution(size), size, cap, weight)
    elif valid_inequalities == 1:
        constraint = cutting_planes_problem(size, cap, weight)
    root_node.set_root(root_node)
    root_node.set_cutting_planes(constraint)
    return root_node


def cutting_planes_solution(solution, size, cap, weight):
    """
    Generates cutting planes based on the LP relaxation solution. For each bag of the solution we take the weight of the
    object whose fractional value is closest to 1. Then we devide the weight of all the object by the chosen weight and
    round down the value since the inequalities are lower or equal. We also divide the capacity (being the coefficient
    of the bags) by the same weight and also round down. We finaly obtain a new valid inequality by multipling
    elementwise the computed coefficient for the weights to their corresponding x_pb, the sum of the resulting operation
    must be smaller or equal to the y_b multiplied its coefficient.
    This way we generated for each column of the solution new cutting planes.
    :param solution: (List) the solution x_pb representing if an object p is in box b.
    :param size: (Int) the size of the problem.
    :param cap: (Int) the capacity of each bag.
    :param weight: (List) a list of the weights of each object.
    :return: (List) list of constraint
    EXAMPLE:
    constraint = [[[1, 2, 1],
                  [1, 2, 0],
                  [0, 1, 0],
                  [4, 12, 1],
                  [0, 0, 0]]]
    Meaning that we have the following constraints :
        - 1*X_11 + 1*X_21 <= 4*Y_1 + 0
        - 2*X_12 + 2*X_22 + 1*X_32 <= 12*Y_2 + 0
        - 1*X_13 <= 1*Y_3 + 0
    """
    solution = solution
    bag_constraint = np.full(size, cap, dtype=int)
    prod_constraint = np.rot90(np.tile(weight, (size, 1)), -1)
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


def cutting_planes_problem(size, cap, weight):
    """
    Generates cutting planes based on the problem. We will make a combination of the objects whose weights combined surpasses
    the capacity of a bag and we will create a new equality based on the formula:
        - b = sum of the weight of the chosen objects
        - c = capacity of a bag
        - k = ceil(b/c)
        - g = b - (k - 1) * c
    Generating the following valid inequality :
        - x <= b-g*k + g*y
    :param size: (Int) the size of the problem.
    :param cap: (Int) the capacity of each bag.
    :param weight: (List) a list of the weights of each object.
    :return: (List) list of constraints
    EXAMPLE:
    constraint = [[[70, 70, 70],
                  [0, 0, 0],
                  [30, 30, 30],
                  [20, 20, 20],
                  [50, 50, 50]],
                  [[0, 0, 0],
                  [40, 40, 40],
                  [30, 30, 30],
                  [12, 12, 12],
                  [58, 58, 58]]]
    Meanining that we have the following constraints :
        - 70*X_1j + 40*X_2j + 30X_3j <= 20*Y_j + 50 for j = 1,2,3
        - 40*X_1j + 30*X_3j <= 12*Y_j + 58 for j = 1,2,3
    """
    constraint = []
    for i in range(size):
        val = weight[i]
        selected = [i]
        for j in range(i + 1, size):
            if val < 150:
                val += weight[j]
                selected.append(j)
            else:
                break
        if val > 150:  # To consider only interesting cuts.
            sol = np.zeros(size + 2, dtype=int)
            b = 0
            for elem in selected:
                sol[elem] = weight[elem]
                b += weight[elem]
            c = cap
            k = ceil(b / c)
            g = b - (k - 1) * c
            sol[-2] = g  # constraint is x <= b - g*(k-y)
            sol[-1] = b - g * k
            constraint.append(np.rot90(np.tile(sol, (size, 1)), -1))
    return constraint


def select_node_to_expand(node, branching_scheme):
    """
    Select the next node that will be expanded next.
    :param node: (Object Node) the root node of the tree.
    :param branching_scheme: (Int) The branching method to use.
        - 0 : DEPTH_FIRST
        - 1 : BEST_FIRST
    :return: (Object Node) the selected node to expand.
    """
    selected = None

    if branching_scheme == BRANCH["DEPTH_FIRST"]:
        selected = select_node_to_expand_depth_first(node)
    elif branching_scheme == BRANCH["BEST_FIRST"]:
        selected = select_node_to_expand_best_first(node)
    else:
        print("The branching scheme selected has not been implemented yet.")
        exit(0)
    return selected


def select_node_to_expand_depth_first(node):
    """
    Select the left most node to expand if it has not been visited yet.
    :param node: (Object Node) the root node of the tree.
    :return: (Object Node) the selected node to expand.
    """
    childs = node.get_childs()
    if len(childs):
        for i in range(len(childs)):
            if not childs[i].get_is_done():
                return select_node_to_expand_depth_first(childs[i])
    return node


def select_node_to_expand_best_first(node):
    """
    Select the best node such that the child with the lowest upperbound will be selected, and then it will select the
    child with the lowest upperbound and so on until it reaches a node that is not done yet and has no childs (in other words a leaf).
    :param node: (Object Node) the root node of the tree.
    :return: (Object Node) the selected node to expand.
    """
    childs = node.get_childs()
    counter = 0
    while len(childs) and not node.get_is_done():
        u = 9999
        index = -1
        for i in range(len(childs)):
            if (not childs[i].get_is_done()) and (childs[i].get_upperbound() < u):
                u = deepcopy(childs[i].get_upperbound())
                index = i
        if index != -1:
            node = childs[i]
            childs = node.get_childs()
        counter += 1
        if counter > 10000:  # security if it starts looping for ever, but it shouldn't
            node.set_is_done(True)
            if node.get_parent() is not None:
                update_parent(node.get_parent())
            node = node.get_root()
    return node


def expand_tree(node, instance_name, variable_selection_scheme, size, cap, weight):
    """
    Expands a node by giving it 2 childs based on its LP relaxation solution. First the variable to branch on is
    selected then the problem is solved and if the solution is feasible a new child is added. At the same time, a
    reparation heuristic will repaire the LP relaxation solution just obtained to generate the upperbound of the new
    child. Once the new child is added, an update is called on the tree to update the upperbounds and lowerbounds of the
    parents.
    :param node: (Object Node) the node that will be expanded.
    :param instance_name: (String) The name of the .dat file to run.
    :param variable_selection_scheme: (Int) The variable (to branch at each node) selection method you want to use.
        - 0 : ROUNDED (takes the variable whose fractional value is closest to an integer (in our case : 0 or 1))
        - 1 : HALF (takes the variable whose fractional value is closest to 1/2.)
        - 2 : FULL (takes the variable whose fractional value is closest to 1.)
    :param size: (Int) the size of the problem.
    :param cap: (Int) the capacity of each bag.
    :param weight: (List) a list of the weights of each object.
    """
    if len(node.get_non_int()):
        constr = node.get_non_int()  # return a list with in first the position and the value [(1,2), 1], [(1,2), 0]
        for i in range(2):
            constr[1] = abs(constr[1] - i)
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
                    print("current solution found with upperbound value :", upperbound, "the lowerbound is :",
                          lowerbound, "my depth =", node.get_depth() + 1)
                else:
                    upperbound = deepcopy(calculator.compute_int_solution(size, cap, weight))
                node.add_child(
                    Node(deepcopy(constraints), deepcopy(upperbound), deepcopy(lowerbound), node, node.get_root(),
                         deepcopy(non_int), node.get_cutting_planes(), deepcopy(node.get_depth() + 1),
                         deepcopy(is_done)))
            else:
                if lowerbound > node.get_root().get_upperbound():
                    print("The solution has a lowerbound that is higher than the upperbound. It has thus not been "
                          "added in the tree")
                else:
                    print("not feasible solution")

        if len(node.get_childs()) == 0:  # Case where none of the childs are possible.
            node.set_is_done(True)
            if node.get_parent() is not None:
                update_parent(node.get_parent())
        else:
            update_parent(node)
    else:
        node.set_is_done(True)
        if node.get_parent() is not None:
            update_parent(node.get_parent())


def update_bounds(node):
    """
    Will update the bounds (upper and lower) of the node. if the parent is update, the function is called recursively
    with this time the parent of the parent to update the whole tree.
    The node received by argument will always be a parent node, meaning it will always have childs.
    But the number of childs is not fixed, it can be 1 or 2.
    :param node: (Object Node) the node that may be updated.
    """
    ub = 9999
    lb = 9999
    childs = node.get_childs()
    for i in range(len(childs)):
        if childs[i].get_lowerbound() < lb:  # Comparer les 2 lowerbound et la donner à parent.
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

    if flag:
        parent = node.get_parent()
        if parent is not None:
            update_bounds(parent)


def update_state(node):
    """
    If the node has both its childs as 'is_done=True', then we will set it to 'is_done=True'.
    If it is set to True, we apply the function to the parent node in a recursive manner.
    :param node: (Object Node) the node that may be updated.
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
    Update the node on its bound and its state (if its done or not)
    :param node: (Object Node) the parent node that may be updated.
    """
    update_bounds(node)
    update_state(node)


class Calculator:
    def __init__(self, instance_name):  # elements
        """
        Initiator of the class. 
        :param instance_name: (String) The name of the .dat file to run.
        """

        self.instance_name = instance_name
        self.model = pyo.AbstractModel()

        """ 2. LISTS AND CONSTANTS DECLARATION """

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
            if b + 1 in m.b:
                return m.y[b] >= m.y[b + 1]
            else:
                return m.y[1] >= 0 # dummy constraint because we are force to loop on all b, when we only need b-1.

        """ 6. SOLVER SETTINGS """

        self.solver = pyo.SolverFactory('glpk')
        self.solver.options['tmlim'] = time_limit

        """ 7. DATA GATHERING """

        self.data = pyo.DataPortal(model=self.model)
        self.data.load(filename=self.instance_name, model=self.model)
        self.instance = self.model.create_instance(self.data)
        self.instance.constraint_list = pyo.ConstraintList()

    def run(self):
        """ 8. RUNNING THE SOLVER """
        result = self.solver.solve(self.instance)  # , tee=True).write()

    def affichage_result(self):
        """
        Displays the result of the LP relaxation in the STDIO.
        Only displays the value for x and y that are different than 0, in oher words, only the relevant information.
        """
        # instance.display() # usefull command to show to full result but quite heavy in the output stream

        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) != 0:
                print(self.instance.x[j], "=", pyo.value(self.instance.x[j]))

        for i in self.instance.y:
            if pyo.value(self.instance.y[i]) != 0:
                print(self.instance.y[i], "=", pyo.value(self.instance.y[i]))

        print("Objective value of LP relaxation:", pyo.value(self.instance.obj_expression))

    def add_constraint_model(self, corrected):
        """
        Adds a constaint on a specific varible to be etiher grater than 1 or smaller than 0. This constraint is used
        on the branching part.
        :param corrected: (List) a list ocntaining the variable to put a constraint on, and the value to put it to.
        """
        for elem in corrected:
            if elem[1]:
                self.instance.constraint_list.add(self.instance.x[elem[0]] >= elem[1])
            else:
                self.instance.constraint_list.add(self.instance.x[elem[0]] <= elem[1])

    def add_cutting_planes(self, constraint):
        """
        Adds the cutting plane to the constraint of the instances.
        :param constraint: (List) list of all the constraint to add. See the example to follow the format of the
        constraint you want to add.
        Example :
        constraint = [[[70, 70, 70],
                      [0, 0, 0],
                      [30, 30, 30],
                      [20, 20, 20],
                      [50, 50, 50]],
                      [[0, 0, 0],
                      [40, 40, 40],
                      [30, 30, 30],
                      [12, 12, 12],
                      [58, 58, 58]]]
        Meanining that we have the following constraints :
            - 70*X_1j + 40*X_2j + 30X_3j <= 20*Y_j + 50 for j = 1,2,3
            - 40*X_1j + 30*X_3j <= 12*Y_j + 58 for j = 1,2,3
        """
        for m in range(len(constraint)):
            for j in range(len(constraint[m][0])):
                self.instance.constraint_list.add(
                    sum(self.instance.x[(i, j)] * constraint[m][i][j] for i in range(len(constraint[m]) - 2)) <=
                    self.instance.y[j] * constraint[m][-2][j] + constraint[m][-1][j])

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
        """
        Constructs the X_pb solution based on the solution the solver has generated and returns it.
        :param size: (Int) the size of the problem.
        :return: (List) The X_pb solution
        """
        solution = np.zeros((size, size))
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]):
                solution[j] = pyo.value(self.instance.x[j])
        return solution

    def get_relaxed_bags(self, size):
        """
        Constructs the Y_b solution based on the solution the solver has generated and returns it.
        :param size: (Int) the size of the problem.
        :return: (List) The Y_b solution
        """
        solution = np.zeros(size)
        for j in self.instance.y:
            if pyo.value(self.instance.y[j]):
                solution[j] = pyo.value(self.instance.y[j])
        return solution

    def get_one_values(self):
        """
        Will gather in a list (list_int) all the variable of the solution which value is equal to 1.

        Example of the list_int :
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
        Compute the upperbound value of the LP relaxation by applying a repair heurisitic on the LP relaxation provided
        by the solver. 2 repair heuristics have been implemented:
            - 0 : BEST FIT
            - 1 : FIRST FIT
        By experimentation, best fit always provides better or equal solution to first fist, thus all our experiments
        have been conducted using best fit, and by default it will always apply best-fit.
        :param size: (Int) the size of the problem.
        :param cap: (Int) the capacity of each bag.
        :param weight: (List) a list of the weights of each object.
        :param scheme: (Int) the scheme used to repair the solution. (either 0 or 1)
        :return: (Int) the objective value of the repaired solution.
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
        else:
            print("This repair Heuristic has not been implemented yet.")
            exit(1)
        return up

    def rebuild_best_fit(self, size, bag, solution, obj, weight):
        """
        Repairs a LP relaxation solution using best fit. To do so, first we gather all the integer value (equal to 1
        only) of the lp relaxation and set them to one in our reparation. Then for all the object that have not yet
        been assigned a bag, we will assign to them using best fit (i.e. putting it in the bag where the space left is
        minimum).
        :param size: (Int) The size of the problem.
        :param bag: (List) The remaining space available for each bag.
        :param solution: (List) Solution with integer value of the LP relaxation already fixed.
        :param obj: (List) The object which haven been assigned to a bag yet.
        :param weight: (List) a list of the weights of each object.
        :return: (Int) the objective value of the repaired solution.
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
        Repairs a LP relaxation solution using first fit. To do so, first we gather all the integer value (equal to 1
        only) of the lp relaxation and set them to one in our reparation. Then for all the object that have not yet
        been assigned a bag, we will assign to them using first fit (i.e. putting it in the first bag that can accept
        the object).
        :param size: (Int) The size of the problem.
        :param bag: (List) The remaining space available for each bag.
        :param solution: (List) Solution with integer value of the LP relaxation already fixed.
        :param obj: (List) The object which haven been assigned to a bag yet.
        :param weight: (List) a list of the weights of each object.
        :return: (Int) the objective value of the repaired solution.
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
        :return: (Int) the objective value of the LP relaxation.
        """
        return pyo.value(self.instance.obj_expression)

    def get_int_objective(self, size):
        solution = np.zeros((size, size))
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]):
                solution[j] = 1
        return np.count_nonzero(sum(solution))

    def checkFinishedProduct(self):
        """
        Checks if there are fractional value in the LP relaxation solution X_pb.
        :return: (Boolean) True if there are fractional values, False otherwhise.
        """
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) != 0 and pyo.value(self.instance.x[j]) != 1:
                return False
        return True


class Node:
    def __init__(self, constraints, upperbound, lowerbound, parent, root, non_int, cutting_planes, depth, is_done):
        """
        Initiator of the class.
        :param constraints: (List) The constriant that have been branched on its parent and it self.
        :param upperbound: (Int) the upperbound of the node.
        :param lowerbound: (Int) the lowerbound of the node.
        :param parent: (Object Node) The parent node.
        :param root: (Object Node) The root node.
        :param non_int: (List) all the fractional variable and value of the LP relaxation of this node.
        :param cutting_planes: (List) The cutting planes of the node.
        :param depth: (Int) The depth of the node in the tree.
        :param is_done: (Boolean) If the node is a leaf node, or if it can have children.
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


    """ GETTERS AND SETTERS """

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
branch_and_bound(instance_name, branching_scheme=BRANCH["BEST_FIRST"], variable_selection_scheme=VARIABLE["FULL"],
                 valid_inequalities=INEQUALITIES["PROBLEM"])
