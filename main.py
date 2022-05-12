from math import ceil, floor

from time import time
from calculator import TIME_LIMIT, Calculator
from copy import deepcopy
import numpy as np
from node import Node


file_name = "Instances/bin_pack_50_1.dat"

BRANCH = {
    "DEPTH_FIRST" : 0,
    "BEST_FIRST" : 1,
}

VARIABLE = {
    "ROUNDED" : 0,
    "HALF" : 1,
    "FULL" : 2
}

def extract_data(filename):
    """

    :param filename:
    :return:
    """
    empty_list = []
    with open(filename) as datFile:
        for data in datFile:
            empty_list.append(data.split())

    size = int(empty_list[0][-1].split(';')[0]) + 1
    cap = int(empty_list[2][-1].split(';')[0])
    weight = []

    for i in range(5, len(empty_list)):
        weight.append(int(empty_list[i][1].split(';')[0]))

    weight = np.array(weight, dtype=int)

    return size, cap, weight


def branch_and_bound(instance_name, branching_scheme=0, variable_selection_scheme=0, valid_inequalities=0, time_limit=600):
    """

    :param instance_name:
    :param branching_scheme:
    :param variable_selection_scheme:
    :param valid_inequalities:
    :param time_limit:
    :return:
    """
    start = time()
    size, cap, weight = extract_data(instance_name)
    heuristic = get_best_heuristic(size, cap, weight)
    root_node = build_root_node(heuristic[1], file_name, variable_selection_scheme, valid_inequalities, size, cap, weight)
    iteration = 0
    privious = root_node.get_upperbound()
    text = str("current upperbound:"+ str(root_node.get_lowerbound())+ "  number of iteration:"+ str(iteration+1)+ " Time "+ str(round(time() - start, 2))+"\n")
    while root_node.get_lowerbound() != root_node.get_upperbound() and not root_node.get_is_done() and iteration <10000:
        if not iteration % 1:
            print(sum(weight)/cap, "current lowerbound:", root_node.get_lowerbound(), "  current upperbound:", root_node.get_upperbound(), "  number of iteration:", iteration, " Time ", round(time() - start, 2))
           
        selected = select_node_to_expand(root_node, branching_scheme)
        expand_tree(selected, variable_selection_scheme, size, cap, weight)
        iteration += 1
        if privious > root_node.get_upperbound():
            text = str("current upperbound:"+ str(root_node.get_upperbound())+ "  number of iteration:"+ str(iteration)+ " Time "+ str(round(time() - start, 2))+"\n")
            privious = root_node.get_upperbound()
        if time() - start > time_limit/10:
            print("the algo took to long best solution found by b&b is :", privious, "the best solution found by heuristics is:", heuristic[1])
            break
    with open('benchmark.txt', 'a') as f:
        f.write(text+"iteration per second : " + str(iteration/(time() - start))+"\n")
    print("the algo is done, the objective value is :", root_node.get_upperbound(), " and initial lowerbound is :", ceil(sum(weight)/cap))
        
def get_best_heuristic(size, cap, weight):
    """
    Will select the best solution among all the ones proposed by the heuristics.
    Return: a list with the the solution and its objective value
    ex : best = [sol, obj]
    :param size:
    :param cap:
    :param weight:
    :return:
    """
    heur_list = []

    # sol_best_fit = build_best_fit_solution(size, cap, weight)
    # print("best packing : ", get_obj(sol_best_fit, size, cap, weight, True))
    # heur_list.append([sol_best_fit, get_obj(sol_best_fit, size, cap, weight, True) ])

    # sol_greedy = build_greedy_solution(size, cap, weight)
    # print("greedy packing : ", get_obj(sol_greedy, size, cap, weight, True))
    # heur_list.append([sol_greedy , get_obj(sol_greedy, size, cap, weight, True)])

    sol_evenly_fill = build_evenly_fill_solution(size, cap, weight)
    print("evenly packing : ", get_obj(sol_evenly_fill, size, cap, weight, True))
    heur_list.append([sol_evenly_fill, get_obj(sol_evenly_fill, size, cap, weight, True)])

    # sol_full_packing = build_full_packing_solution(size, cap, weight)
    # print("full packing : ", get_obj(sol_full_packing, size, cap, weight, True))
    # heur_list.append([sol_full_packing, get_obj(sol_full_packing, size, cap, weight, True)])
    
    best = 9999
    elem = []
    for i in range(len(heur_list)):
        if heur_list[i][1] < best: 
            elem = heur_list[i]
            best = heur_list[i][1]
    return elem



def build_root_node(upperbound, file_name, variable_selection_scheme, valid_inequalities, size, cap, weight):
    """

    :param upperbound:
    :param file_name:
    :param variable_selection_scheme:
    :return:
    """
    calculator = Calculator(file_name)
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

    :param node:
    :return:
    """
    """
    Select the left most node to expand if it has not been visited yet.
    """
    childs = node.get_childs()
    if len(childs):
        for i in range(len(childs)):
            if not childs[i].get_is_done():
                return select_node_to_expand_depth_first(childs[i])
    return node

def select_node_to_expand_best_first(node):
    """

    :param node:
    :return:
    """
    """
    Select the best node such that the node with the lowest upperbound 
    will be expanded if it has not been visited yet.
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

def cutting_planes_2(size, cap, weight):
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
    """"
    
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
                
def expand_tree(node, variable_selection_scheme, size, cap, weight):
    """

    :param node:
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
            calculator = Calculator(file_name)
            calculator.add_constraint_model(constraints)
            calculator.add_cutting_planes(node.get_cutting_planes())
            calculator.run()
            lowerbound = ceil(calculator.get_objective())
            if lowerbound and lowerbound <= node.get_root().get_upperbound():
                # print("im feasible, and my depth is : ", node.get_depth()+1)
                non_int = calculator.get_non_int(variable_selection_scheme)
                upperbound = None
                is_done = False
                if calculator.checkFinishedProduct():
                    upperbound = deepcopy(calculator.get_int_objective(size))
                    is_done = True
                    print("current solution found with upperbound value :", upperbound, "the lowerbound is :", lowerbound, "my depth =", node.get_depth()+1)  
                else :
                    upperbound = deepcopy(calculator.compute_int_solution(size, cap, weight))
                    #print("rebuilded solution with upperbound value : ", upperbound, "my depth =", node.get_depth()+1)
                node.add_child(Node(deepcopy(constraints), deepcopy(upperbound), deepcopy(lowerbound), node, node.get_root(), deepcopy(non_int), node.get_cutting_planes(),deepcopy(node.get_depth()+1) ,deepcopy(is_done)))
            else:
                if lowerbound > node.get_root().get_upperbound():
                    print("The solution has a lowerbound that is higher to the upperbound.")
                else :
                    print("not feasible solution")  

        if len(node.get_childs()) == 0: #Case where none of the childs are possible. 
            node.set_is_done(True)
            if node.get_parent() is not None:
                update_parent(node.get_parent())
            print("I am indeed done")    
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
    update_bounds(node)
    update_state(node)
            

def check_valid_sol(solution, size, cap, weight):
    """
    Will return true if the solution build is valid, false otherwise.
    """
    for col in range(size):
        value = 0
        for row in range(size):
            value += solution[row][col] * weight[row]
        if value > cap:
            return False

    for row in solution:
        if sum(row)<1:
            return False

    return True


def build_greedy_solution(size, cap, weight):
    """
    Greedy heuristic algorithm that will fill the first bag that can accept an object. 
    It will return the first solution for the root node. 
    As the objects are already sorted by weights this algorithm is equivalent to "First-Fit decreasing". 
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




# file_name = "Instances/bin_pack_50_1.dat"
beg = 50
end = 150
for a in range(3):
    with open('benchmark.txt', 'a') as f:
        f.write("sel"+str(a+1)+"\n")
    for i in range(beg, end+5, 50):
        for j in range(1,2):
            file_name = "Instances/bin_pack_" + str(i) + "_" + str(j) + ".dat"
            with open('benchmark.txt', 'a') as f:
                f.write(file_name+"\n")
            branch_and_bound(file_name, BRANCH["BEST_FIRST"], a, valid_inequalities=1)

