from logging import RootLogger
from math import ceil
from sre_constants import BRANCH
from calculator import Calculator
from copy import deepcopy
import numpy as np
from node import Node


file_name = "Instances/bin_pack_50_0.dat"

BRANCH = {
    "DEPTH_FIRST" : 0,
    "BREADTH_FIRST" : 1,
    "MIX" : 2
}

VARIABLE = {
    "ROUNDED" : 0,
    "HALF" : 1,
    "FULL" : 2
}



# def runpart1():
#     corrected = []
#     calculator = Calculator(file_name)

#     calculator.run()
#     calculator.affichage_result()
#     bags = []
#     i = 0
#     while not calculator.checkFinishedProduct():
#         pos, value = calculator.getNonInt()
#         if pos[1] in bags:
#             value = 0
#         else:
#             if value < 0.5:
#                 value = 0
#             else:
#                 value = 1
#                 bags.append(pos[1])
#         corrected.append([pos, value])
#         print("SIZE of corrected: ", len(corrected))
#         calculator = Calculator(file_name)
#         calculator.add_constraint_model(corrected)
#         print(corrected)
#         calculator.run()
#         calculator.affichage_working_progress()
#         i += 1
#         if i % 100 == 1:
#             print("I AM DONE, BABY", i)

#     calculator.affichage_result()


def extract_data(filename):
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


def branch_and_bound(instance_name, branching_scheme=0, variable_selection_scheme=0, valid_inequalities=[], time_limit=600):
    size, cap, weight = extract_data(instance_name)
    heuristic = get_best_heuristic(size, cap, weight)
    root_node = build_root_node(heuristic[1], file_name, variable_selection_scheme)
    iteration = 0
    while root_node.get_lowerbound() != root_node.get_upperbound() and not root_node.get_is_done():
        if not iteration % 10:
            print("current lowerbound: ", root_node.get_lowerbound(), "current upperbound: ", root_node.get_upperbound(), "number of iteration: ", iteration)
        selected = select_node_to_expand(root_node, branching_scheme)
        expand_tree(selected, variable_selection_scheme)
        iteration += 1
    print("the algo is done, here is the value of the upperbound : ", root_node.get_upperbound())
        
def get_best_heuristic(size, cap, weight):
    """
    Will select the best solution among all the ones proposed by the heuristics. 
    Return: a list with the the solution and its objective value 
    ex : best = [sol, obj]
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



def build_root_node(upperbound, file_name, variable_selection_scheme):
    calculator = Calculator(file_name)
    calculator.run()
    non_int = calculator.get_non_int(variable_selection_scheme)
    root_node = Node([], upperbound, ceil(calculator.get_objective()), None, None, non_int, 0, False)
    root_node.set_root(root_node)
    return root_node


def select_node_to_expand(node, branching_scheme):
    selected = None

    if branching_scheme == 0:
        selected = select_node_to_expand_depth_first(node)
    elif branching_scheme == 1:
        selected = select_node_to_expand_breadth_first(node)
    else :
        print("Sorry this branching scheme has not been implemented yet.")
        exit(0)
    return selected
    
def select_node_to_expand_depth_first(node):
    """
    Select the left most node to expand if it has not been visited yet.
    """
    childs = node.get_childs()
    if len(childs):
        for i in range(len(childs)):
            if not childs[i].get_is_done():
                return select_node_to_expand_depth_first(childs[i])
    return node

def select_node_to_expand_breadth_first(node):
    """
    TODO
    """
    childs = node.get_childs()
    if len(childs):
        for i in range(len(childs)):    
            if not childs[i].get_is_done() and len(childs[i].get_childs())==0:
                return select_node_to_expand_breadth_first(childs[i])   
        for i in range(len(childs)):
            if not childs[i].get_is_done():
                return select_node_to_expand_breadth_first(childs[i])
    return node

def select_node_to_expand_breadth_first(node, depth):
    """
    TODO
    """
    
    childs = node.get_childs()
    if len(childs) and depth:
        for i in range(len(childs)):    
            if not childs[i].get_is_done() and len(childs[i].get_childs())==0:
                return select_node_to_expand_breadth_first(childs[i])   
        for i in range(len(childs)):
            if not childs[i].get_is_done():
                return select_node_to_expand_breadth_first(childs[i])
    return node


                
def expand_tree(node, variable_selection_scheme):
    if len(node.get_non_int()):
        constr = node.get_non_int() #return a list with in first the position and the value [(1,2), 1], [(1,2), 0]
        for i in range(2):
            constr[1] = abs(constr[1]-i)
            constraints = deepcopy(node.get_constraints())
            constraints.append(constr)
            calculator = Calculator(file_name)
            calculator.add_constraint_model(constraints)
            calculator.run()
            lowerbound = ceil(calculator.get_objective())
            if lowerbound and lowerbound <= node.get_root().get_upperbound():
                print("im feasible, and my depth is : ", node.get_depth()+1)
                non_int = calculator.get_non_int(variable_selection_scheme)
                upperbound = None
                is_done = False
                if calculator.checkFinishedProduct():
                    upperbound = deepcopy(lowerbound)
                    is_done = True
                    print("current solution found with upperbound value : ", upperbound)  
                child = Node(deepcopy(constraints), deepcopy(upperbound), deepcopy(lowerbound), node, node.get_root(), deepcopy(non_int), node.get_depth()+1 ,deepcopy(is_done))
                node.add_child(child)
            else:
                if lowerbound > node.get_root().get_upperbound():
                    print('ich bin bad solution')
                else :
                    print("not feasible solution")  

        if len(node.get_childs()) == 0: #Case where none of the childs are possible. 
            node.set_is_done(True)
            update_parent(node.get_parent())
            print("I am indeed done")    
        else:
            update_parent(node)
        

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
    bag = np.zeros(size)
    for col in range(size):
        for rows in range(size):
            bag[col] += solution[rows][col] * weight[rows] / cap
    if rounded:
        value = 0
        for i in range(size):
            if bag[i]>0:
                value+=1
    else:
        value = sum(bag)
    return value

branch_and_bound(file_name, BRANCH["BREADTH_FIRST"], VARIABLE["HALF"])
