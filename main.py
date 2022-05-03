from math import ceil
from calculator import Calculator
from copy import deepcopy
import numpy as np
from node import Node
# import sys


file_name = "Instances/bin_pack_50_0.dat"

percent = 0

def runpart1():
    corrected = []
    calculator = Calculator(file_name)

    calculator.run()
    calculator.affichage_result()
    bags = []
    i = 0
    while not calculator.checkFinishedProduct():
        pos, value = calculator.getNonInt()
        if pos[1] in bags:
            value = 0
        else:
            if value < 0.5:
                value = 0
            else:
                value = 1
                bags.append(pos[1])
        corrected.append([pos, value])
        print("SIZE of corrected: ", len(corrected))
        calculator = Calculator(file_name)
        calculator.add_constraint_model(corrected)
        print(corrected)
        calculator.run()
        calculator.affichage_working_progress()
        i += 1
        if i % 100 == 1:
            print("I AM DONE, BABY", i)

    calculator.affichage_result()


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


def branch_and_bound(instance_name, branching_scheme=0, valid_inequalities=0, time_limit=600):
    size, cap, weight = extract_data(instance_name)
    heuristic = get_best_heuristic(size, cap, weight)
    if branching_scheme == 0:
        depth_first(heuristic[1], instance_name)
    else:
        print("this methode has not been built yet :/")
        
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

    # sol_evenly_fill = build_evenly_fill_solution(size, cap, weight)
    # print("evenly packing : ", get_obj(sol_evenly_fill, size, cap, weight, True))
    # heur_list.append([sol_evenly_fill, get_obj(sol_evenly_fill, size, cap, weight, True)])

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

def compute_dist(all_frac, methode):
    """
    Can either take the fractionnary value close to an integer value (the closest to 1 or 0), 
    or it can select the value closest to 1/2.
    methode closest to int -> 1
    methode closest to 1/2 -> 2

    all_frac = [elem_1, elem_2, ..., elem_n] 
    elem_1 = [pos, value]
    """
    best = 2
    coord = None
    for i in all_frac:  
        if methode == 1:
            dist = 0.5 - abs(i[1]-0.5)
        elif methode == 2: 
            dist = abs(i[1]-0.5)
        if dist < best: 
            best = dist
            coord = i
    coord[1] = round(coord[1]) #If the value is = 1/2 the chance of it being a 1 or a 0 is equiprobable. 
    return coord


def depth_first(upperbound, file_name):
    """
    """
    calculator = Calculator(file_name)
    calculator.run()
    root_node = Node([], upperbound, ceil(calculator.get_objective()), None, None, False)
    root_node.set_root(root_node)
    iteration = 0
    while root_node.get_lowerbound() != root_node.get_upperbound():
        print("current lowerbound: ", root_node.get_lowerbound(), "current upperbound: ", root_node.get_upperbound(), "number of iteration: ", iteration)
        selected = select_node_to_expand(root_node)
        expand_tree_depth_first(selected, calculator, 2)
        iteration +=1

    
def select_node_to_expand(node):
    """
    """
    childs = node.get_childs()
    if len(childs):
        for i in range(len(childs)):
            if not childs[i].get_is_done():
                return select_node_to_expand(childs[i])
    return node

                
    

def expand_tree_depth_first(node, calculator, mode=1):
    all_non_int = calculator.getAllNonInt()
    if len(all_non_int):
        constr = compute_dist(all_non_int, mode) #return a list with in first the position and the value [(1,2), 1], [(1,2), 0]
        for i in range(2):
            constr[1] = abs(constr[1]-i)
            constraints = node.get_constraints()
            constraints.append(constr)
            calculator = Calculator(file_name)
            calculator.add_constraint_model(constraints)
            calculator.run()
            lowerbound = ceil(calculator.get_objective())
            if lowerbound and lowerbound < node.get_root().get_upperbound():
                upperbound = None
                is_done = False
                if calculator.checkFinishedProduct():
                    upperbound = lowerbound
                    is_done = True
                    print("current solution found with upperbound value : ", upperbound)  
                child = Node(constraints, upperbound, lowerbound, node, node.get_root(), is_done)
                node.add_child(child)
                update_parent(node)
                prune_tree(node.get_root())
            else:
                print("not feasible solution")
        if len(node.get_childs()) == 0: #Case where none of the childs are possible. 
            node.toggle_is_done()

            


def prune_tree(node):
    childs = node.get_childs()
    copy_childs = deepcopy(childs)
    for i in range(len(childs)):
        if childs[i].get_lowerbound() > node.get_root().get_upperbound() or childs[i].get_lowerbound() == childs[i].get_upperbound():
            copy_childs.remove(childs[i])
        else :
            prune_tree(childs[i])
    
    node.set_childs(copy_childs)

    

def update_parent(node):
    childs = node.get_childs()
    lb = 9999
    ub = 9999
    counter = 0
    for i in range(len(childs)):
        if childs[i].get_lowerbound()<lb:
            lb = childs[i].get_lowerbound()
            node.set_lowerbound(lb)
        if childs[i].get_upperbound() is not None and childs[i].get_upperbound() < ub:
            ub = childs[i].get_upperbound()
            node.set_upperbound(ub)  
        if childs[i].get_is_done():
            counter +=1
    
    if counter == len(childs):
        node.toggle_is_done()

    parent = node.get_parent()
    if parent is not None:
        update_parent(parent)
            

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


def getNextAvailableBag(bag, size, init=0):
    for i in range(init, size):
        if bag[i] < 1:
            return i

def get_obj(solution, size, cap, weight, rounded=False):
    bag = np.zeros(size)
    for col in range(size):
        for rows in range(size):
            bag[col] += solution[rows][col] * weight[rows] / cap
    if rounded:
        value = 0
        for i in range(size):
            if bag[i]>0:
                value+=1
        #value = sum(np.ceil(bag))
    else:
        value = sum(bag)
    return value

beg = 20
end = 155
# for i in range(beg, end, 5):
#     for j in range(3):
#         file_name = "Instances/bin_pack_" + str(i) + "_" + str(j) + ".dat"
#         # size, cap, weight = extract_data(file_name)
#         # print(file_name)
#         percent += branch_and_bound(file_name)

branch_and_bound(file_name)
#print("percentage of completion : ", percent, round((end-beg)*3/5), str(round(percent*100/((end-beg)*3/5)))+"%") #*100/((155-20)*3/5)



# calculator = Calculator(file_name)
# calculator.run()
# calculator.affichage_result()