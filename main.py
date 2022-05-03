from math import ceil
from calculator import Calculator
from copy import deepcopy
import numpy as np
from node import Node


file_name = "Instances/bin_pack2_245_0.dat"

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
    prox = heuristic[1]-ceil(sum(weight)/cap)
    print("heuristic :",heuristic[1], " LB : ",ceil(sum(weight)/cap), " proximity to LB:", prox)
    # print("is the heuristic's solution valid ? ", check_valid_sol(heuristic[0], size, cap, weight))
    if heuristic[1]-ceil(sum(weight)/cap) == 0:
        return 1
    print(instance_name)
    return 0
    if branching_scheme == 0:
        #depth_first(heuristic[1], instance_name)
        # leastCost(size, cap, weight)
        pass
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

def compute_dist(fracs):
    best = 2
    coord = None
    for i in fracs: 
        dist = 0.5 - abs(i[1]-0.5)
        if dist < best: 
            best = dist
            coord = i
    coord[1] = round(coord[1])
    return coord


def depth_first(upperbound, file_name):
    """
    """
    node_list = []
    calculator = Calculator(file_name)
    calculator.run()
    root_node = Node(None, upperbound, calculator.get_objective(), None)
    node_list.append(root_node)
    while root_node.get_lowerbound != root_node.get_upperbound:
        selected = select_node_to_expand(node_list)
        new_nodes = expand_tree_depth_first(selected, calculator)
        node_list.extend(new_nodes)
    


    
def select_node_to_expand(nodes):
    return nodes[-1]
                
    

def expand_tree_depth_first(node, calculator):
    all_non_int = calculator.getAllNonInt()
    constr = compute_dist(all_non_int) #return a list with in first pos the value, and sec. pos the pos
    constraints = node.getConstraints()
    constraints.append(constr)
    calculator = Calculator(file_name)
    calculator.add_constraint_model(constraints)
    calculator.run()
    lowerbound = calculator.get_objective()
    upperbound = None
    if calculator.checkFinishedProduct():
        upperbound = lowerbound
    child = Node(constraints, upperbound, lowerbound, node)
    node.add_child(child)


    
# def leastCost(size, cap, weight):
#     nodesToExpand = []
#     empty = np.zeros((size, size))
#     grid = build_solution(size, cap, weight, empty, 0)
#     upperbound, cost = computeUC(grid)
#     node = Node(grid, upperbound, cost, 0)
#     nodesToExpand.append(node)
#     upper = deepcopy(upperbound)
#     running = True
#     solution = None
#     i = 0
#     while running:
#         i += 1
#         selected = selectNodeToExpand(nodesToExpand)
#         if i % 10 == 1:
#             print("I am here: ", selected.getRow(), "  current upper value :", upper,
#                   "  and cost value (lowerbound) is :", cost, "  and the length of the list of nodes is : ",
#                   len(nodesToExpand))
#         if selected.getRow() == size or selected.getUpperbound() == cost:
#             solution = selected
#             running = False
#             break
#         nodesToExpand.remove(selected)
#         if selected.getRow() < size:
#             newExpandedNodes = expandTreeLC(selected, size, cap, weight)
#             upperbound = getBestUpper(newExpandedNodes)

#             if upperbound < upper:
#                 upper = upperbound
#                 nodesToExpand.extend(newExpandedNodes)  # On ajoute toutes les nodes et puis on fait le filtre
#                 nodesToExpand = removeBadNodes(nodesToExpand, upper)
#             else:
#                 newExpandedNodes = removeBadNodes(newExpandedNodes, upper)
#                 nodesToExpand.extend(newExpandedNodes)

#     print("value : ", solution.getCost(), "  lower bound :", cost)


def removeBadNodes(nodes, upper):
    length = len(nodes)
    i = 0
    while i < length:
        if nodes[i].getUpperbound() > upper + 3:
            nodes.remove(nodes[i])
            i -= 1
            length -= 1
        i += 1
    return nodes


def getBestUpper(nodes):
    upper = 9999
    for node in nodes:
        if node.getUpperbound() < upper:
            upper = node.getUpperbound()
    return upper



# def expandTreeLC(node, size, cap, weight):
#     init = node.getSolution()
#     row = node.getRow()
#     new_nodes = []
#     for i in range(size):
#         new_sol = deepcopy(init)
#         new_sol[row] = np.zeros(size)
#         new_sol[row][i] = 1
#         for j in range(row + 1, size):
#             new_sol[j] = np.zeros(size)
#         if check_valid_sol(new_sol, size, cap, weight, row + 1):
#             new_sol = build_solution(size, cap, weight, new_sol, row + 1)
#             upperbound, cost = computeUC(new_sol)
#             new_node = Node(new_sol, upperbound, cost, row + 1)
#             new_nodes.append(new_node)
#     return new_nodes


def check_valid_sol(solution, size, cap, weight):
    for col in range(size):
        value = 0
        for row in range(size):
            value += solution[row][col] * weight[row]
        # if value:
        #     print(value)
        if value > cap:
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
        for sac in range(size):
            if bag[sac] >= weight[obj] and bag[sac]-weight[obj]<rest:
                rest = bag[sac] - weight[obj]
                index = sac
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
print("percentage of completion : ", percent, round((end-beg)*3/5), str(round(percent*100/((end-beg)*3/5)))+"%") #*100/((155-20)*3/5)

# calculator = Calculator(file_name)

# calculator.run()
# calculator.affichage_result()