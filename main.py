from calculator import Calculator
from copy import deepcopy
import numpy as np
from node import Node


# file_name = "Instances/bin_pack_55_0.dat"


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
    if branching_scheme == 0:
        leastCost(size, cap, weight)
    else:
        print("this methode has not been built yet :/")


def leastCost(size, cap, weight):
    nodesToExpand = []
    empty = np.zeros((size, size))
    grid = build_solution(size, cap, weight, empty, 0)
    upperbound, cost = computeUC(grid)
    node = Node(grid, upperbound, cost, 0)
    nodesToExpand.append(node)
    upper = deepcopy(upperbound)
    running = True
    solution = None
    i = 0
    while running:
        i += 1
        selected = selectNodeToExpand(nodesToExpand)
        if i % 10 == 1:
            print("I am here: ", selected.getRow(), "  current upper value :", upper,
                  "  and cost value (lowerbound) is :", cost, "  and the length of the list of nodes is : ",
                  len(nodesToExpand))
        if selected.getRow() == size or selected.getUpperbound() == cost:
            solution = selected
            running = False
            break
        nodesToExpand.remove(selected)
        if selected.getRow() < size:
            newExpandedNodes = expandTreeLC(selected, size, cap, weight)
            upperbound = getBestUpper(newExpandedNodes)

            if upperbound < upper:
                upper = upperbound
                nodesToExpand.extend(newExpandedNodes)  # On ajoute toutes les nodes et puis on fait le filtre
                nodesToExpand = removeBadNodes(nodesToExpand, upper)
            else:
                newExpandedNodes = removeBadNodes(newExpandedNodes, upper)
                nodesToExpand.extend(newExpandedNodes)

    print("value : ", solution.getCost(), "  lower bound :", cost)


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


def selectNodeToExpand(nodesToExpand):
    upper = getBestUpper(nodesToExpand)
    row = -1
    selected = None
    for node in nodesToExpand:
        if node.getUpperbound() == upper and row < node.getRow():
            row = node.getRow()
            selected = node
    return selected


def expandTreeLC(node, size, cap, weight):
    init = node.getSolution()
    row = node.getRow()
    new_nodes = []
    for i in range(size):
        new_sol = deepcopy(init)
        new_sol[row] = np.zeros(size)
        new_sol[row][i] = 1
        for j in range(row + 1, size):
            new_sol[j] = np.zeros(size)
        if check_valid_sol(new_sol, size, cap, weight, row + 1):
            new_sol = build_solution(size, cap, weight, new_sol, row + 1)
            upperbound, cost = computeUC(new_sol)
            new_node = Node(new_sol, upperbound, cost, row + 1)
            new_nodes.append(new_node)
    return new_nodes


def check_valid_sol(solution, size, cap, weight, row):
    for col in range(size):
        value = 0
        for rows in range(row):
            value += solution[rows][col] * weight[rows]
        if value > cap:
            return False
    return True


def build_solution(size, cap, weight, solution, row=0):
    bag = np.zeros(size)
    for col in range(size):
        for rows in range(row):
            bag[col] += solution[rows][col] * weight[rows] / cap

    j = getNextAvailableBag(bag, size)
    capacity = (1 - bag[j]) * cap
    for i in range(row, size):
        w = 1 * weight[i]
        frac = 0
        while w > 0:
            if capacity - w >= 0:
                solution[i][j] = w / weight[i]
                capacity -= w
                bag[j] += w / cap
                w = 0
            else:
                frac = capacity / w
                solution[i][j] = frac
                bag[j] = 1
                j = getNextAvailableBag(bag, size)
                capacity = (1 - bag[j]) * cap
                w = w * (1 - frac)
    return solution

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
        for i in range(1, len(weight)-1):
            work_bag = np.array([weight[0][0]], dtype=int)
            index = [0]
            for j in range(i, len(weight)-1):
                if sum(work_bag)+weight[j][0] == cap:
                    work_bag = np.append(work_bag,weight[j][0])
                    index.append(j)
                    break
                elif sum(work_bag)+weight[j][0] < cap:
                    work_bag = np.append(work_bag,weight[j][0])
                    index.append(j)
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

def get_obj(solution, cap, weight, rounded=False):
    bag = np.zeros(size)
    for col in range(size):
        for rows in range(size):
            bag[col] += solution[rows][col] * weight[rows] / cap
    if rounded:
        value = 0
        for i in range(size):
            if bag[i]>0:
                value+=1
        
        value = sum(np.ceil(bag))
    else:
        value = sum(bag)
    return value

def computeUC(solution):
    summary_sol = sum(solution)
    cost = 0
    upperbound = 0
    flag = False
    for i in range(len(summary_sol)):
        if summary_sol[i] > 0:
            cost += 1
        for j in range(len(solution)):
            if solution[i][j] != 1 and solution[i][j] != 0:
                upperbound += 1
                flag = True
        if flag:
            upperbound -= 1
            flag = False
    upperbound += cost
    return upperbound, cost


# doneList = []
# for i in range(150, 155, 5):
#     for j in range(2, 3):
#         file_name = "Instances/bin_pack_" + str(i) + "_" + str(j) + ".dat"
#         branch_and_bound(file_name)
#         doneList.append((i, j))
#         print(doneList)
for i in range(20, 155, 5):
    for j in range(3):
        file_name = "Instances/bin_pack_" + str(i) + "_" + str(j) + ".dat"
        size, cap, weight = extract_data(file_name)
        solution = build_full_packing_solution(size, cap, weight)
        # solution1 = build_greedy_solution(size, cap, weight)
        solution2 = build_best_fit_solution(size, cap, weight)
        solution4 = build_solution(size, cap, weight, np.zeros((size, size)))

        obj = get_obj(solution, cap, weight, True)
        # obj1 = get_obj(solution1, cap, weight, True)
        obj2 = get_obj(solution2, cap, weight, True)
        obj4 = get_obj(solution4, cap, weight)

        print("value of the upperbound of supposed best", obj)
        # print("value of the upperbound of greedy", obj1)
        print("value of the upperbound of bestfit", obj2)
        print("value of the upperbound of relaxation", obj4, "value of lb", sum(weight)/cap)

        print("are solution of greedy and best fit identical ? " + str(i) + "_" + str(j), (solution == solution2).all())


