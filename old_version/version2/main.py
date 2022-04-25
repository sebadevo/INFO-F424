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
        calculator.add_constraint(corrected)
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


def getNextAvailableBag(bag, size, init=0):
    for i in range(init, size):
        if bag[i] < 1:
            return i


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

file_name = "Instances/bin_pack_20_0.dat"
runpart1()

