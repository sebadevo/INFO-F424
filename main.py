from calculator import Calculator
from copy import deepcopy
import numpy as np
from node import Node

file_name = "Instances/bin_pack_20_0.dat"


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
            value =0
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
        calculator.affichage_result()
        i+=1
        if (i % 100 == 1):
            print("I AM DONE, BABY", i)


def extract_data(file_name):
    empty_list = []
    with open(file_name) as datFile:
        for data in datFile:
            empty_list.append(data.split())
            
    size = int(empty_list[0][-1].split(';')[0]) + 1
    cap =  int(empty_list[2][-1].split(';')[0])
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
    nodesToExpand= []
    empty = np.zeros((size, size))
    solution = build_solution(size, cap, weight, empty, 0)
    upperbound, cost = computeUC(solution)
    # node = Node(solution, upperbound, cost, 0)
    # nodesToExpand.append(node)
    # upper = deepcopy(upperbound)

    # boucle while a partir d'ici. !!!!!!!!!!!!!!!!!

    # selected = selectNodeToExpand(nodesToExpand) #ici , juste après on vérifie si row de cette node = size, si c'est le cas alors on a finit l'aglo et cette node est notre solution.
    # nodesToExpand.remove(selected)
    # newExpandedNodes = expandTreeLC(selected, size, cap, weight)

    # il reste a ajouter la verification des costs, si ils sont plus élevé que le que upper, alors on la retire des nodes à explorer
    # creer la fonction get best upper, si elle renvoie un upper inférieur à celui, qu'on a déjà alors on ajoute les new nodes
    # à la liste des newExpanded nodes et on fait un nettoyage des badnodes sur tout le monde avec le nouveau upper, sinon on fait juste un nettoyage sur les 
    # les nouvelles nodes a explorer qui vont être ajouter (donc sur newExpandedNodes).

    # après ça il faut juste vérifier si on a finit l'algo ou pas donc verifier si la node qui à le cost le plus bas est celle qui est au feuille de notre arbre.
    # (row == size if true) faire ça au moment ou on selection selectNodeToExpand.

    # nodesToExpand.extend(newExpandedNodes)

    # notFinished = False
    # while notFinished:
    #     pass
    print(solution)
    print("the cost of this node is : ", cost)
    print("and it's upperbound is : ", upperbound)

def selectNodeToExpand(nodesToExpand):
    cost = 999
    selected = None
    for node in nodesToExpand:
        if node.getCost < cost:
            cost = node.getCost
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
        for j in range(row+1,size):
            new_sol[j] = np.zeros(size)
        if check_valid_sol(new_sol, size, cap, weight):
            new_sol = build_solution(size, cap, weight, new_sol, row+1)
            upperbound, cost = computeUC(new_sol)
            new_node = Node(new_sol, upperbound, cost, row+1)
            new_nodes.append(new_node)
    return new_nodes

def check_valid_sol(solution, size, cap, weight):
    for col in range(size):
        value = 0
        for row in range(size):
            value += solution[row][col]*weight[row]
        if value > cap:
            return False
    return True


def build_solution(size, cap, weight, solution, row=0):
    bag = sum(solution)
    j = getNextAvailableBag(bag, size)
    capacity = (1-bag[j])*cap
    for i in range(row, size):
        w = 1*weight[i]
        frac = 0
        while w > 0:
            if capacity-w >=0:
                solution[i][j] = w/weight[i]
                capacity -=w
                bag[j] +=w/cap
                w = 0
            else: 
                frac = capacity/w
                solution[i][j] = frac
                bag[j] = 1
                j = getNextAvailableBag(bag, size)
                capacity = (1-bag[j])*cap
                w = w*(1-frac)
    return solution

def getNextAvailableBag(bag,size, init=0):
    for i in range(init, size):
        if bag[i] < 1:
            return i

def computeUC(solution):
    summary_sol = sum(solution)
    cost = 0
    upperbound = 0
    flag = False
    for i in range(len(summary_sol)):
        if summary_sol[i]>0:
            cost+=1
        for j in range(len(solution)):
            if solution[i][j] != 1 and solution[i][j] != 0:
                upperbound +=1
                flag = True
        if flag:
            upperbound-=1
            flag = False
    upperbound += cost
    return upperbound, cost
    
branch_and_bound(file_name)
