from calculator import Calculator
from copy import deepcopy
import numpy as np

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
    size, cap, weight = extract_data(file_name)
    solution = np.zeros((size, size))
    solution = build_init_solution(size, cap, weight, solution)
    computeUC(solution)


def build_init_solution(size, cap, weight, solution):
    capacity = deepcopy(cap)
    solution[0][0] = 1
    capacity -=weight[0]
    j = 0
    for i in range(1, size):
        if capacity-weight[i] >=0:
            solution[i][j] = 1
            capacity -=weight[i]
        else :
            frac = capacity/weight[i]
            solution[i][j] = frac
            j+=1
            capacity = deepcopy(cap)
            frac = 1 - frac
            capacity -= frac*weight[i]
            solution[i][j] = frac
    return solution

def computeUC(solution):
    summary_sol = sum(solution)
    u = 0
    c = 0
    for i in range(len(summary_sol)):
        if summary_sol[i]>0:
            u+=1
        for j in range(len(solution)):
            if solution[i][j] != 1 and solution[i][j] != 0:
                c +=1
                break
    c += u
    return u, c
    
branch_and_bound(file_name)
