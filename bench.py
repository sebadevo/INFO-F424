from copy import deepcopy
from math import ceil
import numpy as np

bench_file = "benchmark_heuristic.txt"
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

    sol_best_fit = build_best_fit_solution(size, cap, weight)
    print("best packing : ", get_obj(sol_best_fit, size, cap, weight, True))
    heur_list.append([sol_best_fit, get_obj(sol_best_fit, size, cap, weight, True) ])

    sol_greedy = build_greedy_solution(size, cap, weight)
    print("greedy packing : ", get_obj(sol_greedy, size, cap, weight, True))
    heur_list.append([sol_greedy, get_obj(sol_greedy, size, cap, weight, True)])

    sol_evenly_fill = build_evenly_fill_solution(size, cap, weight)
    print("evenly packing : ", get_obj(sol_evenly_fill, size, cap, weight, True))
    heur_list.append([sol_evenly_fill, get_obj(sol_evenly_fill, size, cap, weight, True)])

    sol_full_packing = build_full_packing_solution(size, cap, weight)
    print("full packing : ", get_obj(sol_full_packing, size, cap, weight, True))
    heur_list.append([sol_full_packing, get_obj(sol_full_packing, size, cap, weight, True)])

    with open(bench_file, 'a') as f:
        f.write(","+str(get_obj(sol_best_fit, size, cap, weight, True)-ceil(sum(weight)/cap))+","+str(get_obj(sol_greedy, size, cap, weight, True)-ceil(sum(weight)/cap))+","+str(get_obj(sol_evenly_fill, size, cap, weight, True)-ceil(sum(weight)/cap))+","+str(get_obj(sol_full_packing, size, cap, weight, True)-ceil(sum(weight)/cap))+"\n")
    
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


beg = 20
end = 500
with open(bench_file, 'a') as f:
    f.write("instance_name,best,greedy,evenly,full"+"\n")
for i in range(beg, end+5, 5):
    for j in range(3):
        file_name = "Instances/bin_pack_" + str(i) + "_" + str(j) + ".dat"
        with open(bench_file, 'a') as f:
            f.write(file_name)
        size, cap, weight = extract_data(file_name)
        get_best_heuristic(size, cap, weight)
beg = 155
end = 245
for i in range(beg, end+5, 5):
    for j in range(3):
        file_name = "Instances/bin_pack2_" + str(i) + "_" + str(j) + ".dat"
        with open(bench_file, 'a') as f:
            f.write(file_name)
        size, cap, weight = extract_data(file_name)
        get_best_heuristic(size, cap, weight)