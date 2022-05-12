from math import ceil
import numpy as np

file_name = "Instances/bin_pack_150_0.dat"

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
        if val > 150:
            sol = np.zeros(size+2, dtype=int)
            b = 0
            for elem in selected:
                sol[elem] = weight[elem]
                b += weight[elem]
            c = cap
            k = ceil(b/c)
            g = b-(k-1)*c
            sol[-2] = g             # constraint is x <= b - g*(k-y)
            sol[-1] = b-g*k
            constraint.append(np.rot90(np.tile(sol,(size,1)), -1))
    print(len(constraint))


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

size, cap, weight = extract_data(file_name)
cutting_planes_2(size, cap, weight)