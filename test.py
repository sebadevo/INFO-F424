import numpy as np



def cutting_planes(sol, size, cap, weight):
    """"
    
    """
    solution = sol #node.get_relaxed_solution(size)
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

    prod_constraint = np.vstack((prod_constraint, bag_constraint))
    print(prod_constraint)
    return prod_constraint

sol = np.array([[1, 0, 0],
                [0.3, 0.5, 0.2],
                [0.7, 0.3, 0]])
weight = np.array([77, 40, 30])
print(cutting_planes(sol, 3, 150, weight)[-1][1])

