import pyomo.environ as pyo
from calculator import Calculator

file_name = "Instances/bin_pack_50_0.dat"
corrected = []
calculator = Calculator(file_name)

calculator.run()
calculator.affichage_result()
bags = []
i = 0
while not calculator.checkFinishedProduct():
    temp = calculator.getReduced()
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
    calculator.add_constraint(temp)
    calculator.run()
    calculator.affichage_result()
    i+=1
    if (i % 100 == 1):
        print("I AM DONE, BABY", i)

