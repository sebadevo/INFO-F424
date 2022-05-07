import pyomo.environ as pyo
import numpy as np

TIME_LIMIT = 10


class Calculator:
    def __init__(self, filename):  # elements
        """

        :param filename:
        """

        self.filename = filename
        self.model = pyo.AbstractModel()

        """ 2. DÉCLARATION DES CONSTANTES ET DES LISTES """

        self.model.I = pyo.Set()
        self.model.cap = pyo.Param()
        self.model.size = pyo.Param(self.model.I, within=pyo.NonNegativeIntegers)

        self.model.p = pyo.Set(initialize=self.model.I)
        self.model.b = pyo.Set(initialize=self.model.I)

        """ 3. DÉCLARATION DES VARIABLES """

        self.model.y = pyo.Var(self.model.b, domain=pyo.NonNegativeReals, initialize=0,
                               bounds=(0, 1))  # 1 if box b is used
        self.model.x = pyo.Var(self.model.p, self.model.b, domain=pyo.NonNegativeReals, initialize=0,
                               bounds=(0, 1))  # 1 if product p is in box b

        """ 4. DÉCLARATION DE LA FONCTION OBJECTIVE """

        @self.model.Objective()
        def obj_expression(m):
            return pyo.summation(m.y)

        """ 5. DÉCLARATION DES CONTRAINTES """

        @self.model.Constraint(self.model.b)
        def xcy_constraint_rule(m, b):
            return sum(m.x[p, b] * m.size[p] for p in m.p) <= m.cap * m.y[b]

        @self.model.Constraint(self.model.p)
        def x_constraint_rule(m, p):
            return sum(m.x[p, b] for b in m.b) == 1

        """ 6. PARAMÊTRE DU SOLVEUR """

        self.solveur = pyo.SolverFactory('glpk')
        self.solveur.options['tmlim'] = TIME_LIMIT

        """ 7. RÉCUPÉRATION DES DATAS """

        self.data = pyo.DataPortal(model=self.model)
        self.data.load(filename=self.filename, model=self.model)
        self.instance = self.model.create_instance(self.data)
        self.instance.constraint_list = pyo.ConstraintList()

    def run(self):
        """ 8. LANCEMENT DU SOLVEUR """
        result = self.solveur.solve(self.instance)  # , tee=True).write()

    def affichage_result(self):
        """

        """
        """ 9. RÉCUPÉRATION DES RÉSULTATS """

        # instance.display() # usefull command to show to full result but quite heavy in the terminal

        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) != 0:
                print(self.instance.x[j], " of value ", pyo.value(self.instance.x[j]))

        for i in self.instance.y:
            if pyo.value(self.instance.y[i]) != 0:
                print(self.instance.y[i], " of value ", pyo.value(self.instance.y[i]))

        print(pyo.value(self.instance.obj_expression))

    def affichage_working_progress(self):
        """ 9. RÉCUPÉRATION DES RÉSULTATS """
        somme = 0
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) != 1 and pyo.value(self.instance.x[j]) != 0:
                somme += pyo.value(self.instance.x[j])
                # print(self.instance.x[j], " of value ", pyo.value(self.instance.x[j]))
        print("Il reste encore à résoudre: ", somme)
        print(pyo.value(self.instance.obj_expression))

    def add_constraint(self, corrected):
        for elem in corrected:
            self.instance.x[elem[0]].fix(elem[1])

    def add_constraint_model(self, corrected):
        """

        :param corrected:
        :return:
        """
        for elem in corrected:
            if elem[1]:
                self.instance.constraint_list.add(self.instance.x[elem[0]] >= elem[1])
            else:
                self.instance.constraint_list.add(self.instance.x[elem[0]] <= elem[1])


    def get_non_int(self, variable_selection_scheme):
        """
        It will return the variable that will be branched on. To do so it first gathers all fractional value in the
        solution computed by the solver (i.e. GLPK). The position of the fractional values and their actual value are
        stored in the list : list_non_int. Once we gathered them all, we send this list to the compute dist methode
        which will return the variable selected to branch.

        :param variable_selection_scheme: (Int) The scheme used to select the variable (cf. compute_dist())
        :return: (List) [pos, value] the position of the variable and the integer it was closest to.
        """
        list_non_int = []
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) < 0.99 and pyo.value(self.instance.x[j]) > 0.01:
                pos = j
                value = pyo.value(self.instance.x[j])
                list_non_int.append([pos, value])
        return self.compute_dist(list_non_int, variable_selection_scheme)

    def compute_dist(self, all_frac, variable_selection_scheme):
        """
        It will select the variable x[i,j] that will be branched. To select it, a distance is computed (depending on the
        variable_selection_scheme) for each fractionnal vraible and the one with the minimal distance is selected.


        :param all_frac: (List) of all fractional value.

        Example of all_frac:
            - all_frac = [elem_1, elem_2, ..., elem_n].
            - elem_i = [pos, value] for all i = 1, ..., n.
            - pos = (x,y) tuple of coordinates of the fractional value in the solution matrix.
            - value = value of the fractional value at the given position.

        :param variable_selection_scheme: (Int) The scheme used.

        The different available schemes are
            - 0 -> variable whose fractional value is closest to int (either 1 or 0).
            - 1 -> variable whose fractional value is closest to 1/2.
            - 2 -> variable whose fractional value is closest to 1.
        By testing, the one who works best is variable_selection_scheme=2.

        :return: (List) [pos, value] the position of the variable and the integer it was closest to.
        """
        best = 2
        coord = []
        for i in all_frac:
            if variable_selection_scheme == 0:
                dist = 0.5 - abs(i[1] - 0.5)
            elif variable_selection_scheme == 1:
                dist = abs(i[1] - 0.5)
            elif variable_selection_scheme == 2:
                dist = abs(i[1] - 1)
            if dist < best:
                best = dist
                coord = i
        if len(coord):
            coord[1] = round(coord[1])  # If the value is = 1/2 the chance of it being a 1 or a 0 is equiprobable.
        return coord

    def get_solution(self, size):
        solution = np.zeros((size,size))
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]):
                solution[j] = 1
        return solution

    def getAllNonInt(self):
        """
        Will be removed because useless in the end.
        :return:
        """
        list_non_int = []
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) != 1 and pyo.value(self.instance.x[j]) != 0:
                pos = j
                value = pyo.value(self.instance.x[j])
                list_non_int.append([pos, value])
        return list_non_int

    def get_one_values(self):
        """
        Will gather in a list (list_int) all the variable of the solution which value is equal to 1.

        Example of the list :
            - list_int = [pos_1, pos_2, ..., pos_n]
            - pos_i = (x,y) tuple of coordinates of the fractional value in the solution matrix.
        :return: (List) list_int the position of the variables which equal 1.
        """
        list_int = []
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) == 1:
                pos = j
                list_int.append(pos)
        return list_int

    def compute_int_solution(self, size, cap, weight, scheme=0):
        """

        :param size:
        :param cap:
        :param weight:
        :param scheme:
        :return:
        """
        fixed_values = self.get_one_values()
        bag = [cap for i in range(size)]
        solution = np.zeros((size, size))
        obj = [i for i in range(size)]

        for pos in fixed_values:
            solution[pos] = 1
            bag[pos[1]] -= weight[pos[0]]
            obj.remove(pos[0])

        up = 0

        if scheme == 0:
            up = self.rebuild_best_fit(size, bag, solution, obj, weight)
        elif scheme == 1:
            up = self.rebuild_first_fit(size, bag, solution, obj, weight)
        return up

    def rebuild_best_fit(self, size, bag, solution, obj, weight):
        """

        :param size:
        :param bag:
        :param solution:
        :param obj:
        :param weight:
        :return:
        """
        for o in obj:
            rest = 1000
            index = 0
            w = -1
            for sac in range(size):
                if bag[sac] >= weight[o] and bag[sac] - weight[o] < rest:
                    rest = bag[sac] - weight[o]
                    w = weight[o]
                    index = sac
            if w != -1:
                solution[o][index] = 1
                bag[index] -= weight[o]
            else:
                print("solution impossible to build")
        used = sum(solution)
        up = 0
        for i in used:
            if i > 0:
                up += 1
        return up

    def rebuild_first_fit(self, size, bag, solution, obj, weight):
        """

        :param size:
        :param bag:
        :param solution:
        :param obj:
        :param weight:
        :return:
        """
        for o in obj:
            for sac in range(size):
                if bag[sac] >= weight[o]:
                    solution[o][sac] = 1
                    bag[sac] -= weight[o]
                    break

        used = sum(solution)
        up = 0
        for i in used:
            if i > 0:
                up += 1
        return up

    def get_objective(self):
        """

        :return:
        """
        return pyo.value(self.instance.obj_expression)
    
    def get_int_objective(self, size):
        solution = np.zeros((size,size))
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]):
                solution[j] = 1
        return np.count_nonzero(sum(solution)==0) 

    def getReduced(self):
        """
        useless
        :return:
        """
        temp = []
        for j in self.instance.x:
            if 0 < pyo.value(self.instance.x[j]) < 0.1:
                pos = j
                value = pyo.value(self.instance.x[j])
                temp.append([pos, value])
        return temp

    def checkFinishedProduct(self):
        """

        :return:
        """
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) != 0 and pyo.value(self.instance.x[j]) != 1:
                return False
        return True

    def checkFinishedBox(self):
        """

        :return:
        """
        for i in self.instance.y:
            if pyo.value(self.instance.y[i]) != 0 and pyo.value(self.instance.y[i]) != 1:
                return False
        return True
