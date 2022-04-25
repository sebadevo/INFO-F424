import pyomo.environ as pyo

TIME_LIMIT = 10

class Calculator:
    def __init__(self, filename): #elements
        self.filename = filename
        self.model = pyo.AbstractModel()


        """ 2. DÉCLARATION DES CONSTANTES ET DES LISTES """

        self.model.I = pyo.Set()
        self.model.cap = pyo.Param()
        self.model.size = pyo.Param(self.model.I, within=pyo.NonNegativeIntegers)

        self.model.p = pyo.Set(initialize=self.model.I)
        self.model.b = pyo.Set(initialize=self.model.I)
        

        """ 3. DÉCLARATION DES VARIABLES """

        self.model.y = pyo.Var(self.model.b, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, 1))  # 1 if box b is used
        self.model.x = pyo.Var(self.model.p, self.model.b, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, 1))  # 1 if product p is in box b 

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
        result = self.solveur.solve(self.instance)#, tee=True).write()

    
    
    def affichage_result(self):
        """ 9. RÉCUPÉRATION DES RÉSULTATS """
        # instance.display() # usefull command to show to full result but quite heavy in the terminal
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) != 0:
                print(self.instance.x[j], " of value ", pyo.value(self.instance.x[j]))
        print(pyo.value(self.instance.obj_expression))

    def affichage_working_progress(self):
        """ 9. RÉCUPÉRATION DES RÉSULTATS """
        # instance.display() # usefull command to show to full result but quite heavy in the terminal
        somme = 0
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) != 1 and pyo.value(self.instance.x[j]) != 0:
                somme += pyo.value(self.instance.x[j])
                #print(self.instance.x[j], " of value ", pyo.value(self.instance.x[j]))
        print("Il reste encore à résoudre: ", somme)
        print(pyo.value(self.instance.obj_expression))

    
    def add_constraint(self, corrected):
        for elem in corrected:
            self.instance.x[elem[0]].fix(elem[1])
            self.instance.Constraint

    def add_constraint_model(self, corrected):
        for elem in corrected:
            if elem[1]:
                self.instance.constraint_list.add(self.instance.x[elem[0]]>=elem[1])
            else:
                self.instance.constraint_list.add(self.instance.x[elem[0]]<=elem[1])


    def getNonInt(self):
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) < 1 and pyo.value(self.instance.x[j]) > 0.001:
                pos = j
                value = pyo.value(self.instance.x[j])
                return pos, value

    def getAllNonInt(self):
        list_non_int = []
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) != 1 and pyo.value(self.instance.x[j]) != 0:
                pos = j
                value = pyo.value(self.instance.x[j])
                list_non_int.append([pos, value])
        return list_non_int

    def get_objective(self):
        return pyo.value(self.instance.obj_expression)

    def getReduced(self):
        temp = []
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) > 0 and pyo.value(self.instance.x[j]) < 0.1:
                pos = j
                value = pyo.value(self.instance.x[j])
                temp.append([pos, value])
        return temp

    def checkFinishedProduct(self):
        for j in self.instance.x:
            if pyo.value(self.instance.x[j]) !=0 and pyo.value(self.instance.x[j]) != 1:
                return False
        return True
        
    def checkFinishedBox(self):
        for i in self.instance.y:
            if pyo.value(self.instance.y[i]) !=0 and pyo.value(self.instance.y[i]) != 1:
                return False
        return True
    
        
