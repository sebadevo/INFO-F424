import pyomo.environ as pyo

file_name = "Instances/bin_pack_110_2.dat"
TIME_LIMIT = 10

""" 1. CRÉATION DU MODÈLE """

model = pyo.AbstractModel()

""" 2. DÉCLARATION DES CONSTANTES ET DES LISTES """

model.I = pyo.Set()
model.cap = pyo.Param()
model.size = pyo.Param(model.I, within=pyo.NonNegativeIntegers)

model.p = pyo.Set(initialize=model.I)
model.b = pyo.Set(initialize=model.I)

""" 3. DÉCLARATION DES VARIABLES """

model.y = pyo.Var(model.b, domain=pyo.NonNegativeReals, bounds=(0, 1))  # 1 if box b is used
model.x = pyo.Var(model.p, model.b, domain=pyo.NonNegativeReals, bounds=(0, 1))  # 1 if product p is in box b

""" 4. DÉCLARATION DE LA FONCTION OBJECTIVE """


def obj_expression(m):
    return pyo.summation(m.y)


model.OBJ = pyo.Objective(rule=obj_expression)

""" 5. DÉCLARATION DES CONTRAINTES """


def xcy_constraint_rule(m, b):
    return sum(m.x[p, b] * m.size[p] for p in m.p) <= m.cap * m.y[b]


model.XcyConstraint = pyo.Constraint(model.b, rule=xcy_constraint_rule)


def x_constraint_rule(m, p):
    return sum(m.x[p, b] for b in m.b) == 1


model.XConstraint = pyo.Constraint(model.p, rule=x_constraint_rule)

""" 6. PARAMÊTRE DU SOLVEUR """

solveur = pyo.SolverFactory('glpk')
solveur.options['tmlim'] = TIME_LIMIT

""" 7. RÉCUPÉRATION DES DATAS """

data = pyo.DataPortal(model=model)
data.load(filename=file_name, model=model)
instance = model.create_instance(data)

""" 8. LANCEMENT DU SOLVEUR """

result = solveur.solve(instance, tee=True).write()

""" 9. RÉCUPÉRATION DES RÉSULTATS """

# instance.display() # usefull command to show to full result but quite heavy in the terminal
somme = 0
for j in instance.x:
    if pyo.value(instance.x[j]) > 0.001:
        somme += pyo.value(instance.x[j])
        print(instance.x[j], " of value ", pyo.value(instance.x[j]))
print("valeur des partie fractionnaire : ", round(somme))
print(pyo.value(instance.OBJ))
