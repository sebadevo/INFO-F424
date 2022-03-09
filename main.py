import pyomo.environ as pyo

file_name = "Instances/bin_pack_50_0.dat"

""" 1. CRÉATION DU MODÈLE """


model = pyo.AbstractModel()


""" 2. DÉCLARATION DES CONSTANTES, DES LISTES ET DICTIONNAIRES """


model.I = pyo.Set()
model.cap = pyo.Param()
model.size = pyo.Param(model.I, within=pyo.NonNegativeIntegers)

model.p = pyo.Set(initialize=model.I)
model.b = pyo.Set(initialize=model.I)

time = 10  # number of minutes the program can run


""" 3. DÉCLARATION DES VARIABLES """


model.y = pyo.Var(model.b, domain=pyo.NonNegativeReals)  # 1 if box b is used
model.x = pyo.Var(model.p, model.b, domain=pyo.NonNegativeReals)  # 1 if product p is in box b


""" 4. DÉCLARATION DE LA FONCTION OBJECTIVE """


def obj_expression(m):
    return pyo.summation(m.y)


model.OBJ = pyo.Objective(rule=obj_expression)


""" 4. DÉCLARATION DES CONTRAINTES """


def xcy_constraint_rule(m, b):
    return sum(m.x[p, b] * m.size[p] for p in m.p) <= m.cap * m.y[b]


model.XcyConstraint = pyo.Constraint(model.b, rule=xcy_constraint_rule)


def x_constraint_rule(m, p):
    return sum(m.x[p, b] for b in m.b) == 1


model.XConstraint = pyo.Constraint(model.p, rule=x_constraint_rule)


""" 6. RÉCUPÉRATION DES RÉSULTATS """

# data = pyo.DataPortal(model=model)
# data.load(filname=file_name, model=model)
# instance = model.create_instance(data)
# instance.solve()
# pyo.value(instance.obj)
