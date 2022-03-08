from __future__ import division
from pulp import GLPK
import pyomo.environ as pyo

model = pyo.AbstractModel()
""" 1. DÉCLARATION DES CONSTANTES, DES LISTES ET DICTIONNAIRES """

model.I = pyo.Set(within=pyo.NonNegativeIntegers)
model.size = pyo.Param()
model.cap = pyo.Param()

tmps = 1  # number of minutes

""" 2. CRÉATION DU MODÈLE """

model.p = pyo.RangeSet(1, model.I)
model.b = pyo.RangeSet(1, model.I)

# model = LpProblem(name="Bin Packing problem", sense=LpMinimize)

""" 3. DÉCLARATION DES VARIABLES """

model.y = pyo.Var(model.b, domain=pyo.Binary)  # 1 if box b is used
model.x = pyo.Var(model.p, model.b, domain=pyo.Binary)  # 1 if product p is in box b


# Objective function definition

def obj_expression(m):
    return pyo.summation(m.y)


model.OBJ = pyo.Objective(rule=obj_expression)


def xcy_constraint_rule(m, b):
    return sum(m.x[p, b] for p in m.p) <= m.cap * m.y[b]


model.XcyConstraint = pyo.Constraint(model.b, rule=xcy_constraint_rule)


def x_constraint_rule(m, p):
    return sum(m.x[p, b] for b in m.b) == 1


model.XConstraint = pyo.Constraint(model.p, rule=x_constraint_rule)


""" 3. DÉCLARATION DES VARIABLES """
# x = LpVariable.dicts("BagFiledWith", [(p, b) for p in range(1, P + 1) for b in range(1, B + 1)], 0, 1, cat=LpBinary)

# y = LpVariable.dicts("BagFiledWith", [p for p in range(1, P + 1)], 0, 1, cat=LpBinary)

""" 4. DÉCLARATION DES CONTRAINTES """

""" 5. DÉCLARATION DE LA FONCTION OBJECTIVE """

# status = model.solve(solver=GLPK(msg=True, keepFiles=True, options=["--tmlim", str(60 * tmps)]))
""" 6. RÉCUPÉRATION DES RÉSULTATS """
