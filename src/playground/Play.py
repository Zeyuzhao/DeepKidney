# Weird Issue,

from pyscipopt import Model
import networkx as nx


comp_graph = nx.fast_gnp_random_graph(5, 0.2, directed=True)

model = Model("Graph")  # model name is optional

x = model.addVar("x")

model.setObjective(x + y)
model.addCons(2*x - y*y >= 0)
model.optimize()
sol = model.getBestSol()
print("x: {}".format(sol[x]))
print("y: {}".format(sol[y]))
model.freeTransform()
