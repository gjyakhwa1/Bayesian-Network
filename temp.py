from node import Node
from bayesianNetwork import BayesianNetwork
from distribution import DiscreteDistribution,ConditionalProbabilityTable
rain=Node(DiscreteDistribution({
    "none": 0.7,
    "light": 0.2,
    "heavy": 0.1
}, name="rain"))
maintenance=Node(ConditionalProbabilityTable([
    ["none", "yes", 0.4],
    ["none", "no", 0.6],
    ["light", "yes", 0.2],
    ["light", "no", 0.8],
    ["heavy", "yes", 0.1],
    ["heavy", "no", 0.9]
], [rain], name="maintenance"))
train=Node(ConditionalProbabilityTable([
    ["none", "yes", "on time", 0.8],
    ["none", "yes", "delayed", 0.2],
    ["none", "no", "on time", 0.9],
    ["none", "no", "delayed", 0.1],
    ["light", "yes", "on time", 0.6],
    ["light", "yes", "delayed", 0.4],
    ["light", "no", "on time", 0.7],
    ["light", "no", "delayed", 0.3],
    ["heavy", "yes", "on time", 0.4],
    ["heavy", "yes", "delayed", 0.6],
    ["heavy", "no", "on time", 0.5],
    ["heavy", "no", "delayed", 0.5],
], [rain, maintenance], name="train"))
appointment=Node(ConditionalProbabilityTable([
    ["on time", "attend", 0.9],
    ["on time", "miss", 0.1],
    ["delayed", "attend", 0.6],
    ["delayed", "miss", 0.4]
], [train], name="appointment"))

model = BayesianNetwork()
model.addStates(rain, maintenance, train, appointment)
model.addEdge(rain, maintenance)
model.addEdge(rain, train)
model.addEdge(maintenance, train)
model.addEdge(train, appointment)
print(maintenance.randomVariable.getJoint())