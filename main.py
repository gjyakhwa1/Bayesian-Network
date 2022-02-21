from node import Node
from bayesianNetwork import BayesianNetwork
from distribution import DiscreteDistribution,ConditionalProbabilityTable
"""
            guest   prize
                monty
"""

if __name__=="__main__":
    model=BayesianNetwork()
    guest=Node(DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3},name="Guest"))
    prize=Node(DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3},name="Prize"))
    monty=Node(ConditionalProbabilityTable(
        [['A', 'A', 'A', 0.0],
         ['A', 'A', 'B', 0.5],
         ['A', 'A', 'C', 0.5],
         ['A', 'B', 'A', 0.0],
         ['A', 'B', 'B', 0.0],
         ['A', 'B', 'C', 1.0],
         ['A', 'C', 'A', 0.0],
         ['A', 'C', 'B', 1.0],
         ['A', 'C', 'C', 0.0],
         ['B', 'A', 'A', 0.0],
         ['B', 'A', 'B', 0.0],
         ['B', 'A', 'C', 1.0],
         ['B', 'B', 'A', 0.5],
         ['B', 'B', 'B', 0.0],
         ['B', 'B', 'C', 0.5],
         ['B', 'C', 'A', 1.0],
         ['B', 'C', 'B', 0.0],
         ['B', 'C', 'C', 0.0],
         ['C', 'A', 'A', 0.0],
         ['C', 'A', 'B', 1.0],
         ['C', 'A', 'C', 0.0],
         ['C', 'B', 'A', 1.0],
         ['C', 'B', 'B', 0.0],
         ['C', 'B', 'C', 0.0],
         ['C', 'C', 'A', 0.5],
         ['C', 'C', 'B', 0.5],
         ['C', 'C', 'C', 0.0]], [guest,prize],name="Monty"))

    model.addStates(guest,prize,monty)
    model.addEdge(guest,monty)
    model.addEdge(prize,monty)
    condition={
        'Guest':'B',
        'Prize':None,
        'Monty':'B'
        
    }
    # print(model.predictProbability(**condition))
    print(condition)
    print(model.predictProbability(condition))