import numpy as np
import pandas as pd

class DiscreteDistribution:
    def __init__(self,eventChoices,name):
        self.eventChoices=eventChoices
        self.name=name
    
    def getProbability(self,queryChoice):
        return self.eventChoices.get(queryChoice)


class ConditionalProbabilityTable:
    def __init__(self,conditions,dependencyList,name):
        self.dependencyList=dependencyList
        self.name=name
        columns=[distribution.name for distribution in self.dependencyList]+[self.name,"probability"]
        self.conditions=pd.DataFrame(conditions,columns=columns)
        self.jointProbability=self.conditions.copy()
    #creating joint probability distribution
    def joint(self):
        #P(Monty,Prize,Guest)=P(Monty|Prize,Guest)*P(Prize)*P(Guest)
        for node in self.dependencyList:
            self.jointProbability['probability']=np.multiply(np.array(self.jointProbability['probability']),\
                np.array([node.randomVariable.eventChoices.get(choice) for choice in self.jointProbability[node.name]]))


