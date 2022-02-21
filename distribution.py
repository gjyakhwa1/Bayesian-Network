import numpy as np
import pandas as pd

class DiscreteDistribution:
    def __init__(self,eventChoices,name):
        self.eventChoices=eventChoices#evenChoices is a dictionary containing possible choices
        self.name=name
    
    def getProbability(self,queryChoice):
        return self.eventChoices.get(queryChoice)#return the marginal probability for given query


class ConditionalProbabilityTable:
    def __init__(self,conditions,dependencyList,name):
        self.dependencyList=dependencyList#list of dependent Node
        self.name=name
        columns=[distribution.name for distribution in self.dependencyList]+[self.name,"probability"]#adding columns name in dataframe
        self.conditions=pd.DataFrame(conditions,columns=columns)#creating a dataframe for CPT
        self.jointProbability=self.conditions.copy()#creating a dataframe for joint porbability distribution
    #Calculating joint probability distribution
    def joint(self):
        #P(Monty,Prize,Guest)=P(Monty|Prize,Guest)*P(Prize)*P(Guest)
        for node in self.dependencyList:
            self.jointProbability['probability']=np.multiply(np.array(self.jointProbability['probability']),\
                #returns the marginal probability for given choice 
                np.array([node.randomVariable.eventChoices.get(choice) for choice in self.jointProbability[node.name]]))


