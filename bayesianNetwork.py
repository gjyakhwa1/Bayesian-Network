import numpy as np
import pandas as pd

class BayesianNetwork:
    def addStates(self,*args):
        self.nodes=[node for node in args]

    def addEdge(self,source,destination):
        destination.addParent(source)
    
    def getStates(self):
        return self.nodes
    
    def _getConditionalNode(self):
        for node in self.nodes:
            if node.randomVariable.__class__.__name__=='ConditionalProbabilityTable':
                return node
                
    def predict(self,Guest,Monty,Prize=None):
        node=self._getConditionalNode()
        randomV=node.randomVariable
        randomV.joint()
        prizeChoice={'A':0,'B':0,'C':0}
        for choice in prizeChoice.keys():
            #P(guest,monty,prize):-from joint probability
            numerator=randomV.jointProbability[(randomV.jointProbability['Guest']==Guest) & \
               (randomV.jointProbability['Monty']==Monty) & \
                    (randomV.jointProbability['Prize']==choice)]
            #numerator/P(Monty|guest)*P(guest)
            prizeChoice[choice]=float(numerator['probability'])/(0.5*self.nodes[0].randomVariable.getProbability(Guest))
        return prizeChoice
    
    def setDenominator(self,q):
        for node in self.nodes:
            if node.name==q:
                if node.randomVariable.__class__.__name__=='DiscreteDistribution':
                    return 0.5
                else:
                    return 0.33333333

    def predictProbability(self,condition):
        node=self._getConditionalNode()
        randomV=node.randomVariable
        randomV.joint()
        query=list(condition.keys())[list(condition.values()).index(None)]#get dictionary element with None
        choices={k:0 for k in randomV.conditions[query]}#create a dictionary for unique elements in query
        d=self.setDenominator(query)
        for choice in choices.keys():
            #P(guest,monty,prize):-from joint probability
            condition[query]=choice
            #choose the joint distribution according to the condition given 
            numerator=randomV.jointProbability.loc[(randomV.jointProbability[list(condition)] == pd.Series(condition)).all(axis=1)]['probability']
            #numerator/P(Monty|guest)*P(guest)
            choices[choice]=float(numerator)/(d*self.nodes[0].randomVariable.getProbability(list(condition.values())[0]))
        return choices

    
 