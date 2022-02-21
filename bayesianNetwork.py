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

    def getNumerator(self,randomV,predictionDictionary):
        dfFilterCondition=(randomV.jointProbability[predictionDictionary.keys()] == predictionDictionary.values()).all(axis=1)
        return randomV.jointProbability.loc[dfFilterCondition]['probability']
    
    def getDenominator(self,randomV,predictionDictionary,query):
        #get the node in node list which is none in the query
        node=[node for node in self.nodes if node.name==query][0]
        filterCondition=predictionDictionary
        filterCondition.pop(query)
        if node.randomVariable.__class__.__name__=='DiscreteDistribution':
            #P(Monty|guest)*P(guest)
            #P(Monty|guest)  or P(Monty|prize)
            #calculations--first finding all the tuples from CPT with Monty and (guest or prize) given
            cpt=randomV.conditions.loc[(randomV.conditions[filterCondition.keys()]==filterCondition.values()).all(axis=1)]['probability']
            denominator=np.array(cpt).mean()
            #P(guest) or P(prize)
            discreteNode=[i for i in randomV.dependencyList if i!=node][0]
            choice=filterCondition.get(discreteNode.name)
            #P(Monty|guest)*P(guest)
            denominator*=discreteNode.randomVariable.eventChoices.get(choice)
            return denominator
        else:
            #P(Prize)*P(Guest)
            denominator=1
            dNodeList=[i for i in self.nodes if i.name in filterCondition.keys()]
            for dNode in dNodeList:
                denominator*=dNode.randomVariable.eventChoices.get(filterCondition.get(dNode.name))
            return denominator

    def predictProbability(self,predictionDictionary):
        node=self._getConditionalNode()
        randomV=node.randomVariable
        randomV.joint()#calculate joint probability using Conditional Probability table and Discrete Disctribution
        query=list(predictionDictionary.keys())[list(predictionDictionary.values()).index(None)]#get dictionary element with None
        choices={k:0 for k in randomV.conditions[query]}#create a dictionary for unique elements in query
        for choice in choices.keys():
            #P(guest,monty,prize):-from joint probability
            predictionDictionary[query]=choice
            #choose the joint distribution according to the condition given 
            numerator=self.getNumerator(randomV,predictionDictionary)
            denominator=self.getDenominator(randomV,predictionDictionary,query)
            #numerator/P(Monty|guest)*P(guest)
            choices[choice]=float(numerator)/denominator
        return choices

    
 