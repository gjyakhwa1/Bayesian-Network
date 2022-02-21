import numpy as np
from distribution import DiscreteDistribution, ConditionalProbabilityTable
from typing import Union

class Node:
    def __init__(self,randomVariable:Union[DiscreteDistribution,ConditionalProbabilityTable]):
        self.randomVariable = randomVariable #can be Discrete Distribution or CPT
        self.name= self.randomVariable.name 
        self.parentNode = None
        self.childNodes=None

    def addParent(self, node):
        if self.parentNode==None:
            self.parentNode=[]
        self.parentNode.append(node)
        self._addChild(node)
    
    def _addChild(self,node):
        if node.childNodes==None:
            node.childNodes=[]
        node.childNodes.append(self)
    


