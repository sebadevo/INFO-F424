class Node:
    def __init__(self, solution, upperbound, cost, row):
        self.upperbound = upperbound
        self.cost = cost
        self.solution = solution
        self.row = row
        self.isDone = False # on s'en fou

    def getIsDone(self):
        return self.isDone

    def setIsDone(self):
        return self.isDone

    def getRow(self):
        return self.row
    
    def setRow(self, row):
        self.row = row

    def getUpperbound(self):
        return self.upperbound

    def setUpperbound(self, upperbound):
        self.upperbound = upperbound
    
    def getCost(self):
        return self.cost
    
    def setCost(self, cost):
        self.cost = cost
    
    def getSolution(self):
        return self.solution
    
    def setSolution(self, solution):
        self.solution = solution