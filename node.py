class Node:
    def __init__(self, solution):
        self.u = 0
        self.c = 0
        self.solution = solution
        self.visited = []
        self.notVisited = []

    def getVisited(self):
        return self.visited
    
    def setVisited(self, visited):
        self.visited = visited

    def getNotVisited(self):
        return self.notVisited
    
    def setNotVisited(self, notVisited):
        self.notVisited = notVisited

    def getU(self):
        return self.u

    def setU(self, u):
        self.u = u
    
    def getC(self):
        return self.c
    
    def setC(self, c):
        self.c = c
    
    def getSolution(self):
        return self.solution
    
    def setSolution(self, solution):
        self.solution = solution