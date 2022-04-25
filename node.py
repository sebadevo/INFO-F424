class Node:
    def __init__(self, constraints, upperbound, lowerbound, parent):
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.constraints = constraints
        self.parent = parent
        self.childs = []
        
    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_childs(self):
        return self.childs
    
    def add_child(self, child):
        self.childs.append(child)
                
    def get_upperbound(self):
        return self.upperbound

    def set_upperbound(self, upperbound):
        self.upperbound = upperbound
    
    def get_lowerbound(self):
        return self.lowerbound
    
    def set_lowerbound(self, lowerbound):
        self.lowerbound = lowerbound
    
    def get_constraints(self):
        if self.constraints is None:
            return []
        return self.constraints
    
    def set_constraints(self, constraints):
        self.constraints = constraints