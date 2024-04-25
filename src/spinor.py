from vampyr import vampyr3d as vp
import numpy as np 
import scfsolv, utils
from complex_fcn import complex_fcn as cf 

class spinor: 
    mra : vp.MultiResolutionAnalysis
    compVect : np.array
    length : int

    def __init__(self, mra, N_components = 4):
        self.mra = mra 
        self.compVect = np.array([cf(mra) for i in range(N_components)])
        self.length = N_components

    def __len__(self):
        return self.length
    
    def __add__(self, other):
        output = spinor(self.mra, self.length)
        output.compVect = self.compVect + other.compVect
        return output

    def __sub__(self, other):
        output = spinor(self.mra, self.length)
        output.compVect = self.compVect - other.compVect
        return output
    
    def __rmul__(self, other):
        # output = spinor(self.mra, self.length)
        # if type(other) == float or type(other) == int or type(other) == complex: 
        #     output.compVect = other * self.compVect
        # elif type(other) == spinor: 
        #     for i in range(self.length):
        #         output.compVect[i] = other.compVect[i]*self.compVect[i]
        return self*other #I hope this is calling __mul__ 
    

    def __mul__(self, other):
        # output = spinor(self.mra, self.length)
        # output.compVect = factor * self.compVect
        output = spinor(self.mra, self.length)
        if type(other) == float or type(other) == int or type(other) == complex: 
            output.compVect = other * self.compVect
        elif type(other) == spinor: 
            for i in range(self.length):
                output.compVect[i] = other.compVect[i]*self.compVect[i]
        return output 
    
    def __truediv__(self, other):
        if type(other) == float or type(other) == int or type(other) == complex:
            return self*(1/other)
        elif type(other) == spinor: 
            output = spinor(self.mra, self.length)
            for i in range(self.length):
                output.compVect[i] = self.compVect[i]/other

    def compSqNorm(self): 
        norm = 0
        for i in range(self.length):
            norm += self.compVect[i].squaredNorm()
        return norm 
    
    def compNorm(self):
        norm2 = self.compSqNorm()
        return np.sqrt(norm2)

    def normalize(self):
        # output = spinor(self.mra, self.length)
        norm = self.compNorm()
        for i in range(self.length):
            self.compVect[i] = self.compVect[i]/norm

    def dot(self, other):
        output = 0.
        for i in range(self.length):
            output += cf.dot(self.compVect[i], other.compVect[i])
        return output
    
    def setZero(self):
        for i in range(self.length):
            self.compVect[i].setZero()

    def derivative(self, direction, der = "ABGV"):
        output = spinor(self.mra, self.length)
        for i in range(self.length):
            output.compVect[i] = self.compVect[i].derivative(direction, der)
        return der
        
    # def invert(self)
    #     one = spinor(self.mra, self.length)
    #     for i in range(self.length):
    #         one.compVect[i] = 
        
        