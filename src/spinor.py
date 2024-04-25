from vampyr import vampyr3d as vp
import numpy as np 
# import scfsolv

#Stuff to import complex_fcn from far away
import sys
import os
# Construct the absolute path to the directory containing the module.
module_path = os.path.abspath("/home/qpitto/Tests_KAIN/ZORA/ReMRChem2C/orbital4c")
# Append the module path to sys.path
sys.path.append(module_path)
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
        # output.compVect = self.compVect + other.compVect
        for i in range(self.length):
                output.compVect[i] = other.compVect[i] + self.compVect[i]
        return output

    def __sub__(self, other):
        # output = spinor(self.mra, self.length)
        # output.compVect = self.compVect - other.compVect
        return self + (-1)*other
    
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
            # output.compVect = other * self.compVect
            for i in range(self.length):
                output.compVect[i] = other*self.compVect[i]
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
                output.compVect[i].real = self.compVect[i].real*(other.compVect[i].real**(-1))
                output.compVect[i].imag = self.compVect[i].imag*(other.compVect[i].imag**(-1))
            return output

    def __pow__(self,exponent):
        output = spinor(self.mra, self.length)
        for i in range(self.length):
            output.compVect[i].real = self.compVect[i].real**(exponent)
            output.compVect[i].imag = self.compVect[i].imag**(exponent)
        return output

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
        return output
        
    # def invert(self)
    #     one = spinor(self.mra, self.length)
    #     for i in range(self.length):
    #         one.compVect[i] = 
        
        