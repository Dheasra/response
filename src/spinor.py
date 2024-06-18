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
        output.setZero()
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
            # print("spinor element-wise mult")
            for i in range(self.length):
                output.compVect[i] = other.compVect[i]*self.compVect[i]
            # print("mult norm", output.dot(output))
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
            if np.sqrt(self.compVect[i].real.squaredNorm()) > 1e-12:
                print("power spinor real")
                print(self.compVect[i].squaredNorm())
                print("power spinor real tut")
                output.compVect[i].real = self.compVect[i].real**(exponent)
                print("print spinor real ok")
            if np.sqrt(self.compVect[i].imag.squaredNorm()) > 1e-12:
                print("power spinor imag")
                output.compVect[i].imag = self.compVect[i].imag**(exponent)
        print("power spinor done")
        return output
    
    def __call__(self,component,  r): #r is the position 3-vector
        realval = self.compVect[component].real(r) 
        imagval = self.compVect[component].imag(r) 
        return realval + 1j*imagval

    def compSqNorm(self): 
        norm = 0.0
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
            if norm > 1e-12:
                self.compVect[i] = self.compVect[i]/norm

    def dot(self, other):
        output = 0.
        for i in range(self.length):
            # print("spinor dot")
            output += cf.dot(self.compVect[i], other.compVect[i])
        return output
    
    def setZero(self):
        for i in range(self.length):
            self.compVect[i].setZero()

    def derivative(self, direction, der = "BS"):
        output = spinor(self.mra, self.length)
        output.setZero()
        for i in range(self.length):
            if np.sqrt(self.compVect[i].squaredNorm()) > 1e-12:
                # print("derivative")
                output.compVect[i] = self.compVect[i].derivative(direction, der)
                # print("derivative", type(output.compVect[i]))
        return output
    
    def crop(self, prec): #Removes unnecessary leaves that are left after multiplying trees
        output = spinor(self.mra, self.length)
        for i in range(self.length):
            #This formulation should allow for this function to be called on both newly defined spinors and also the same way as the crop function of trees. 
            self.compVect[i].crop(prec) 
            output.compVect[i] = self.compVect[i]
        return output
        
    # def invert(self)
    #     one = spinor(self.mra, self.length)
    #     for i in range(self.length):
    #         one.compVect[i] = 
        
        