from vampyr import vampyr3d as vp
import numpy as np

class complex_tree:
    real : vp.functionTree
    imag : vp.functionTree

    def __init__(self, mra_input):
        self.real = vp.FunctionTree(mra_input)
        self.imag = vp.FunctionTree(mra_input)
        self.setZero()

    def setZero(self):
        self.real.setZero()
        self.imag.setZero()

    def __call__(self, position):
        rval = self.real(position)
        ival = self.imag(position)
        return rval + 1j * ival

    def __add__(self, other):
        output = complex_tree(self.mra)
        output.real = self.real + other.real
        output.imag = self.imag + other.imag
        return output
    
    def __sub__(self, other):
        return self + (-1)*other
    
    def __rmul__(self, other):
        if isinstance(other, complex) or isinstance(other, float) or isinstance(other, int):
            output = complex_tree(self.mra)
            # Separate the real and imaginary parts of the multiplier
            real_part = np.real(other)
            imag_part = np.imag(other)
            # Apply the real and imaginary multiplications
            output.real = self.real * real_part - self.imag * imag_part
            output.imag = self.real * imag_part + self.imag * real_part
            return output
        elif isinstance(other, complex_tree):
            output = complex_tree(self.mra)
            # Apply the real and imaginary multiplications
            output.real = self.real * other.real - self.imag * other.imag
            output.imag = self.real * other.imag + self.imag * other.real
            return output
        else:
            print("other =" ,type(other))
            raise TypeError("Unsupported type for multiplication with complex_tree")
        
    def __truediv__(self, other):
        if isinstance(other, complex) or isinstance(other, float) or isinstance(other, int):
            return self * (1/other)
        
    def __mul__(self, other):
        output = complex_fcn(self.mra)
        output.real = self.real * np.real(other) - self.imag * np.imag(other)
        output.imag = self.real * np.imag(other) + self.imag * np.real(other)
        return output
    
    #Things below this line are taken straight from Luca's complex_fcn class
    def dot(self, other, cc_first = True):
        out_real = 0
        out_imag = 0
        func_a = self.real
        func_b = self.imag
        func_c = other.real
        func_d = other.imag

        fbd = 1.0
        fbc = -1.0
        if(not cc_first):
            fbd = -1.0
            fbc = 1.0
        
        if(func_a.squaredNorm() > 0 and func_c.squaredNorm() > 0):
           out_real = out_real + vp.dot(func_a, func_c)
        if(func_b.squaredNorm() > 0 and func_d.squaredNorm() > 0):
           out_real = out_real + fbd * vp.dot(func_b, func_d)
        if(func_a.squaredNorm() > 0 and func_d.squaredNorm() > 0):
           out_imag = out_imag + vp.dot(func_a, func_d)
        if(func_b.squaredNorm() > 0 and func_c.squaredNorm() > 0):
           out_imag = out_imag + fbc * vp.dot(func_b, func_c)

        return out_real + 1j * out_imag
    
    def derivative(self, dir = 0, der = 'ABGV'):
        if(der == 'ABGV'):
            D = vp.ABGVDerivative(self.mra, 0.5, 0.5)
        elif(der == 'PH'):
            D = vp.PHDerivative(self.mra)
        elif(der == 'BS'):
            D = vp.BSDerivative(self.mra)
        else:
            exit("Derivative operator not found")
        re_der = D(self.real, dir)
        im_der = D(self.imag, dir)
        der_func = complex_fcn(self.mra)
        der_func.real = re_der
        der_func.imag = im_der
        return der_func
    
    def complex_conj(self):
        output = complex_fcn(self.mra)
        output.real = self.real 
        output.imag = -1.0 * self.imag
        return output
    
    def squaredNorm(self):
        re = self.real.squaredNorm()
        im = self.imag.squaredNorm()
        print("Cplx Fct Squared norm = ", re , "+ i", im)
        return re + im