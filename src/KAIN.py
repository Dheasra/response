import numpy as np
from vampyr import vampyr3d as vp
from typing import Any

class Kain():
    """Mimics the mrchem::KAIN class from mrchem.
    The procedure is as follows:
    - the user provides a function f^n and an update, df^{n'} = f^{n+1'} - f^n, through application of a greens kernel. e.g. f^{n+1'} = G\star{f^n} .
    - the user calls the accelerate method, which appends f^n and df^{n'} to the function history vector f, and the update history vector df.
    - The accelerate methods solves a linear system constructed from both history vectors, f and df, to generate a kain update, df^{n}.
    - The accelerate method returns both f^n and df^{n} to the user, who can then use these to construct a proper update f^{n+1} = f^n + df^{n}.
    - for the next iteration the user provides the new fucntion f^{n+1} and a new update df^{n+1'} = f^{n+2'} - f^{n+1} (where  f^{n+2'} =G\star{f^{n+1}}), and repeats the procedure as before.
    
    For the zeroth iteration the accelerate method only appends and does not solve a linear system, as there is not enough history to do so.
    """
    
    instances = []
    
    A : np.ndarray[(Any, Any)]
    b : np.ndarray[(Any,)]
    c : np.ndarray[(Any,)]
    
    def __init__(self, history) -> None:
        self.instance_index = len(Kain.instances)
        self.instances.append(self)
        self.history = history
        
        self.f = []
        self.df = []
    
        
    def accelerate(self, func, dfunc):
        """
        Accelerate the function func using the Kain update procedure.
        We receive the function and its update, and append them to the history.
        then the linear system is setup and solved, and the solution is expanded.
        The original function is returned together with the Kain update, these two are meant to construct the new function for the next iteration.
        this is meant to mimic the way the mrchem::KAIN class works.

        Args:
            func (vampyr.vampyr3d.FunctionTree): function to be appended to the history
            dfunc (vampyr.vampyr3d.FunctionTree): update of the function before acceleration. To be appended to the history

        Returns:
            tuple(vampyr.vampyr3d.FunctionTree, vampyr.vampyr3d.FunctionTree): input function and its kain update
        """
        
        self.f.append(func.deepCopy())
        self.df.append(dfunc.deepCopy())
        
        if (len(self.f) == 1) or (len(self.df) == 1):
            return func, dfunc
        
        if ((len(self.f) > self.history) or (len(self.df) > self.history)):
            self.f.pop(0)
            self.df.pop(0)

        self.setupLinearSystem()
        self.solveLinearSystem()
        kain_update = self.expandSolution()
        
        self.clear()  # fill all ndarrays with zeros, might not be necessary
        
        return func, kain_update  
        

    def setupLinearSystem(self):
        nHistory = len(self.f) -1

        # Compute matrix A
        self.A = np.zeros((nHistory, nHistory))
        self.b = np.zeros(nHistory)
        phi_m = self.f[nHistory].deepCopy()
        fPhi_m = self.df[nHistory].deepCopy()
        
        for i in range(nHistory):
            phi_i = self.f[i]
            dPhi_im = phi_i - phi_m
            
            for j in range(nHistory):
                fPhi_j = self.df[j]
                dfPhi_jm = fPhi_j - fPhi_m
                self.A[i, j] -= vp.dot(dPhi_im, dfPhi_jm)
            self.b[i] += vp.dot(dPhi_im, fPhi_m)
        return   


    def solveLinearSystem(self):
        self.c = np.zeros(len(self.b))
        self.c = np.linalg.solve(self.A, self.b)
        return


    def expandSolution(self):
        nHistory = len(self.f) -1

        phi_m = self.f[nHistory].deepCopy()
        fPhi_m = self.df[nHistory].deepCopy()
        
        for j in range(nHistory):
            fPhi_m += (self.f[j] + self.df[j] - phi_m - fPhi_m)*self.c[j]
        
        return fPhi_m.deepCopy()
        
    def clear(self):
        self.A.fill(0)
        self.b.fill(0)
        self.c.fill(0)

    @property
    def A(self):
        return self.instances[self.instance_index]._A
    

    @A.setter
    def A(self, A):
        self.instances[self.instance_index]._A = A
        
    
    @property
    def b(self):
        return self.instances[self.instance_index]._b
    

    @b.setter
    def b(self, b):
        self.instances[self.instance_index]._b = b
        
    
    @property
    def c(self):
        return self.instances[self.instance_index]._c
    

    @c.setter
    def c(self, c):
        self.instances[self.instance_index]._c = c
        
    
    @property
    def f(self):
        return self.instances[self.instance_index]._f
    
    @f.setter
    def f(self, f):
        self.instances[self.instance_index]._f = f
    
    
    @property
    def df(self):
        return self.instances[self.instance_index]._df
    
    @df.setter
    def df(self, df):
        self.instances[self.instance_index]._df = df
        
            