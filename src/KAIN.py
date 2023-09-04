import numpy as np
from vampyr import vampyr3d as vp
from typing import Any, List

class KAIN():
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
    
    A : List[np.ndarray[(Any, Any)]]  # vector of A matrices
    b : List[np.ndarray[(Any,)]]      # vector of b vectors
    c : List[np.ndarray[(Any,)]]      # vector of c vectors
    
    func_history : List[List[vp.FunctionTree]]
    update_history : List[List[vp.FunctionTree]]
    
    def __init__(self, history) -> None:
        self.instance_index = len(KAIN.instances)
        self.instances.append(self)
        self.history = history
        
        self.func_history = []
        self.update_history = [] 
        
        self.A = []
        self.b = []
        self.c = []   
        
    def accelerate(self, functions, functions_updates):
        """
        Accelerate the function func using the Kain update procedure.
        We receive the function and its update, and append them to the history.
        then the linear system is setup and solved, and the solution is expanded.
        The original function is returned together with the Kain update, these two are meant to construct the new function for the next iteration.
        this is meant to mimic the way the mrchem::KAIN class works.

        Args:
            functions (vampyr.vampyr3d.FunctionTree): function list to be appended to the history
            functions_updates (vampyr.vampyr3d.FunctionTree): list of updates of the functions before acceleration. To be appended to the history

        Returns:
            tuple(vampyr.vampyr3d.FunctionTree, vampyr.vampyr3d.FunctionTree): input function list and its kain update list
        """
        
        self.func_history.append(functions)
        self.update_history.append(functions_updates)
        
        if (len(self.func_history) == 1) or (len(self.update_history) == 1):
            return functions_updates
        
        if ((len(self.func_history) > self.history) or (len(self.update_history) > self.history)):
            self.func_history.pop(0)
            self.update_history.pop(0)

        self.setupLinearSystem()
        self.solveLinearSystem()
        kain_updates = self.expandSolution()
        self.cleanLinearSystem()
        
        return kain_updates
        

    def setupLinearSystem(self):
        nHistory = len(self.func_history) -1
        nOrbitals = len(self.func_history[nHistory])
        print("nOrbitals: ", nOrbitals )
        for n in range(nOrbitals):
            orbA = np.zeros((nHistory, nHistory))
            orbB = np.zeros(nHistory)
            
            phi_m = self.func_history[nHistory][n]
            fPhi_m = self.update_history[nHistory][n]
            
            for i in range(nHistory):
                phi_i = self.func_history[i][n]
                dPhi_im = phi_i - phi_m
                
                for j in range(nHistory):
                    fPhi_j = self.update_history[j][n]
                    dfPhi_jm = fPhi_j - fPhi_m
                    orbA[i, j] -= vp.dot(dPhi_im, dfPhi_jm)
                orbB[i] += vp.dot(dPhi_im, fPhi_m)
            print("appending to A and b")
            self.A.append(orbA)
            self.b.append(orbB)
        print("length of A: ", len(self.A))
        print("length of b: ", len(self.b))
        return   


    def solveLinearSystem(self):
        print("length of b:", (len(self.b)))
        for i in range(len(self.b)):
            tempC = np.linalg.solve(self.A[i], self.b[i])
            self.c.append(tempC)
        print("length of c: ", len(self.c))
        print("length og c[0]: ", len(self.c[0]))
        return


    def expandSolution(self):
        kain_updates = []
        nHistory = len(self.func_history) -1
        nOribtals = len(self.func_history[nHistory])
        for n in range(nOribtals):    
            phi_m = self.func_history[nHistory][n].deepCopy()
            fPhi_m = self.update_history[nHistory][n].deepCopy()
        
            for j in range(nHistory):

                fPhi_m += (self.func_history[j][n] + self.update_history[j][n] - phi_m - fPhi_m)*self.c[n][j]
        
            kain_updates.append(fPhi_m)
        return kain_updates


    def cleanLinearSystem(self):
        # TODO check that no memory leak happens here
        self.A = []
        self.b = []
        self.c = []
    
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
    def func_history(self):
        return self.instances[self.instance_index]._func_history
    
    @func_history.setter
    def func_history(self, func):
        self.instances[self.instance_index]._func_history = func
    
    
    @property
    def update_history(self):
        return self.instances[self.instance_index]._update_history
    
    @update_history.setter
    def update_history(self, upd):
        self.instances[self.instance_index]._update_history = upd
        
            