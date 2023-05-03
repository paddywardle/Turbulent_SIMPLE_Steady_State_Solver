import numpy as np

class Tensor():

    def __init__(self):

        pass

    def twoSymm(self, T):

        twoSymm = np.zeros_like(T)

        twoSymm[0, :] = 2 * T[0, :]
        twoSymm[4, :] = 2 * T[4, :]
        twoSymm[8, :] = 2 * T[8, :]

        twoSymm[1, :] = T[1, :] + T[3, :]
        twoSymm[2, :] = T[2, :] + T[6, :]
        
        twoSymm[3, :] = T[3, :] + T[1, :]
        twoSymm[5, :] = T[5, :] + T[7, :]

        twoSymm[6, :] = T[6, :] + T[2, :]
        twoSymm[7, :] = T[7, :] + T[5, :]

        return twoSymm

    def Symm(self, T):

        twoSymm = self.twoSymm(T)

        Symm = 0.5 * twoSymm
        
        return Symm

    def DoubleInner(self, T1, T2):
        
        DoubleInner = T1 * T2

        DoubleInner = DoubleInner.sum(axis=0)
        
        return DoubleInner
        

    
