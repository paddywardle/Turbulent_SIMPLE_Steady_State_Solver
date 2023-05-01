import numpy as np

class Tensor():

    def __init__(self):

        pass

    def twoSymm(self, T):

        twoSymm = np.zeros_like(T)

        twoSymm[:, 0] = self.diag(twoSymm, 0)
        twoSymm[:, 4] = self.diag(twoSymm, 4)
        twoSymm[:, 8] = self.diag(twoSymm, 8)

        twoSymm[:, 1] = T[:, 1] + T[:, 3]
        twoSymm[:, 2] = T[:, 2] + T[:, 6]
        
        twoSymm[:, 3] = T[:, 3] + T[:, 1]
        twoSymm[:, 5] = T[:, 5] + T[:, 7]

        twoSymm[:, 6] = T[:, 6] + T[:, 2]
        twoSymm[:, 7] = T[:, 7] + T[:, 5]

        return twoSymm

    def Symm(self, T):

        twoSymm = self.twoSymm(T)

        Symm = 0.5 * twoSymm
        
        return Symm

    def DoubleInner(self, T):

        Symm = self.Symm(T)
        
        DoubleInner = np.zeros_like(Symm)

        DoubleInner = np.square(T[:, 0]) + np.square(T[:, 1]) + np.square(T[:, 2]) + np.square(T[:, 3]) + np.square(T[:, 4]) + np.square(T[:, 5]) + np.square(T[:, 6]) + np.square(T[:, 7]) + np.square(T[:, 8])

    def Diag(self, T, idx):

        return 2 * T[:, idx]
        

    
