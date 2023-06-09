import numpy as np

class GaussianElim():

    def ge(self, A):

        """
        This function performs Gaussian Elimination on a dense matrix.

        Args:
            A (np.array): array containing the diagonal and off-diagonal contributions to the linear system.
            b (np.array): array containing the source contributions to the linear system.
            tol (float): tolerance for algorithm convergence.
            maxIts (int): maximum number of iterations that algorithm should run for.
        Returns:
            np.array: solution from Gauss-Seidel algorithm.

        """

        N = len(A)
        A = A.copy()

        for i in range(1, N):
            for k in range(i):
                A[i,k] = A[i,k]/A[k,k]
                for j in range(k+1, N):
                    A[i,j] -= A[i,k] * A[k,j]

        return A
