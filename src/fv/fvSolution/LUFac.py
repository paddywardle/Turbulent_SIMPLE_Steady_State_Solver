import numpy as np

class GaussianElim():

    def lu(self, A):

        """
        This function performs LU Factorisation on a dense matrix.

        Args:
            A (np.array): array containing the diagonal and off-diagonal contributions to the linear system.
            b (np.array): array containing the source contributions to the linear system.
            tol (float): tolerance for algorithm convergence.
            maxIts (int): maximum number of iterations that algorithm should run for.
        Returns:
            np.array: solution from Gauss-Seidel algorithm.

        """

        N = len(A)
        L = np.eye(N)
        U = A.copy()

        for i in range(1, N):
            for k in range(i):
                L[i,k] = U[i,k]/U[k,k]
                U[i,k+1:N] -= L[i,k] * U[k,k+1:N]
            U[i,:i] = 0.0

        return L, U

if __name__ == "__main__":

    A = np.array([[2,2,3], [5,9,10], [4,1,2]], dtype=float)

    ge = GaussianElim()

    L, U = ge.lu(A)
    print(L)
    print(U)
