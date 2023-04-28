import numpy as np

class LinearSolvers():

    def gauss_seidel(self, A, b, u, tol=1e-6, maxIts=1000):

        """
        This function uses the Gauss-Seidel algorithm to solve the linear system.

        Args:
            A (np.array): array containing the diagonal and off-diagonal contributions to the linear system.
            b (np.array): array containing the source contributions to the linear system.
            tol (float): tolerance for algorithm convergence.
            maxIts (int): maximum number of iterations that algorithm should run for.
        Returns:
            np.array: solution from Gauss-Seidel algorithm.

        """
        for k in range(maxIts):
            # forward sweep
            for i in range(A.shape[0]):
                u_new = b[i]
                for j in range(A.shape[0]):
                    if (j != i):
                        u_new -= A[i,j] * u[j]
                u[i] = u_new / A[i,i]
            # backward sweep
            for i in reversed(range(A.shape[0])):
                u_new = b[i]
                for j in reversed(range(A.shape[0])):
                    if (j != i):
                        u_new -= A[i,j] * u[j]
                u[i] = u_new / A[i,i]
            res = np.sum(b - np.matmul(A, u))
            if res < tol:
                break

        return u
