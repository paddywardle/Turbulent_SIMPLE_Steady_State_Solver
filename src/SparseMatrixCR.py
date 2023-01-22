import numpy as np

class SparseMatrixCR:

    def __init__(self, rows, cols, data=np.array([]), col_array=np.array([]), row_ptrs=np.array([]), default=0):

        self.rows = rows
        self.cols = cols
        self.default = default
        # initialise as an empty dictionary
        self.data = data
        self.col_array = col_array
        self.row_ptrs = row_ptrs

    def __getitem__(self, key):

        """
        Dunder method for operator overload of [] operator for getting a value based on key.

        Args:
            key (tuple): contains indices of value being requests (row, column)
        Returns:
            int: value at key index.
        """

        # testing to see if key is out of matrix dimensions
        if (key[0] >= self.rows) or (key[1] >= self.cols):
            print("Key is out of Matrix dimensions.")
        elif (key[0] < 0) or (key[1] < 0):
            print("Negative key is invalid.")
        # return data from self.data with corresponding key if it's there, otherwise return 0 (sparse matrix)
        row_start = self.row_ptrs[key[0]]
        row_end = self.row_ptrs[key[0]+1]
        row_vals = self.data[row_start:row_end]
        col_indices = self.col_array[row_start:row_end]
        if key[1] in col_indices:
            return row_vals[col_indices.index(key[1])]
        return self.default
    
    def __setitem__(self, key, val):

        """
        Dunder method for operator overload of [] operator for setting a value based on a key and value.

        Args:
            key (tuple): contains indices of value being set (row, column)
            val (float): value being added to sparse matrix at key.

        """

        # testing to see if key is out of matrix dimensions
        if (key[0] >= self.rows) or (key[1] >= self.cols):
            print("Key is out of Matrix dimensions.")
        elif (key[0] < 0) or (key[1] < 0):
            print("Negative key is invalid.")
        # testing if value is zero
        if val != self.default:
            # adding new value to self.data with corresponding key
            self.data[key] = val
        # testing to see if the key is already in self.data, if it is delete it so it will appear as 0 (sparse matrix format)
        elif key in self.data:
            del self.data[key]

    def from_dense(self, matrix):

        """
        This method converts a dense matrix argument to a sparse matrix format.

        Args:
            key (tuple): contains indices of value being requests (row, column)

        """

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if j == 0:
                    self.row_ptrs = self.row_ptrs.append(self.row_ptrs, len(self.col_array))
                if matrix[i][j] == self.default:
                    continue
                self.data = np.append(self.data, matrix[i][j])
                self.col_array = np.append(self.col_array, j)