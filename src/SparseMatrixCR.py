class SparseMatrixCR:

    def __init__(self, rows, cols, data, default=0):

        self.rows = rows
        self.cols = cols
        self.default = default
        # initialise as an empty dictionary
        self.data = {}
        # if class is initialised with data
        if data:
            self.data = data
    
    def __getitem__(self, key):

        # testing to see if key is out of matrix dimensions
        if (key[0] >= self.rows) or (key[1] >= self.cols):
            print("Key is out of Matrix dimensions.")
        elif (key[0] < 0) or (key[1] < 0):
            print("Negative key is invalid.")
        # return data from self.data with corresponding key if it's there, otherwise return 0 (sparse matrix)
        return self.data.get(key, self.default)

    def __setitem__(self, key, val):

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