**Introduction**

This project involves derivation and implementation of a SIMPLE-based
solver for incompressible flow using a simplified pressure projection method.
Performance of the solver is to be demonstrated on a 2-D lid-driven cavity
simulation case.

Relevant Files:
    * **src.main** - Script to run simulation
    * **src.SIMPLE** - Class to use the SIMPLE algorithm to solve the Incompressible Navier-Stokes equations
    * **src.LinearSystem** - Class to discretise and sovle the Incompressible Navier-Stokes and Pressure Laplacian equations, using a Fintie Volume discretisation approach
    * **src.SparseMatrixCR** - Class for compressed row sparse matrix format
    * **src.Mesh** - Class to store the mesh needed for simulations
    * **src.utils.ReadFiles** - Class to hold functions for all file reading requirements
    * **src.utils.WriteFiles** - Class to hold functions for all file writing requirements
    * **src.utils.InitialConds** - Script to produce initial condition files
    * **src.CellsParser** - Script to parse cell file from OpenFOAM owner neighbour file format
    * **src.BoundaryParser** - Script to parse boundary_patches file from OpenFOAM boundary file format
    * **config.config.json** - Simulation settings file
    * **plots.plot_script.py** - Script to plot results graphs



