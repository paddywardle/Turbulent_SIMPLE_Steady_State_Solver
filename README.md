**Introduction**

This project involves the derivation and implementation of a SIMPLE-based solver for turbulent incompressible flow using a simplified pressure projection method and a k-epsilon turbulence model.
Performance of the solver is to be demonstrated on a backward-facing step test case.

Relevant Files:
- **src.main** - Script to run simulation
- **src.SIMPLE** - Class to use the SIMPLE algorithm to solve the Incompressible Navier-Stokes equations
- **src.LinearSystem** - Class to discretise and solve the Incompressible Navier-Stokes and Pressure Laplacian equations, using a Fintie Volume discretisation approach.
- **src.LinearSystemBCs** - Class to discretise the Incompressible Navier-Stokes and Pressure Laplacian boundaries, using a Fintie Volume discretisation approach.
- **src.TurbulenceModel** - Class to discretise and solve the k-epsilon turbulence model, using a Fintie Volume discretisation approach.
- **src.TurbulenceModelBCs** - Class to discretise the k-epsilon turbulence model boundaries, using a Fintie Volume discretisation approach.
- **src.SparseMatrixCR** - Class for compressed row sparse matrix format.
- **src.Mesh** - Class to store the mesh needed for simulations.
- **src.utils.MeshParse** - Class to parse in OpenFOAM mesh format.
- **src.utils.ReadFiles** - Class to hold functions for all file reading requirements
- **src.utils.WriteFiles** - Class to hold functions for all file writing requirements
- **src.utils.InitialConds** - Script to produce initial condition files



