import numpy as np
from fv.TurbulenceModel.TurbulenceModelBCs import TurbulenceModelBCs
from fv.fvMatrices.fvm.SuSp import SuSp
from fv.fvMatrices.fvc.Grad import Grad
from fv.fvMatrices.fvc.Div import Div as fvcDiv
from fv.fvMatrices.fvm.Div import Div as fvmDiv
from fv.fvMatrices.fvMatrix import fvMatrix

class TurbulenceModel(fvmDiv, fvcDiv, SuSp, fvMatrix, TurbulenceModelBCs):

    """
    Class to discretise the k-e turbulence model equations to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh, conv_scheme, alpha_u, Cmu, C1, C2, C3, sigmak, sigmaEps):

        self.mesh = mesh
        self.conv_scheme = conv_scheme
        self.alpha_u = alpha_u
        self.Cmu = Cmu
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.sigmak = sigmak
        self.sigmaEps = sigmaEps
    
    def KDisc(self, k, e, F, veffk, G, BC):

        veff_face = self.veff_face(veffk)

        Aconv, bconv = self.fvmDiv(F)
        Adiff, bdiff = self.laplacian(veff_face)
        Aconv, bconv = self.ConvMatKEBCs(Aconv, bconv, F, BC, 4)
        Adiff, bdiff = self.DiffMatKEBCs(Adiff, bdiff, F, veff_face, BC, 4)
        
        ASp, bSp = self.Sp(e/k, k)

        A = Aconv - Adiff + ASp # should ASp be + or -
        b = bconv - bdiff - bSp # check signs
        b += G * self.mesh.cell_volumes()

        A, b = self.relax(A, b, k)

        return A, b

    def EDisc(self, k, e, F, veffe, G, BC):
        
        veff_face = self.veff_face(veffe)

        Aconv, bconv = self.fvmDiv(F)
        Adiff, bdiff = self.laplacian(veff_face)
        Aconv, bconv = self.ConvMatKEBCs(Aconv, bconv, F, BC, 5)
        Adiff, bdiff = self.DiffMatKEBCs(Adiff, bdiff, F, veff_face, BC, 5)
        
        ASp, bSp = self.Sp(self.C2 * e/k, e)

        A = Aconv - Adiff + ASp # should ASp be + or -
        b = bconv - bdiff - bSp # check signs
        b += (self.C1 * G * (e / k)) * self.mesh.cell_volumes()

        A, b = self.relax(A, b, e)

        return A, b
