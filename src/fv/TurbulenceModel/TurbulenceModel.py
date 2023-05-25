import numpy as np
from fv.TurbulenceModel.TurbulenceModelBCs import TurbulenceModelBCs
from fv.fvMatrices.fvm.SuSp import SuSp
from fv.fvMatrices.fvc.Grad import Grad
from fv.fvMatrices.fvm.Ddt import Ddt
from fv.fvMatrices.fvc.Div import Div as fvcDiv
from fv.fvMatrices.fvm.Div import Div as fvmDiv
from fv.fvMatrices.fvMatrix import fvMatrix

class TurbulenceModel(Ddt, fvmDiv, fvcDiv, SuSp, fvMatrix, TurbulenceModelBCs):

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
    
    def KDisc(self, k, e, F, nu_face, nut_face, G, deltaT, BC):

        nueff_face = nu_face + nut_face / self.sigmak

        Addt, bddt = self.ddt(k, deltaT)
        Aconv, bconv = self.fvmDiv(F)
        Adiff, bdiff = self.laplacian(nueff_face)
        
        Aconv, bconv = self.ConvMatKEBCs(Aconv, bconv, F, BC, 4)
        Adiff, bdiff = self.DiffMatKEBCs(Adiff, bdiff, F, nueff_face, BC, 4)
        
        ASp, bSp = self.Sp(e/k, k)

        A = Addt + Aconv - Adiff + ASp # should ASp be + or -
        b = bddt + bconv - bdiff - bSp # check signs
        b += G * self.mesh.cell_volumes()

        A, b = self.relax(A, b, k)

        return A, b

    def EDisc(self, k, e, F, nu_face, nut_face, G, deltaT, BC):

        nueff_face = nu_face + nut_face / self.sigmaEps

        Addt, bddt = self.ddt(k, deltaT)
        Aconv, bconv = self.fvmDiv(F)
        Adiff, bdiff = self.laplacian(nueff_face)
        
        Aconv, bconv = self.ConvMatKEBCs(Aconv, bconv, F, BC, 5)
        Adiff, bdiff = self.DiffMatKEBCs(Adiff, bdiff, F, nueff_face, BC, 5)
        
        ASp, bSp = self.Sp(self.C2 * e/k, e)

        A = Addt + Aconv - Adiff + ASp # should ASp be + or -
        b = bddt + bconv - bdiff - bSp # check signs
        b += (self.C1 * G * (e / k)) * self.mesh.cell_volumes()

        A, b = self.relax(A, b, e)

        return A, b
