# functions for setting initial conditions for FEM
# Boundary functions for fenics

from fenics import *
import numpy as np
set_log_level(40)

def non_negative(u):
    """
    Force solutions to be non-negative
    Args:
        Function Class
    """
    Uvector = as_backend_type(u.vector()).get_local()
    Uvector[Uvector <= DOLFIN_EPS_LARGE] =  DOLFIN_EPS_LARGE # if value less than small tol, set to tol
    u.vector().set_local(Uvector)
    return u

def boundary(x, on_boundary):
    return on_boundary

# initial conditions with random perturbations around values Ac (activator) and Ic (inhibitor)
class InitialConditionsRandom(UserExpression):
    """
    Create random initial conditions, i.e. perturbations around constants:
        Activator concentration Ac
        Inhibitor concentration Ic
    """
    def __init__(self,
                 noise,
                 Ac,
                 Ic,
                 degree=1):
        self.noise = noise
        self.Ac = Ac
        self.Ic = Ic
        super().__init__(degree)
    def eval(self, values, x):
        values[0] = self.Ac + self.noise*np.random.normal()
        values[1] = self.Ic + self.noise*np.random.normal()
    def value_shape(self):
        return (2,)

class InitialConditionsGradient(UserExpression):
    """
    Create initial conditions from Tenary et al (2017) paper
    For a circular domain with radius R
    Initial conditions are:
        Activator concentration constant at Ai
        Inhibitor concentration a gradient (high conc at boundary, low at center)
    """
    def __init__(self,
                 degree,
                 Ai,
                 R):
        self.Ai = Ai
        self.R = R
        super().__init__(degree)

    def eval(self, values, x):
        values[0] = self.Ai
        values[1] = 1 - (x[0]/self.R)*(x[0]/self.R) - (x[1]/self.R)*(x[1]/self.R)

    def value_shape(self):
        return (2,)
    
def interpolate_ic(u0, mesh):
    P1 = FiniteElement('P', triangle, 1)
    element = MixedElement([P1, P1])
    V = FunctionSpace(mesh, element)
    u = Function(V)
    Ai, Ii = split(u)
    u.interpolate(u0)
    return Ai, Ii



