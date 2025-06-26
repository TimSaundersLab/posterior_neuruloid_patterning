import model.conditions as boundcond
from fenics import *
import ufl
import numpy as np
set_log_level(40)

DA = 0.007
DI = 0.1
delta = DI/DA
aA = 0.08
bA = -0.08
cA = 0.05
dA = 0.03
aI = 0.1
bI = 0
cI = -0.15
dI = 0.06

def dimensionless_kondo(mesh,
                        u0,
                        T,
                        dt,
                        param_set,
                        boundary_conditions,
                        save_file=False,
                        save_step=0.1,
                        check_production=False):
    """
    Dimensionless Kondo model
    Args:
        mesh (mshr): domain of modelling
        u0 (function space): initial conditions
        T (int): final time
        dt (float): discretisation of time
        param_dict (dict): dictionary of 12 constants for the equation
    """
    # define space
    P1 = FiniteElement('P', triangle, 1)
    element = MixedElement([P1, P1])
    V = FunctionSpace(mesh, element)

    # define function space for variational problem
    v_A, v_I = TestFunction(V)
    u = Function(V)
    u_A, u_I = split(u)

    # initial conditions space
    u_n = Function(V)
    u_nA, u_nI = split(u_n)
    u_n.interpolate(u0)

    # define constants used in variational problem
    # unknowns are [u_A, u_I] and corresponding test function [v_A, v_I]
    k = Constant(dt)
    delta = Constant(param_set["delta"])
    gamma = Constant(param_set["gamma"])
    kappa = Constant(param_set["kappa"])
    Amax = Constant(param_set["Amax_nd"])
    Imax = Constant(param_set["Imax_nd"])
    aA = Constant(param_set["aA"])
    bA = Constant(param_set["bA"])
    cA = Constant(param_set["cA_nd"])
    dA = Constant(param_set["dA"])
    aI = Constant(param_set["aI"])
    bI = Constant(param_set["bI"])
    cI = Constant(param_set["cI_nd"])

    F = ((u_A - u_nA) / k)*v_A*dx + dot(grad(u_A), grad(v_A))*dx \
        - gamma*((1/dA)*ufl.Max(0, ufl.Min(Amax, aA*u_A + bA*u_I + cA)) - u_A)*v_A*dx \
        + ((u_I - u_nI) / k)*v_I*dx + delta*dot(grad(u_I), grad(v_I))*dx \
        - gamma*((1/dA)*ufl.Max(0, ufl.Min(Imax, aI*u_A + bI*u_I + cI)) - kappa*u_I)*v_I*dx \
        
    if boundary_conditions=="Mixed":
        bc = DirichletBC(V.sub(0), 1, boundcond.boundary)
    elif boundary_conditions=="Neumann": 
        bc = None

    # time stepping
    num_steps = int(T/dt)
    t = 0

    # files for saving
    if type(save_file)==str:
        vtkfile_A = File(f"{save_file}/A.pvd")
        vtkfile_I = File(f"{save_file}/I.pvd")
        _u_nA, _u_nI = u_n.split() # first save initial conditions at t=0
        vtkfile_A << (_u_nA, t)
        vtkfile_I << (_u_nI, t)

        if check_production:
            # to check how production term is varying
            vtk_prod_A = File(f"{save_file}/prod_A.pvd")
            vtk_prod_I = File(f"{save_file}/prod_I.pvd")
            P_A, P_I = FunctionSpace(mesh, 'P', 1), FunctionSpace(mesh, 'P', 1)
            prod_A = project(ufl.Max(0, ufl.Min(Amax, aA*_u_nA + bA*_u_nI + cA)), P_A)
            prod_I = project(ufl.Max(0, ufl.Min(Imax, aI*_u_nA + bI*_u_nI + cI)), P_I)
            vtk_prod_A << (prod_A, t)
            vtk_prod_I << (prod_I, t)

    for n in range(num_steps):

        # solve variational problem
        solve(F==0, u, bcs=bc)

        # update previous solution
        u_n.assign(u)

        t += dt

        if type(save_file)==str:
            if round(t/save_step, 5).is_integer():
                _u_A, _u_I = u.split()
                vtkfile_A << (_u_A, t)
                vtkfile_I << (_u_I, t)

                if check_production:
                    P_A, P_I = FunctionSpace(mesh, 'P', 1), FunctionSpace(mesh, 'P', 1)
                    prod_A = project(ufl.Max(0, ufl.Min(Amax, aA*_u_A + bA*_u_I + cA)), P_A)
                    prod_I = project(ufl.Max(0, ufl.Min(Imax, aI*_u_A + bI*_u_I + cI)), P_I)
                    vtk_prod_A << (prod_A, t)
                    vtk_prod_I << (prod_I, t)

    return u_n

def fixed_points(aA=aA,
                 bA=bA,
                 cA=cA,
                 dA=dA,
                 aI=aI,
                 bI=bI,
                 cI=cI,
                 dI=dI):
    """
    Take parameter values and find fixed points values
    """
    denom = (aA - dA) * (dI - bI) + bA * aI
    Aeq = (-bA * cI - cA * (dI - bI)) / denom
    Ieq = (cI * (aA - dA) - aI * cA) / denom
    return Aeq, Ieq
 
def check_Turing_instability(delta=delta,
                             aA=aA,
                             bA=bA,
                             cA=cA,
                             dA=dA,
                             aI=aI,
                             bI=bI,
                             cI=cI,
                             dI=dI):
    """ 
    Take parameter values and check if Turing instability satisfied
    Returns:
      Stability of fixed point (Bool): True if stable 
      Turing Instability (Bool): True if Turing unstable
    """
    fp = False
    if aA - dA + bI - dI < 0 and (-(aA - dA) * (dI - bI) - bA * aI) > 0:
        fp = True
    
    Ti = False
    if fp:
        if delta * (aA - dA) + (bI - dI) > 0 and (delta*(aA - dA) + (bI - dI))**2 > 4*delta*((aA - dA) * (bI - dI) - bA * aI):
            Ti = True
    
    return fp, Ti

