# kondo model with thresholded RD model
import model.conditions as boundcond
from fenics import *
import ufl
import numpy as np
set_log_level(40)
 
# parameters for neuruloid 
R = 3 # radius in 10^2 um
T = 0.5
dt = 0.1

# default parameters for Kondo Model
Ai = 50 # bc and ic
DA = (0.33 * 10**2) / 2
DI = (1.98 * 10**2) / 2
Imax = 15
a_A = 0.0073 * (1 + 0.0018 * Ai) * 360 / 2
b_A = 3.6 / 2
c_A = 1.08 / 2
d_A = 1.08 / 2
a_I = 0.0047 * (1 + 0.0030 * Ai) * 360 / 2
b_I = 0
c_I = -5.4 / 2
d_I = 3.24 / 2 
param_dict = {"DA":DA, "DI":DI, "Abound":Ai, "Amax":Ai, "Imax":Imax, 
              "aA":a_A, "bA":b_A, "cA":c_A, "dA":d_A,
              "aI":a_I, "bI":b_I, "cI":c_I, "dI":d_I}

def kondo_model_mixed_bc(mesh, 
                         u0, 
                         T, 
                         dt, 
                         param_dict, 
                         boundary_conditions,
                         save_file=False, 
                         save_step=0.5,
                         check_production=False):
    """
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
    DA = Constant(param_dict["DA"])
    DI = Constant(param_dict["DI"])
    Abound = Constant(param_dict["Abound"])
    Amax = Constant(param_dict["Amax"])
    Imax = Constant(param_dict["Imax"])
    aA = Constant(param_dict["aA"])
    bA = Constant(param_dict["bA"])
    cA = Constant(param_dict["cA"])
    dA = Constant(param_dict["dA"])
    aI = Constant(param_dict["aI"])
    bI = Constant(param_dict["bI"])
    cI = Constant(param_dict["cI"])
    dI = Constant(param_dict["dI"])

    F = ((u_A - u_nA) / k)*v_A*dx + DA*dot(grad(u_A), grad(v_A))*dx \
        - (ufl.Max(0, ufl.Min(Amax, aA*u_A + bA*u_I + cA)) - dA*u_A)*v_A*dx \
        + ((u_I - u_nI) / k)*v_I*dx + DI*dot(grad(u_I), grad(v_I))*dx \
        - (ufl.Max(0, ufl.Min(Imax, aI*u_A + bI*u_I + cI)) - dI*u_I)*v_I*dx \
    
    if boundary_conditions=="Mixed":
        bc = DirichletBC(V.sub(0), Abound, boundcond.boundary)
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
            A, I = u_n.split()
            P_A = FunctionSpace(mesh, 'P', 1)
            P_I = FunctionSpace(mesh, 'P', 1)
            prod_A = project(ufl.Max(0, ufl.Min(Amax, aA*A + bA*I + cA)), P_A)
            prod_I = project(ufl.Max(0, ufl.Min(Imax, aI*A + bI*I + cI)), P_I)
            vtk_prod_A << (prod_A, t)
            vtk_prod_I << (prod_I, t)

    for n in range(num_steps):

        # solve variational problem for time step
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
                    A, I = u_n.split()
                    P_A = FunctionSpace(mesh, 'P', 1)
                    P_I = FunctionSpace(mesh, 'P', 1)
                    prod_A = project(ufl.Max(0, ufl.Min(Amax, aA*A + bA*I + cA)), P_A)
                    prod_I = project(ufl.Max(0, ufl.Min(Imax, aI*A + bI*I + cI)), P_I)
                    vtk_prod_A << (prod_A, t)
                    vtk_prod_I << (prod_I, t)

    return u_n # return final state

def fixed_points(a_A=a_A,
                 b_A=b_A,
                 c_A=c_A,
                 d_A=d_A,
                 a_I=a_I,
                 b_I=b_I,
                 c_I=c_I,
                 d_I=d_I):
    """
    Take parameter values and find fixed points values
    """
    denom = (a_A - d_A) * (d_I - b_I) + b_A * a_I
    Aeq = (-b_A * c_I - c_A * (d_I - b_I)) / denom
    Ieq = (c_I * (a_A - d_A) - a_I * c_A) / denom
    return Aeq, Ieq
 
def check_Turing_instability(D_A=DA,
                             D_I=DI,
                             a_A=a_A,
                             b_A=b_A,
                             c_A=c_A,
                             d_A=d_A,
                             a_I=a_I,
                             b_I=b_I,
                             c_I=c_I,
                             d_I=d_I):
    """
    Take parameter values and check if Turing instability satisfied
    Returns:
      Stability of fixed point (Bool): True if stable 
      Turing Instability (Bool): True if Turing unstable
    """
    fp = False
    if a_A - d_A + b_I - d_I < 0 and (-(a_A - d_A) * (d_I - b_I) - b_A * a_I) > 0:
        fp = True
    
    Ti = False
    if fp:
        if D_I * (a_A - d_A) - D_A * (d_I - b_I) > 0 and (D_I * (a_A - d_A) - D_A * (d_I - b_I))**2 > 4*D_A*D_I*(-(a_A - d_A) * (d_I - b_I) - b_A * a_I):
            Ti = True
    
    return fp, Ti