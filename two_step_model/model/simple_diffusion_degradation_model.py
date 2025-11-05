# model for simple diffusion degradation model with fixed boundary conditions
import model.conditions as boundcond
from fenics import *
import ufl
import numpy as np
set_log_level(40)

D = 0.1
d = 0.05

def simple_diffusion_degradation(mesh,
                                 u0,
                                 T,
                                 dt,
                                 param_set,
                                 boundary_conditions,
                                 save_file=False,
                                 save_step=0.1):
    """
    Dimensionless Kondo model
    Args:
        mesh (mshr): domain of modelling
        u0 (function space): initial conditions
        T (int): final time
        dt (float): discretisation of time
        param_dict (dict): dictionary of 2 constants for the equation
    """
    # define space
    V = FunctionSpace(mesh, 'P', 1)

    # define variational problem
    u = Function(V)
    v = TestFunction(V)

    # initial conditions space
    u_n = Function(V)
    u_n.interpolate(u0)

    # define constants used in variational problem
    # unknowns are [u_A, u_I] and corresponding test function [v_A, v_I]
    k = Constant(dt)
    D = Constant(param_set["D"])
    d = Constant(param_set["d"])

    F = ((u - u_n) / k)*v*dx + D*dot(grad(u), grad(v))*dx + d*u*v*dx 
        
    if boundary_conditions=="Mixed":
        bc = DirichletBC(V, 1, boundcond.boundary)
    elif boundary_conditions=="Neumann": 
        bc = None

    # time stepping
    num_steps = int(T/dt)
    t = 0

    # files for saving # to save for t=0
    if type(save_file)==str:
        vtkfile_A = File(f"{save_file}/A.pvd")
        vtkfile_A << (u_n, t)

    for n in range(num_steps):

        # solve variational problem
        solve(F==0, u, bcs=bc)

        # update previous solution
        u_n.assign(u)

        t += dt

        if type(save_file)==str:
            if round(t/save_step, 5).is_integer():
                vtkfile_A << (u, t)

    return u_n