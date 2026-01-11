# contains useful functions for implementation of SPDEs

import numpy as np

def explicit_diffusion_spde(S, D, dx, dv, dt, mode="clip", noise=True):
    """
    One time step explicit diffusion with gaussian noise
    - if mode is "clip", hard boundary conditions, returns diffusion vector of size len(S)-2
    - if mode is "wrap", periodic boundary conditions, returns diffusion vector of size len(S)
    Return:
        deterministic diffusion (array)
        gaussian noise of diffusion (array)
    """
    nx = len(S)

    if mode=="clip":
        diff_vector = S[2:] - 2*S[1:-1] + S[:-2]
        diff_noise_vector = np.sqrt(modified_arithmetic_mean(S[1:-1], S[2:], dx))*np.random.normal(size=nx-2) - np.sqrt(modified_arithmetic_mean(S[1:-1], S[:-2], dx))*np.random.normal(size=nx-2)
    
    elif mode=="wrap":
        right_vector = np.insert(S[:-1], 0, S[-1])
        left_vector = np.insert(S[1:], -1, S[0])
        diff_vector = right_vector - 2*S + left_vector
        diff_noise_vector = np.sqrt(modified_arithmetic_mean(S, right_vector, dx))*np.random.normal(size=nx) - np.sqrt(modified_arithmetic_mean(S, left_vector, dx))*np.random.normal(size=nx)

    diff = (D*dt/(dx**2)) * diff_vector                        # deterministic part of diffusion
    diff_noise = np.sqrt(2*D*dt/dv)*(1/dx)*diff_noise_vector # noisy part of diffusion

    return diff, diff_noise

def implicit_diffusion_spde(S, D, dx, dv, dt, mode="clip"):
    """
    One time step implicit diffusion with gaussian noise
    - if mode is "clip", hard boundary conditions, returns diffusion vector of size len(S)-2
    - if mode is "wrap", periodic boundary conditions, returns diffusion vector of size len(S)
    Return:
        LHS: deterministic diffusion (matrix to solve implicitly)
        RHS: gaussian noise of diffusion (array)
    """
    nx = len(S)
    # set up implicit diffusion matrix on lhs
    As = np.zeros((nx, nx))
    if mode=="clip":
        for i in range(1, nx-1):
            As[i, i] = 1 + 2*D*dt/(dx**2)
            As[i, i+1] = - D*dt/dx**2
            As[i, i-1] = - D*dt/dx**2
        As[0, 0] = 1
        As[-1, -1] = 1
        diff_noise_vector = np.sqrt(modified_arithmetic_mean(S[1:-1], S[2:], dx))*np.random.normal(size=nx-2) - np.sqrt(modified_arithmetic_mean(S[1:-1], S[:-2], dx))*np.random.normal(size=nx-2)

    elif mode=="wrap":
        for i in range(nx):
            As[i, i] = 1 + 2*D*dt/(dx**2)
            As[i, (i+1)%nx] = - D*dt/dx**2
            As[i, (i-1)%nx] = - D*dt/dx**2
        right_vector = np.insert(S[:-1], 0, S[-1])
        left_vector = np.insert(S[1:], -1, S[0])
        diff_noise_vector = np.sqrt(modified_arithmetic_mean(S, right_vector, dx))*np.random.normal(size=nx) - np.sqrt(modified_arithmetic_mean(S, left_vector, dx))*np.random.normal(size=nx)

    diff_noise = np.sqrt(2*D*dt/dv)*(1/dx)*diff_noise_vector

    return As, diff_noise

def explicit_diffusion(S, D, dx, dt, mode="clip"):
    """
    One time step explicit deterministic diffusion
    - if mode is "clip", hard boundary conditions, returns diffusion vector of size len(S)-2
    - if mode is "wrap", periodic boundary conditions, returns diffusion vector of size len(S)
    Return:
        deterministic diffusion step (array)
    """
    
    if mode=="clip":
        diff_vector = S[2:] - 2*S[1:-1] + S[:-2]
    
    elif mode=="wrap":
        right_vector = np.insert(S[:-1], 0, S[-1])
        left_vector = np.insert(S[1:], -1, S[0])
        diff_vector = right_vector - 2*S + left_vector
    
    diff = (D*dt/(dx**2)) * diff_vector
    return diff

def implicit_diffusion(D, Nx, dx, dt, mode="clip"):
    """
    Set up LHS matrix for implicit diffusion for grid of length Nx
    - mode "clip" for other boundary conditions A[0,0]=1, A[-1,-1]=1
    - mode "wrap" for periodic boundary conditions
    """  
    A = np.zeros((Nx, Nx))

    if mode=="wrap":
        for i in range(Nx):
            A[i, i] = 1 + 2*D*dt/(dx**2)
            A[i, (i+1)%Nx] = - D*dt/dx**2
            A[i, (i-1)%Nx] = - D*dt/dx**2

    elif mode=="clip":
        for i in range(1, Nx-1):
            A[i, i] = 1 + 2*D*dt/(dx**2)
            A[i, i+1] = - D*dt/dx**2
            A[i, i-1] = - D*dt/dx**2
        A[0, 0] = 1
        A[Nx, Nx] = 1

    return A

def poisson_event(rates, dv, dt):
    """
    Simulate poisson variables given propensity (in concentration)
    Inputs:
        rates (array): poisson rate in RDME (units in concentration per time)
        dv (float): discretization of volume
        dt (float): discretization of time
    Return:
        Array of poisson random variables
    """
    po_mean = rates * dv * dt
    po_rv = np.random.poisson(po_mean)
    in_conc = po_rv / dv
    return  in_conc

def modified_arithmetic_mean(a, b, dv):
    """
    Arithmetic mean of molecule concentration between grids with volume size dv
    - modified with heaviside step function to be 0 when molecule concentration is small
    """
    return (1/2)*(a+b)*H0(dv*a)*H0(dv*b)

def H0(x):
    """
    Modified heaviside step function 
    - to shut down fluctuations when concentration of molecules is small
    """
    return np.clip(x, 0, 1)