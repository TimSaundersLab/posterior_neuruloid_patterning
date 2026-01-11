import numpy as np
import math
from tqdm import tqdm
import utils as u
import spde_utils as spde
# functions from Scoones et al 2020

# default parameters
DS = 0.05
DP = 1
KOFF = 1
KON = 20*KOFF
N_F = 2
S0 = np.sqrt(10)

DT = 0.001
T_STEPS = 100000
DX = 0.05

def f(S, n, S0):
    """Sigmoid function for feedback of substrate"""
    return S**n/(S0**n + S**n)

def solve_SD_model(Si, 
                   Pi,
                   N, 
                   dt=DT, 
                   dx=DX, 
                   t_steps=T_STEPS, 
                   D_s=DS, 
                   D_p=DP, 
                   kon=KON, 
                   koff=KOFF, 
                   n=N_F, 
                   S0=S0):

    """Substrate depletion model from equations 1,2"""
    St = np.zeros((len(Si), t_steps+1))
    Pt = np.zeros((len(Pi), t_steps+1))
    St[:, 0] = Si
    Pt[:, 0] = Pi

    for t in tqdm(range(1, t_steps+1)):
        S = St[:, t-1]
        P = Pt[:, t-1]
        dSdt = np.zeros_like(S)
        dPdt = np.zeros_like(P)

        # reaction-diffusion
        for i in range(1, len(S) - 1):
            dSdt[i] = D_s * (S[i + 1] - 2 * S[i] + S[i - 1]) / dx**2 + kon * P[i] * f(S[i], n, S0) - koff * S[i]
            dPdt[i] = D_p * (P[i + 1] - 2 * P[i] + P[i - 1]) / dx**2 - kon * P[i] * f(S[i], n, S0) + koff * S[i] 
        dSdt[0] = D_s * (S[1] - 2 * S[0] + S[-1]) / dx**2 + kon * P[0] * f(S[0], n, S0) - koff * S[0]
        dPdt[0] = D_p * (P[1] - 2 * P[0] + P[-1]) / dx**2 - kon * P[0] * f(S[0], n, S0) + koff * S[0]
        dSdt[-1] = D_s * (S[0] - 2 * S[-1] + S[-2]) / dx**2 + kon * P[-1] * f(S[-1], n, S0) - koff * S[-1]
        dPdt[-1] = D_p * (P[0] - 2 * P[-1] + P[-2]) / dx**2 - kon * P[-1] * f(S[-1], n, S0) + koff * S[-1]
        Snew = S + dSdt * dt
        Pnew = P + dPdt * dt
        Snorm = Snew * N / (dx*np.sum(S + P))
        Pnorm = Pnew * N / (dx*np.sum(S + P))
        
        # update matrix
        St[:, t] = Snorm
        Pt[:, t] = Pnorm
    return St, Pt

def imex_SD(Si, 
            Pi,
            N, 
            dt=DT, 
            dx=DX, 
            t_steps=T_STEPS, 
            D_s=DS, 
            D_p=DP, 
            kon=KON, 
            koff=KOFF, 
            n=N_F, 
            S0=S0):

    St = np.zeros((len(Si), t_steps+1))
    Pt = np.zeros((len(Pi), t_steps+1))
    St[:, 0] = Si
    Pt[:, 0] = Pi

    # set up matrix on LHS
    Nx = len(Si)
    As = np.zeros((Nx, Nx))
    Ap = np.zeros((Nx, Nx))
    for i in range(Nx):
        As[i, i] = 1 + 2*D_s*dt/(dx**2)
        As[i, (i+1)%Nx] = - D_s*dt/dx**2
        As[i, (i-1)%Nx] = - D_s*dt/dx**2
        Ap[i, i] = 1 + 2*D_p*dt/(dx**2)
        Ap[i, (i+1)%Nx] = - D_p*dt/dx**2
        Ap[i, (i-1)%Nx] = - D_p*dt/dx**2

    for t in range(1, t_steps+1) :
        S = St[:, t-1]
        P = Pt[:, t-1]

        # set up RHS
        S_rhs = S + dt*(kon*P*f(S, n, S0) - koff*S)
        P_rhs = P - dt*(kon*P*f(S, n, S0) - koff*S)

        # solve 
        Snew = np.linalg.solve(As, S_rhs)
        Pnew = np.linalg.solve(Ap, P_rhs)

        Snorm = Snew * N / (dx*np.sum(Snew + Pnew))
        Pnorm = Pnew * N / (dx*np.sum(Snew + Pnew))

        # update solutions
        St[:, t] = Snorm
        Pt[:, t] = Pnorm

    return St, Pt

def dFdP(S, kon, S0, n):
    """Derivative of reaction function (equation 3)"""
    return (kon*S**n)/((S0**n) + (S**n))

def dFdS(S, P, kon, koff, S0, n):
    """Derivative of reaction function (equation 3)"""
    term1 = ((S0**n) + (S**n))*P*n*kon*S**(n-1)
    term2 = ((S0**n) + (n*S**(n-1)))*P*kon*S**n
    term3 = ((S0**n) + (S**n))**2
    return ((term1-term2)/term3) - koff

def order_parameter(S):
    """Sum of absolute gradient of S(x), substrate concentration"""
    grad = abs(np.gradient(S))
    order = np.sum(grad)
    return order

# stochastic model
def rate_of_reactions(S,
                      P,
                      dx,
                      D_s=DS, 
                      D_p=DP, 
                      kon=KON, 
                      koff=KOFF, 
                      n=N_F, 
                      S0=S0):
    """
    Calculate arrays consisting rates of reactions
    Return:
        P2S: (array) Rates of association of P to S
        S2P: (array) Rates of dissociation of S to P
        S_diff: (array) Rates of diffusion of S
        P_diff: (array) Rates of diffusion of P
    """
    fS = f(S/dx, n, S0)
    P2S = kon*fS*P
    S2P = koff*S
    S_diff = D_s*S/(dx**2)
    P_diff = D_p*P/(dx**2)
    return P2S, S2P, S_diff, P_diff

def gillespie_SD(Si, 
                 Pi, 
                 T_final,
                 dx,
                 D_s=DS, 
                 D_p=DP, 
                 kon=KON, 
                 koff=KOFF, 
                 n=N_F, 
                 S0=S0):
    """
    Compartmental Gillespie algorithm
    Si: (array) number of substrate in each compartment
    Pi: (array) number of pool molecules in each compartment
    T_final: (int) final time of simulation
    """
    S = np.copy(Si)
    P = np.copy(Pi)
    Nx = len(Si)
    t = 0
    while t < T_final:
        r1, r2 = np.random.rand(2)
        PS, SP, S_diff, P_diff = rate_of_reactions(S,
                                                   P,
                                                   dx,
                                                   D_s, 
                                                   D_p, 
                                                   kon, 
                                                   koff, 
                                                   n, 
                                                   S0)
        Sright = S_diff
        Sleft = S_diff
        Pright = P_diff
        Pleft = P_diff

        all_rates = np.concatenate([PS, SP, Sright, Sleft, Pright, Pleft])
        Rtot = np.sum(all_rates)
        probs = all_rates/Rtot
        dt = -np.log(r1)/Rtot
        
        reaction_index = u.find_index(r2, probs)
        reaction = math.floor(reaction_index / Nx)
        rxn_loc = reaction_index % Nx
        
        if reaction == 0:
            P[rxn_loc] -= 1
            S[rxn_loc] += 1
        
        elif reaction == 1:
            P[rxn_loc] += 1
            S[rxn_loc] -= 1
            
        elif reaction == 2:
            S[rxn_loc] -= 1
            S[(rxn_loc + 1)%Nx] += 1
            
        elif reaction == 3:
            S[rxn_loc] -= 1
            S[(rxn_loc - 1)%Nx] += 1

        elif reaction == 4:
            P[rxn_loc] -= 1
            P[(rxn_loc + 1)%Nx] += 1
        
        elif reaction == 5:
            P[rxn_loc] -= 1
            P[(rxn_loc - 1)%Nx] += 1    
            
        t += dt
        
    return S, P

# Explicit Euler-maruyama tau-leaping scheme
def substrate_depletion_EMTL(Si, 
                             Pi,
                             N,
                             Tf,
                             A=1000,
                             dt=DT,
                             dx=DX,
                             D_s=DS,
                             D_p=DP, 
                             kon=KON, 
                             koff=KOFF, 
                             n=N_F, 
                             S0=S0):
    """
    Solving substrate depletion model with SPDE Explicit Euler-Maruyama Tau-Leaping (EMTL) method
    Inputs:
        Si (array): initial concentration of substrate molecules
        Pi (array): initial concentration of pool molecules
        N (int): total number of molecules
    """
    dv = A*dx
    time = np.arange(0, Tf+dt, dt)
    t_steps = len(time)

    nx = len(Si)
    St = np.zeros((nx, t_steps+1))
    Pt = np.zeros((nx, t_steps+1))
    St[:, 0] = Si
    Pt[:, 0] = Pi
    
    for i in range(1, t_steps+1):
        S = St[:, i-1]
        P = Pt[:, i-1]

        # run EMTL scheme
        S_diff, S_diff_noise = spde.explicit_diffusion_spde(S, D_s, dx, dv, dt, mode="wrap")
        P_diff, P_diff_noise = spde.explicit_diffusion_spde(P, D_p, dx, dv, dt, mode="wrap")

        P_to_S = spde.poisson_event(kon*P*(P>0)*f(S, n, S0), dv, dt)
        S_to_P = spde.poisson_event(koff*S*(S>0), dv, dt)

        Snew = S + S_diff + S_diff_noise + P_to_S - S_to_P
        Pnew = P + P_diff + P_diff_noise - P_to_S + S_to_P

        # rescale so that total number of molecules is N
        Snorm = Snew * N / (dx*np.sum(Snew + Pnew))
        Pnorm = Pnew * N / (dx*np.sum(Snew + Pnew))

        # update concentration
        St[:, i] = Snorm
        Pt[:, i] = Pnorm
    
    return St, Pt

# Implicit midpoint tau leaping scheme
def substrate_depletion_IMTL(Si, 
                             Pi,
                             N,
                             Tf,
                             A=1000,
                             dt=DT,
                             dx=DX,
                             D_s=DS,
                             D_p=DP,
                             kon=KON, 
                             koff=KOFF, 
                             n=N_F, 
                             S0=S0):
    """
    Solving substrate depletion model with SPDE Implicit Midpoint Tau-Leaping (IMTL) method
    Inputs:
        Si (array): initial concentration of substrate molecules
        Pi (array): initial concentration of pool molecules
        N (int): total number of molecules
    """
    time = np.arange(0, Tf+dt, dt)
    t_steps = len(time)

    nx = len(Si)
    dv = A*dx
    St = np.zeros((nx, t_steps))
    Pt = np.zeros((nx, t_steps))
    St[:, 0] = Si
    Pt[:, 0] = Pi

    # lhs for corrector step
    ASnew = np.zeros((nx, nx))
    APnew = np.zeros((nx, nx))
    for i in range(nx):
        ASnew[i, i] = 1 + D_s*dt/(dx**2)
        ASnew[i, (i+1)%nx] = - D_s*dt/(2*dx**2)
        ASnew[i, (i-1)%nx] = - D_s*dt/(2*dx**2)

        APnew[i, i] = 1 + D_p*dt/(dx**2)
        APnew[i, (i+1)%nx] = - D_p*dt/(2*dx**2)
        APnew[i, (i-1)%nx] = - D_p*dt/(2*dx**2)
    
    for i in range(1, t_steps):
        S = St[:, i-1]
        P = Pt[:, i-1]

        # run IMTL scheme
        # PREDICTOR STEP
        # diffusion
        AS_star, diff_noise_S1 = spde.implicit_diffusion_spde(S, D_s, dx, dv, dt/2, mode="wrap")
        AP_star, diff_noise_P1 = spde.implicit_diffusion_spde(P, D_p, dx, dv, dt/2, mode="wrap")

        # reactions
        P_to_S_predict = spde.poisson_event(kon*P*(P>0)*f(S, n, S0), dv, dt/2)
        S_to_P_predict = spde.poisson_event(koff*S*(S>0), dv, dt/2)

        # predictor
        S_star_rhs = S + diff_noise_S1 + P_to_S_predict - S_to_P_predict 
        S_star = np.linalg.solve(AS_star, S_star_rhs)

        P_star_rhs = P + diff_noise_P1 - P_to_S_predict + S_to_P_predict
        P_star = np.linalg.solve(AP_star, P_star_rhs)

        # CORRECTOR STEP
        S_dot = 2*S_star - S
        S_dot = S_dot*(S_dot>0)
        P_dot = 2*P_star - P
        P_dot = P_dot*(P_dot>0)

        # diffusion
        S_diff = spde.explicit_diffusion(S, D_s, dx, dt/2, mode="wrap")
        P_diff = spde.explicit_diffusion(P, D_p, dx, dt/2, mode="wrap")
        _, diff_noise_S2 = spde.explicit_diffusion_spde(S_dot, D_s, dx, dv, dt/2, mode="wrap")
        _, diff_noise_P2 = spde.explicit_diffusion_spde(P_dot, D_p, dx, dv, dt/2, mode="wrap")

        # reactions
        P_to_S_correct = spde.poisson_event(kon*P_dot*f(S_dot, n, S0), dv, dt/2)
        S_to_P_correct = spde.poisson_event(koff*S_dot, dv, dt/2)

        # rhs
        Snew_rhs = S + S_diff + diff_noise_S1 + diff_noise_S2 + P_to_S_predict + P_to_S_correct - S_to_P_predict - S_to_P_correct
        Pnew_rhs = P + P_diff + diff_noise_P1 + diff_noise_P2 - P_to_S_predict - P_to_S_correct + S_to_P_predict + S_to_P_correct

        # corrector
        Snew = np.linalg.solve(ASnew, Snew_rhs)
        Pnew = np.linalg.solve(APnew, Pnew_rhs)

        # rescale so that total number of molecules is N
        Snorm = Snew * N / (dx*np.sum(Snew + Pnew))
        Pnorm = Pnew * N / (dx*np.sum(Snew + Pnew))

        # update concentration
        St[:, i] = Snorm
        Pt[:, i] = Pnorm
    
    return St, Pt

