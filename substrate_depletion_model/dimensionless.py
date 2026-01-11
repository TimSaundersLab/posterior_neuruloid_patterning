# Dimensionless parameters analysis
import substrate_depletion_model as sd
import utils as u
import numpy as np
from scipy.signal import argrelextrema

# simulation params
L = 10
dx = 0.05
x = np.arange(0, L+dx, dx)

dt = 0.1
T = 100
tsteps = int(T/dt)

# default parameters
k_ratio = 20 # k_on/k_off
k_off = 1 # TIME SCALE
k_on = k_ratio * k_off

# CONCENTRATION SCALE
average_conc = 3
N = average_conc*L 
S0 = 2*average_conc # np.sqrt(10)

# LENGTH SCALE
D_ratio = 0.05
D0 = (L**2)*k_off
Dk = 50 # np.arange(10, 100, 5) # sqrt(D0/DS)
Ds = D0/(Dk**2)
Dp = Ds/D_ratio

# initial conditions
Si = u.sinusoidal_perturbations(x, N, waves=5)
Pi = np.zeros_like(x)

def length_scale(Dk, 
                 L=L, 
                 k_off=k_off, 
                 k_ratio=k_ratio,
                 D_ratio=D_ratio,
                 Tfinal=T,
                 dt=dt,
                 dx=dx,
                 n=2,
                 avg_conc = average_conc,
                 S0=S0):
    
    """
    Experiments on length by varying diffusion coefficients, Dk=sqrt(D0/Ds)
    Returns:
        npeaks: (array) the number of local maxima for each Dk after solving system for Tfinal
    """
    D0 = (L**2)*k_off
    Ds = D0/(Dk**2)
    Dp = Ds/D_ratio

    k_on = k_ratio*k_off

    tsteps = int(Tfinal/dt)
    x = np.arange(0, L+dx, dx)

    N = avg_conc*L
    Si = u.sinusoidal_perturbations(x, N, waves=5)
    Pi = np.zeros_like(x)

    npeak = np.zeros(len(Dk))

    for i in range(len(Dk)):
        St, Pt = sd.imex_SD(Si,
                            Pi,
                            N=N,
                            dt=dt,
                            dx=dx,
                            t_steps=tsteps,
                            D_s=Ds[i],
                            D_p=Dp[i],
                            kon=k_on,
                            koff=k_off,
                            n=n,
                            S0=S0)
        Sfinal = St[:, -1]
        npeak[i] = len(argrelextrema(Sfinal, np.greater, mode='wrap')[0])

    return npeak

def concentration_scale(S0_conc,
                        L=L, 
                        avg_conc=average_conc,
                        k_off=k_off, 
                        k_ratio=k_ratio,
                        Dk=Dk,
                        D_ratio=D_ratio,
                        Tfinal=T,
                        dt=dt,
                        dx=dx,
                        n=2):
    """
    Experiments on ratio of concentration to S0, S0/(N/L)
    Returns:
        npeaks: (array) the number of local maxima for each conc_S0 ratio after solving system for Tfinal
        Sc_final: (array) substrate concentration in domain after time Tfinal
        order: (array) sum of the absolute substrate gradient in space
    """
    N = avg_conc * L
    S0 = S0_conc * avg_conc # array
    k_on = k_off * k_ratio

    D0 = (L**2)*k_off
    Ds = D0/(Dk**2)
    Dp = Ds/D_ratio

    tsteps = int(Tfinal/dt)
    x = np.arange(0, L+dx, dx)

    Si = u.sinusoidal_perturbations(x, N, waves=5)
    Pi = np.zeros_like(x)

    npeak = np.zeros(len(S0_conc))
    Sc_final = np.zeros(len(S0_conc))
    order = np.zeros(len(S0_conc))

    for i in range(len(S0_conc)):
        St, Pt = sd.imex_SD(Si,
                            Pi,
                            N=N,
                            dt=dt,
                            dx=dx,
                            t_steps=tsteps,
                            D_s=Ds,
                            D_p=Dp,
                            kon=k_on,
                            koff=k_off,
                            n=n,
                            S0=S0[i])
        
        Sfinal = St[:, -1]
        npeak[i] = len(argrelextrema(Sfinal, np.greater, mode='wrap')[0])
        Sc_final[i] = np.sum(Sfinal*dx)/N
        order[i] = sd.order_parameter(Sfinal)

    return npeak, Sc_final, order 

def association_dissociation_scale(k_ratio, 
                                   k_off=k_off,
                                   L=L,
                                   Dk=Dk,
                                   D_ratio=D_ratio,
                                   Tfinal=T,
                                   dt=dt,
                                   dx=dx,
                                   n=2,
                                   avg_conc=average_conc,
                                   S0=S0):
    """
    Experiments on ratio of k_on to k_off, k_ratio=k_on/k_off
    Returns:
        npeaks: (array) the number of local maxima for each k_ratio after solving system for Tfinal
        Sc_final: (array) P substrate concentration in domain after time Tfinal
    """
    k_on = k_off * k_ratio

    D0 = (L**2)*k_off
    Ds = D0/(Dk**2)
    Dp = Ds/D_ratio

    tsteps = int(Tfinal/dt)
    x = np.arange(0, L+dx, dx)

    N = avg_conc*L
    Si = u.sinusoidal_perturbations(x, N, waves=5)
    Pi = np.zeros_like(x)

    npeak = np.zeros(len(k_ratio))
    S_prop = np.zeros(len(k_ratio))
    order = np.zeros(len(k_ratio))

    for i in range(len(k_ratio)):
        St, Pt = sd.imex_SD(Si,
                            Pi,
                            N=N,
                            dt=dt,
                            dx=dx,
                            t_steps=tsteps,
                            D_s=Ds,
                            D_p=Dp,
                            kon=k_on[i],
                            koff=k_off,
                            n=n,
                            S0=S0)
        Sfinal = St[:, -1]
        npeak[i] = len(argrelextrema(Sfinal, np.greater, mode='wrap')[0])
        S_prop[i] = np.sum(Sfinal*dx)/N
        order[i] = sd.order_parameter(Sfinal)

    return npeak, S_prop, order

def diffusion_coefficient_scale(D_ratio,
                                Dk=Dk,
                                L=L,
                                k_ratio=k_ratio,
                                k_off=k_off,
                                Tfinal=T,
                                dt=dt,
                                dx=dx,
                                n=2,
                                avg_conc=average_conc,
                                S0=S0):
    
    """
    Experiments on ratio of D_p to D_s, D_ratio=D_p/D_s
    Returns:
        npeaks: (array) the number of local maxima for each D_ratio after solving system for Tfinal
        Sc_final: (array) P substrate concentration in domain after time Tfinal
    """

    D0 = (L**2)*k_off
    Ds = D0/(Dk**2)
    Dp = Ds/D_ratio

    k_on = k_off * k_ratio

    tsteps = int(Tfinal/dt)
    x = np.arange(0, L+dx, dx)

    N = avg_conc*L
    Si = u.sinusoidal_perturbations(x, N, waves=5)
    Pi = np.zeros_like(x)

    npeak = np.zeros(len(D_ratio))
    S_prop = np.zeros(len(D_ratio))
    order = np.zeros(len(D_ratio))

    for i in range(len(D_ratio)):
        St, Pt = sd.imex_SD(Si,
                            Pi,
                            N=N,
                            dt=dt,
                            dx=dx,
                            t_steps=tsteps,
                            D_s=Ds,
                            D_p=Dp[i],
                            kon=k_on,
                            koff=k_off,
                            n=n,
                            S0=S0)
        Sfinal = St[:, -1]
        npeak[i] = len(argrelextrema(Sfinal, np.greater, mode='wrap')[0])
        S_prop[i] = np.sum(Sfinal*dx)/N
        order[i] = sd.order_parameter(Sfinal)

    return npeak, S_prop, order


def D_ratio_Dk(filename, T):

    FOLDER = "parameter_results"
    Dk = np.arange(10, 100, 5) # sqrt(D0/DS)
    D_ratio = np.linspace(0.01, 0.6, 15) # Dp/Ds
    results = np.zeros((len(D_ratio), len(Dk)))
    Sprop = np.zeros((len(D_ratio), len(Dk)))
    order = np.zeros((len(D_ratio), len(Dk)))

    for i in range(len(Dk)):
        npeak_D, Spropfinal, order_param = diffusion_coefficient_scale(D_ratio,
                                                                    Dk=Dk[i],
                                                                    L=L,
                                                                    k_ratio=k_ratio,
                                                                    k_off=k_off,
                                                                    Tfinal=T,
                                                                    dt=dt,
                                                                    dx=dx,
                                                                    n=2,
                                                                    avg_conc=average_conc,
                                                                    S0=S0)
        results[:, i] = npeak_D
        Sprop[:, i] = Spropfinal
        order[:, i] = order_param
    
    with open(f"{FOLDER}/{filename}.npy", "wb") as file:
        np.save(file, results, allow_pickle=True)

    with open(f"{FOLDER}/{filename}_Sconc.npy", "wb") as file:
        np.save(file, Sprop, allow_pickle=True)

    with open(f"{FOLDER}/{filename}_order.npy", "wb") as file:
        np.save(file, order, allow_pickle=True)

def k_ratio_Dk(filename, T):
    
    FOLDER = "parameter_results"
    Dk = np.arange(10, 100, 5) # sqrt(D0/DS)
    k_ratio = np.arange(10, 40, 2) # k_on/k_off
    results = np.zeros((len(k_ratio), len(Dk)))
    Sprop = np.zeros((len(k_ratio), len(Dk)))
    order = np.zeros((len(k_ratio), len(Dk)))

    for i in range(len(Dk)):
        npeak_k, Spropfinal, order_param = association_dissociation_scale(k_ratio,
                                                                        k_off=k_off,
                                                                        L=L,
                                                                        Dk=Dk[i],
                                                                        D_ratio=D_ratio,
                                                                        Tfinal=T,
                                                                        dt=dt,
                                                                        dx=dx,
                                                                        n=2,
                                                                        avg_conc=average_conc,
                                                                        S0=S0)
        results[:, i] = npeak_k
        Sprop[:, i] = Spropfinal
        order[:, i] = order_param
    
    with open(f"{FOLDER}/{filename}.npy", "wb") as file:
        np.save(file, results, allow_pickle=True)

    with open(f"{FOLDER}/{filename}_Sconc.npy", "wb") as file:
        np.save(file, Sprop, allow_pickle=True)

    with open(f"{FOLDER}/{filename}_order.npy", "wb") as file:
        np.save(file, order, allow_pickle=True)

def S0_conc_Dk(filename, T):

    FOLDER = "parameter_results"
    Dk = np.arange(10, 100, 5) # sqrt(D0/DS)
    # S0_to_conc = np.arange(0.5, 8.0, 0.5) # S0/(N/L)
    S0_to_conc = np.arange(1.1, 2.6, 0.1) # S0/(N/L) zoom in
    results = np.zeros((len(S0_to_conc), len(Dk)))
    Sprop_final = np.zeros((len(S0_to_conc), len(Dk)))
    order = np.zeros((len(S0_to_conc), len(Dk)))

    for i in range(len(Dk)):
        npeak, Sprop, order_param = concentration_scale(S0_to_conc,
                                                        L=L, 
                                                        avg_conc=average_conc,
                                                        k_off=k_off, 
                                                        k_ratio=k_ratio,
                                                        Dk=Dk[i],
                                                        D_ratio=D_ratio,
                                                        Tfinal=T,
                                                        dt=dt,
                                                        dx=dx,
                                                        n=2)
        results[:, i] = npeak
        Sprop_final[:, i] = Sprop
        order[:, i] = order_param   

    with open(f"{FOLDER}/{filename}.npy", "wb") as file:
        np.save(file, results, allow_pickle=True)

    with open(f"{FOLDER}/{filename}_Sconc.npy", "wb") as file:
        np.save(file, Sprop_final, allow_pickle=True)

    with open(f"{FOLDER}/{filename}_order.npy", "wb") as file:
        np.save(file, order, allow_pickle=True)        

if __name__=="__main__":
    S0_conc_Dk(filename="S0_to_conc_zoom_vs_Dk_T100", T=100)
    S0_conc_Dk(filename="S0_to_conc_zoom_vs_Dk_T300", T=300)

    # D_ratio_Dk(filename="D_ratio_wider_vs_Dk_T100", T=100)
    # D_ratio_Dk(filename="D_ratio_wider_vs_Dk_T300", T=300)
    # k_ratio_Dk(filename="k_ratio_vs_Dk_T100", T=100)
    # k_ratio_Dk(filename="k_ratio_vs_Dk_T300", T=300)

