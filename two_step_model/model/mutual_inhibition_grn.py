import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp
from sympy.utilities.lambdify import lambdify

def solve_grn(U, time_array, A, alpha0, beta0, S0, T0, S1):
    """
    Solves GRN ODE across time given parameters
    """
    solution = odeint(ode, U, time_array, args=(A, alpha0, beta0, S0, T0, S1))
    T_result, S_result = solution[:, 0], solution[:, 1]
    return T_result, S_result

def ode(U, t, A, alpha0, beta0, S0, T0, S1):
    """
    ODE of mutually inhibiting SOX2 and TBXT,
    with A as input and self_activating SOX2
    """
    T, S = U

    T_activators = {"A":A}
    T_repressors = {"ST":S/S0}
    dTdt = alpha0 * hill_function(T_activators, T_repressors) - beta0 * T

    S_activators = {"ST":S/S1}
    S_repressors = {"TS":T/T0}
    dSdt = hill_function(S_activators, S_repressors) - S

    return [dTdt, dSdt]

def move_sympyplot_to_axes(x, y, equation, ax, x_space, y_space):
    X, Y = np.meshgrid(x_space, y_space)
    f_lambdify = lambdify((x, y), equation.lhs - equation.rhs, 'numpy')
    Z = f_lambdify(X, Y)
    contour_level = 0
    ax.contour(X, Y, Z, levels=[contour_level], colors='cyan')
    # backend = p.backend(p)
    # backend.ax = ax
    # backend._process_series(backend.parent._series, ax, backend.parent)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_position('zero')
    # plt.close(backend.fig)

def T_nullcline(S_range, A, alpha0, beta0, S0, T0, S1):
    """Function for dTdt=0"""
    T_activators = {"A":A}
    T_repressors = {"ST":S_range/S0}
    T_stable = (alpha0/beta0) * hill_function(T_activators, T_repressors)
    return T_stable

def plot_ode_nullclines(ax,
                        T_range, 
                        S_range, 
                        A, 
                        alpha0, 
                        beta0, 
                        S0, 
                        T0,
                        S1,
                        color="Red"):
    
    args = (alpha0, beta0, S0, T0, S1)
    S_space = np.linspace(S_range[0], S_range[1], 100)
    T_space = np.linspace(T_range[0], T_range[1], 100)
    T_stable = T_nullcline(S_space, A, *args)
    ax.plot(T_stable, S_space, color=color)

    T, S = sp.symbols('T S')
    equation = sp.Eq(S, (S/S1)**2 / ((T/T0)**2 + (S/S1)**2 + 1)) # Equation for dSdt=0
    # sympy_plot = sp.plot_implicit(sp.Eq(equation, 0), (T, T_range[0], T_range[1]), (S, S_range[0], S_range[1]), 
    #                               line_width=10**5,
    #                               show=True,
    #                               line_color=color)
    move_sympyplot_to_axes(T, S, equation, ax, T_space, S_space)

def ode_trajectories(ode, T_range, S_range, A, args, plot, ax=None):
    T, S = np.mgrid[T_range[0]:T_range[1]:20j, S_range[0]:S_range[1]:20j]
    U = (T, S)
    ODE_func = ode(U, 0, A, *args)
    dTdt = ODE_func[0]
    dSdt = ODE_func[1]
    M = (np.hypot(dTdt, dSdt))
    M[M == 0] = 1 # This is to avoid any divisions when normalizing
    dTdt/=M
    dSdt/=M

    if plot==True:
        q = ax.quiver(T, S, dTdt, dSdt, M, pivot='mid', cmap=plt.cm.plasma, width=0.005)
        return dTdt, dSdt, M, q
    else:
        return dTdt, dSdt, M

def hill_function(activators, repressors, a_coeff=2, r_coeff=2):
    """
    Compute value of hill function given the following inputs:
    Inputs:
        Activator values (dict): contains all values of activator concentration / their hill constant  
        Inhibitor values (dict): contains all values of inhibitor concentration / their hill constant  
        a_coeff (int): hill coefficient for activators
        r_coeff (int): hill coefficient for inhibitors
    Return:
        Value from Hill function
    """
    if len(activators.values()) == 0:
        nom = 1
        activators = 0
    else:
        nom = hill_sum([a**a_coeff for a in activators.values()])
        # print(nom)
        activators = nom

    repressors = hill_sum([r**r_coeff for r in repressors.values()])
    # print(repressors)
    denom = 1 + activators + repressors
    # print(denom)
    return nom/denom

def hill_sum(terms_in_list):
    """ Function to calculate sum of terms in hill function given list of terms"""
    array_indexes = [i for i, element in enumerate(terms_in_list) if isinstance(element, np.ndarray)]
    number_indexes = [i for i, element in enumerate(terms_in_list) if not isinstance(element, np.ndarray)]
    variable_sum = np.sum([terms_in_list[array_indexes[i]] for i in range(len(array_indexes))], 
                           axis=0)
    fixed_sum = np.sum([terms_in_list[number_indexes[i]] for i in range(len(number_indexes))])
    sum = variable_sum + fixed_sum
    return sum
