import import_helper
import_helper.add_models()
import model.conditions as ic
import model.nondim_kondo_model as nondim_kondo
import model.mesh_generation as meshgen
import model.mutual_inhibition_grn as grn

import os
from fenics import *
from mshr import *
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import pymc as mc
import arviz as az
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from scipy.optimize import approx_fprime

wnt_simulations = pd.read_csv("../mcmc/simulation_intensity_mcmc.csv")
def model(params):
    gamma, S0, T0, S1 = params

    # reaction_diffusion model to get Wnt profile
    # wnt_profile = rd_model(gamma, x)
    wnt_profile = wnt_simulations[f"gamma{int(gamma)}"].to_numpy()
    # solve ode to get TBXT profile
    TBXT_profile = ode_model(S0, T0, S1, wnt_profile)

    return TBXT_profile

def rd_model(gamma, radius):
    rd_param = {"delta":20, "gamma":gamma, "kappa":2,
                 "Amax_nd":2, "Imax_nd":5, 
                 "aA":8/3, "bA":-8/3, "cA_nd":1/18, "dA":1,
                 "aI":10/3, "bI":0, "cI_nd":-1/6}
    u0 = ic.InitialConditionsRandom(noise=0.001, Ac=0.1, Ic=0.08333)
    mesh = meshgen.circle(R=1, res=30)
    rd_steady_state = nondim_kondo.dimensionless_kondo(mesh, 
                                                       u0,
                                                       T=1, 
                                                       dt=0.1,
                                                       param_set=rd_param,
                                                       boundary_conditions="Mixed")
    A, I = split(rd_steady_state)
    wnt_profile = np.array([A([0, r]) for r in radius[:-1]] + [1])
    return wnt_profile

def ode_model(S0, T0, S1, A_profile):
    t_span = np.linspace(0, 10, 100)
    U0 = [0, 0.2]
    tbxt_profile = np.zeros(len(A_profile))
    for i in range(len(A_profile)):
        solution = odeint(grn.ode, U0, t_span, args=(A_profile[i], 2, 1, S0, T0, S1))
        tbxt_profile[i] = solution[-1, 0]
    return tbxt_profile

def my_loglike(params, sigma, data):
    sim_model = model(params)
    return -0.5 * np.sum(((data - sim_model) / sigma)**2)

class LogLike(Op):
    def make_node(self, gamma, S0, T0, S1, sigma, data) -> Apply:
        # Convert inputs to tensor variables
        gamma = pt.as_tensor(gamma)
        S0 = pt.as_tensor(S0)
        T0 = pt.as_tensor(T0)
        S1 = pt.as_tensor(S1)
        sigma = pt.as_tensor(sigma)
        data = pt.as_tensor(data)

        inputs = [gamma, S0, T0, S1, sigma, data]
        outputs = [sigma.type()] # define output type

        return Apply(self, inputs, outputs)  # Apply is an object that combines inputs, outputs and an Op (self)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # This is the method that compute numerical output
        # given numerical inputs. Everything here is numpy arrays
        gamma, S0, T0, S1, sigma, data = inputs  # this will contain my variables

        # call our numpy log-likelihood function
        loglike_eval = my_loglike((gamma, S0, T0, S1), sigma, data)

        # Save the result in the outputs list provided by PyTensor
        # There is one list per output, each containing another list
        # pre-populated with a `None` where the result should be saved.
        outputs[0][0] = np.asarray(loglike_eval)

loglike_op = LogLike()
def custom_dist_loglike(data, gamma, S0, T0, S1, sigma):
    return loglike_op(gamma, S0, T0, S1, sigma, data)

if __name__=="__main__":
    results_folder = "250um_fit/results_1"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    data_path = "../mcmc_fit/mcmc_data/tbxt_circle_250um_mcmc.csv"
    dataframe = pd.read_csv(data_path)
    data = dataframe[dataframe.columns[2:]].mean(axis=1).to_numpy()
    data_err = dataframe[dataframe.columns[2:]].std(axis=1).to_numpy().mean()

    # n_radius = 10
    # radius_array = np.linspace(0, 1, n_radius+1)[1:]
    # full_data =(radius_array, data, data_err)

    with mc.Model() as model_fit:
        # specify my prior distribution
        gamma = mc.TruncatedNormal('gamma', mu=6, sigma=10, lower=5, upper=150)
        S0 = mc.TruncatedNormal('S0', mu=1.0, sigma=0.05, lower=0.0, upper=2.0)
        T0 = mc.TruncatedNormal('T0', mu=0.07, sigma=0.05, lower=0.0, upper=1.0)
        S1 = mc.TruncatedNormal('S1', mu=0.15, sigma=0.05, lower=0.0, upper=0.5)
        # # specify my proposal distribution
        # step_gamma = mc.Metropolis([gamma], S=np.array([2]))
        # step_S0 = mc.Metropolis([S0], S=np.array([0.05]))
        # step_T0 = mc.Metropolis([T0], S=np.array([0.05]))
        # step_S1 = mc.Metropolis([S1], S=np.array([0.05]))

        Likelihood = mc.CustomDist('likelihood', 
                                   *(gamma, S0, T0, S1), 
                                   data_err,
                                   observed=data, 
                                   logp=custom_dist_loglike)

        trace = mc.sample(2000, 
                          tune=500, 
                          chains=4,
                          cores=4,
                          return_inferencedata=True)

    # trace plots
    fig = az.plot_trace(trace, compact=True)
    plt.savefig(os.path.join(results_folder, "trace_plot.png"))

    # saving traces
    az.to_netcdf(trace, os.path.join(results_folder, "trace_results.nc"))

    # save summary of trace
    df = az.summary(trace)
    df.to_csv(os.path.join(results_folder, "mcmc_results.csv"))

    with model_fit:
        map_estimate = mc.find_MAP()
    with open(os.path.join(results_folder, "map_estimate.txt"), 'w') as f:
        f.write(str(map_estimate))