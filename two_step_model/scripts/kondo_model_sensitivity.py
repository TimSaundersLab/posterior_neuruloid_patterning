import import_helper
import_helper.add_models()
import model.conditions as ic
import model.nondim_kondo_model as nondim_kondo
import model.mesh_generation as meshgen

import os 
import gc
from datetime import datetime
from fenics import *
from mshr import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# params for PDE RD model
Tend = 3
dt = 0.0005
mesh = meshgen.circle(R=1, res=30)
# triangle_length = np.sqrt(4*np.pi/np.sqrt(3))
# mesh = meshgen.equilateral_triangle(length=triangle_length, res=30)
# square_length = np.sqrt(np.pi)
# mesh = meshgen.square(length=square_length, res=30)

# function to solve PDE given all parameters
def solve_kondo_model(IC, 
                      Tend,
                      dt,
                      parameters,
                      file_name,
                      intensity_df,
                      column_name=False):
    """
    Args:
        IC: initial conditions
        R (float): radius of circular mesh
        Tend (int): final time of simulation
        dt (float): time step for simulation
        file_name (str): file path for saving figure
    """
    # solve 
    rd_solution = nondim_kondo.dimensionless_kondo(mesh,
                                                   IC,
                                                   T=Tend,
                                                   dt=dt,
                                                   param_set=parameters,
                                                   boundary_conditions="Mixed")
    A, I = split(rd_solution)

    if type(column_name)==str:
        x_coord = np.linspace(0, 1, 100)[:-1]
        intensity = np.array([A([0,x_coord[i]]) for i in range(len(x_coord))])
        intensity = np.append(intensity, 1)
        intensity_df[column_name] = intensity

    # save activator concentration in file
    plot_and_savefig(A, file_name)
    del(rd_solution)
    gc.collect()

def plot_and_savefig(concentration, file_name): 
    fig = plt.figure(figsize=(5, 5))
    im = plot(concentration, cmap="magma")
    plt.colorbar(im)
    fig.savefig(file_name)
    plt.close()

# A_boundary = np.arange(0, 2.5, 0.5)*Aeq
# def run_for_many_Aboundary(radius, 
#                            folder=FOLDER):
#     for i in tqdm(range(len(A_boundary))):
#         filename = os.path.join(folder, f"A{A_boundary[i]}_R{radius}.png")
#         new_param = param_dict.copy()
#         new_param["Abound"] = A_boundary[i]
#         solve_kondo_model(u0,
#                           radius,
#                           Tend,
#                           dt,
#                           filename,
#                           parameters=new_param)
        
if __name__=="__main__":
    print(datetime.now())

    FOLDER = "vary_Aeq_gamma_plots"
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
    # intensity_file = os.path.join(FOLDER, "intensity_df.csv")
    # make pandas df
    intensity_df = pd.DataFrame()

    # fixed parameters
    delta = 20
    kappa = 2
    Amax_nd = 2
    Imax_nd = 5
    aA = 8/3
    bA = -8/3
    cA_default = 1/3
    # cA_nd = 1/18
    dA = 1
    aI = 10/3
    bI = 0
    cI_default = -1
    # cI_nd = -1/6
    # Aeq_nd = 0.1
    # Ieq_nd = 0.0833
    Aeq_default = 0.6
    Ieq_default = 0.5

    # varied parameters
    gamma = np.linspace(2, 20, 10)**2
    # delta = np.arange(50, 110, 10)
    # varied parameters
    Aeq_nd = np.array([0.8]) #np.arange(0, 2.2, 0.2)
    Ieq_nd = Ieq_default*(Aeq_nd/Aeq_default)
    cA_nd = cA_default*(Aeq_nd/Aeq_default)
    cI_nd = cI_default*(Aeq_nd/Aeq_default)

    # iterate over two params
    for i in range(len(Aeq_nd)):
        for j in tqdm(range(len(gamma))):
            non_dim_param = {"delta":delta, "gamma":gamma[j], "kappa":kappa,
                            "Amax_nd":Amax_nd, "Imax_nd":Imax_nd, 
                            "aA":aA, "bA":bA, "cA_nd":cA_nd[i], "dA":dA,
                            "aI":aI, "bI":bI, "cI_nd":cI_nd[i]}
            IC = ic.InitialConditionsRandom(noise=0.001, Ac=Aeq_nd[i], Ic=Ieq_nd[i])

            file = f"Aeq_nd{np.round(Aeq_nd[i],1)}_gamma{int(gamma[j])}" #f"Aeq{np.round(Aeq_nd[j],1)}_R{R[i]}"
            print(file)
            solve_kondo_model(IC, 
                              Tend=Tend,
                              dt=dt,
                              parameters=non_dim_param,
                              file_name=os.path.join(FOLDER, f"{file}.png"),
                              intensity_df=intensity_df,
                              column_name=False)
            
            # intensity_df.to_csv(intensity_file)

    print(datetime.now())
