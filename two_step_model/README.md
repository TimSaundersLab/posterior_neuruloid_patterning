# Phenomenological modelling of neuruloid patterning 
Modelling of neuruloid patterning using reaction-diffusion model and minimal gene regulatory network.

### Folders
```
├── mcmc/                             # normalised tbxt intensities
        ├── mcmc_functions.py         # mcmc fitting functions
├── model/                            # toy data to run notebook
        ├── conditions.py         # to implement initial and boundary conditions for rd model
        ├── intensity_analysis.py # extracting intensity values from rd simulations
        ├── kondo_model.py        # rd model
        ├── mesh_generation.py    # making meshes of the domain
        ├── mutual_inhibition_grn # ode model of grn
        ├── nondim_kondo_model.py # dimensionless kondo model
├── notebooks/
        ├── simulation_intensity         # saved intensity 
        ├── kondo_model_analysis.ipynb   # analyse fixed points and production function from
        ├── kondo_model_isolines.ipynb   # analyse thickness of TBXT ring
        ├── grn_simulation.ipynb         # analysing ode model
        ├── grn_for_shapes.ipynb         # two-step model on geometrical neuruloids
        └── mcmc_model_fit.ipynb         # fitting two-step model to neuruloid data
└── scripts/
        └── kondo_model_sensitivity.py   # for sensitivity analysis simulations
```