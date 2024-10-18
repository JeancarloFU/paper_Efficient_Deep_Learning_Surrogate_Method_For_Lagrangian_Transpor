## paper_Efficient_Deep_Learning_Surrogate_Method_For_Lagrangian_Transport
This repository contains scripts and data related to the manuscript:

> Fajardo-Urbina, J.M., Liu, Y., Georgievska, S., Gräwe, U., Clercx, H.J.H., Gerkema, T., & Duran-Matute, M. (2024). Efficient deep learning surrogate method for predicting the transport of particle patches
in coastal environments. Available at SSRN: http://dx.doi.org/10.2139/ssrn.4815334

### Data
There are in total 5 NetCDF files. They can be used to run the surrogate and optimal prediction experiments and reproduce results from Figure 5 of the manuscript. They are provided inside the folder [data](https://github.com/JeancarloFU/paper_Efficient_Deep_Learning_Surrogate_Method_For_Lagrangian_Transport/blob/main/data).

### Sofware
The environment used for the analysis uses Python v3.8 and can be found in the file [environment.yml](https://github.com/JeancarloFU/paper_Efficient_Deep_Learning_Surrogate_Method_For_Lagrangian_Transport/blob/main/environment.yml).

### Running the Notebooks
The simplified Lagrangian model (Eq. (4) of the manuscript) is implemented in the notebook located in the folder [notebooks](https://github.com/JeancarloFU/paper_Efficient_Deep_Learning_Surrogate_Method_For_Lagrangian_Transport/blob/main/notebooks). Here, we also add all the necessary instructions to run the surrogate and optimal prediction experiments. The notebook can be run using any of the following instructions:
- Mybinder.org: click on the binder icon to open a jupyter-lab session. 
- Google Colab: follow the instructions from the notebook [clone_repo_using_google_colab.ipynb](https://github.com/JeancarloFU/paper_Efficient_Deep_Learning_Surrogate_Method_For_Lagrangian_Transport/blob/main/clone_repo_using_google_colab.ipynb)
- Clone or download the repository on your PC: install the packages of the file [environment.yml](https://github.com/JeancarloFU/paper_Efficient_Deep_Learning_Surrogate_Method_For_Lagrangian_Transport/blob/main/environment.yml).

**Jup-lab:** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JeancarloFU/paper_Efficient_Deep_Learning_Surrogate_Method_For_Lagrangian_Transport/main?urlpath=lab)

### Information about raw numerical data and the ConvLSTM
The netCDF files provided in this repository are generated from the following raw data:

- Eulerian data from the GETM/GOTM model, and its set-up is described in:
    * Duran-Matute et al. (2014): https://doi.org/10.5194/os-10-611-2014
    * Grawe et al. (2016): https://doi.org/10.1002/2016JC011655
- The Lagrangian model Parcels v2.4.2 can be installed from: 
    * https://anaconda.org/conda-forge/parcels
    * https://oceanparcels.org
- The ConvLSTM model used in our study is built using Pytorch (https://anaconda.org/pytorch/pytorch), and its implementation is described in:
    * Liu et al. (2021) https://doi.org/10.1175/MWR-D-20-0113.1
