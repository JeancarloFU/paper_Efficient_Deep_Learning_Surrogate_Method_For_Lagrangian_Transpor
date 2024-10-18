import numpy as np
import xarray as xr
from matplotlib import path

#save trajectories of the particles released inside the patch----
def save_particle_positions(dst,npa_per_dep,num_dep,dxp,dyp,l,save_file,file_out):
    dst["x"].attrs["long_name"]=f"particle position along GETM model x-axis"
    dst["x"].attrs["units"]=f"m"
    dst["y"].attrs["long_name"]=f"particle position along GETM model y-axis" 
    dst["y"].attrs["units"]=f"m"
    dst.attrs["num_par_per_depoyment"]=npa_per_dep
    dst.attrs["num_deployments"]=num_dep
    dst.attrs["grid_par"]=f"particles were released in {dxp}mx{dyp}m grid cells inside {l}km square"
    if save_file: dst.to_netcdf(file_out)
    return dst


#get a matrix with particle status for each time step---
def get_inout_particles_dws(dst,bdr_dws):
    #particles inside dws inout_particles = 1
    #particles outside dws inout_particles = NaN
    #
    #find if particles are inside DWS---
    radius=1e-10 #>0 to shrink Path(poligon) to exclude vertices and edges
    #to expand path: if ccw (radius>0);  if cw (radius<0)
    #to shrink path: if ccw (radius<0);  if cw (radius>0)
    pc=path.Path(bdr_dws,closed=True)
    pp=np.array([dst.x.values.flatten(),dst.y.values.flatten()]).T
    inout_particles = pc.contains_points(pp,radius=radius)*1. #points inside
    inout_particles[inout_particles==0]=np.nan
    var=list(dst.keys())[0] #'x' variable
    inout_particles=np.reshape(inout_particles,dst[var].shape) #obs,trajectory=dst.x.shape
    ds_inout_particles=xr.zeros_like(dst[[var]]).rename({var:'inout_particles'})
    ds_inout_particles['inout_particles'].values=inout_particles
    ds_inout_particles["inout_particles"].attrs["long_name"]="index to check particles inside/outside dws"
    ds_inout_particles["inout_particles"].attrs["info"]="inside dws = 1, outside dws = NaN"
    return ds_inout_particles


#xarray functions----
def xr_eig(da, dims, **kwargs):
    """Wrap :func:`numpy.linalg.eig`.
    Usage examples of all arguments is available at the :ref:`linalg_tutorial`.
    """
    return xr.apply_ufunc(
        np.linalg.eig, da, input_core_dims=[dims], output_core_dims=[dims[-1:], dims], **kwargs
    )

def xr_matmul(d1,d2,dims1,dims2,out_append="2",**kwargs):
    """Wrap :func:`numpy.linalg.matmul`.
    Usage examples of all arguments is available at the
    :ref:`matmul section <linalg_tutorial/matmul>` of the linear algebra module tutorial.
    """
    rename = True
    dim11, dim12 = dims1
    dim21, dim22 = dims2
    d1 = d1.rename({dim11: "__aux_dim11__", dim12: "__aux_dim12__"})
    d2 = d2.rename({dim21: "__aux_dim21__", dim22: "__aux_dim22__"})
    dims1 = ["__aux_dim11__", "__aux_dim12__"]
    dims2 = ["__aux_dim21__", "__aux_dim22__"]
    out_dims = ["__aux_dim11__", "__aux_dim22__"]
    matmul_aux = xr.apply_ufunc(
        np.matmul,
        d1,
        d2,
        input_core_dims=[dims1, dims2],
        output_core_dims=[out_dims],
        **kwargs,
    )
    if rename:
        return matmul_aux.rename(
            __aux_dim11__=dim11, __aux_dim22__=dim22 + out_append if dim22 == dim11 else dim22
        )
    return matmul_aux

