import numpy as np
import xarray as xr
from pyproj import Transformer 

def get_statistics(dst,ds_inoutp):
    """
    get statistics for the patches as a function of tracking time
    """
    obs=dst.dims['obs']
    #cm, cov
    x=dst["x"]*ds_inoutp.inoutp
    y=dst["y"]*ds_inoutp.inoutp
    npin=ds_inoutp.inoutp.sum(dim='trajectory') 
    per_par=npin/npin.isel(obs=0)*100 #% of particles inside dws as a function of obs and tdep        
    cmx=x.mean(dim='trajectory')
    cmy=y.mean(dim='trajectory')#
    dispxx=((x-cmx)**2).mean(dim='trajectory')#/1e6
    dispyy=((y-cmy)**2).mean(dim='trajectory')#/1e6
    dispxy=((x-cmx)*(y-cmy)).mean(dim='trajectory')#/1e6
    obs=np.arange(0,obs*44714,44714)/86400
    #
    return cmx,cmy,dispxx,dispyy,dispxy,per_par,obs

def save_statistics(cmx,cmy,covxx,covyy,covxy,npin,cpoint_lat,cpoint_lon,dxp,dyp,l,obs,file_out='stats_temp.nc'):
    ds=xr.Dataset()
    ds["npin"]=npin
    ds["npin"].attrs["long_name"]="percentage of particles inside dws"
    #ds["cmx"]=xr.concat(cmx,dim='run')
    ds["cmx"]=cmx
    ds["cmx"].attrs["long_name"]="center of mass x-position"
    ds["cmx"].attrs["units"]="m"
    #ds["cmy"]=xr.concat(cmx,dim='run')
    ds["cmy"]=cmy
    ds["cmy"].attrs["long_name"]="center of mass y-position"
    ds["cmy"].attrs["units"]="m"
    #ds["covxx"]=xr.concat(covxx,dim='run')
    ds["dispxx"]=covxx
    ds["dispxx"].attrs["long_name"]="dispersion xx-component"
    ds["dispxx"].attrs["units"]="m2/s"
    #ds["covyy"]=xr.concat(covyy,dim='run')
    ds["dispyy"]=covyy
    ds["dispyy"].attrs["long_name"]="dispersion yy-component"
    ds["dispyy"].attrs["units"]="m2/s"
    #ds["covxy"]=xr.concat(covxy,dim='run')
    ds["dispxy"]=covxy
    ds["dispxy"].attrs["long_name"]="dispersion xy-component"
    ds["dispxy"].attrs["units"]="m2/s"
    ds["obs"]=obs
    ds["obs"].attrs["long_name"] = "tracking time"
    ds["obs"].attrs["units"] = "day"
    ds.attrs["info1_cloud"] = f"central coord of the cloud (lat,lon) = {cpoint_lat,cpoint_lon}"
    ds.attrs["info2_cloud"] = f"particles were released in {dxp}mx{dyp}m grid cells inside {l}km square"
    #ds.to_netcdf(file_out)
    return ds

def get_bins(xmid,ymid):
    dxx=np.round(np.diff(xmid).mean()); dyy=np.round(np.diff(ymid).mean())
    xedges=np.arange(xmid.min()-dxx/2,xmid.max()+dxx,dxx)
    yedges=np.arange(ymid.min()-dyy/2,ymid.max()+dyy,dyy)
    return xedges,yedges

def binning_particles(xp,yp,xedges,yedges):
    x=xp.flatten();y=yp.flatten()
    x=x[~np.isnan(x)];y=y[~np.isnan(y)] #remove nans
    hist,yedges,xedges=np.histogram2d(y,x,(yedges,xedges)); del x,y
    nrec=np.sum(hist)# = len(y), to the total input records without nans 
    hist[hist==0]=np.nan;
    return hist,nrec

def transform_coordinates(dsto,bdr_dws,xgrid,ygrid):
    #define the transformations----------
    #1)
    #from epgs:28992(DWS) to epgs:4326(LatLon with WGS84 datum used by GPS and Google Earth)
    proj = Transformer.from_crs('epsg:28992','epsg:4326',always_xy=True)
    #2)
    #from epgs:4326(LatLon with WGS84) to epgs:28992(DWS) 
    inproj = Transformer.from_crs('epsg:4326','epsg:28992',always_xy=True)
    #inproj_old=Proj("EPSG:28992") #old method (has errors 10-20m when contrast with the rotated coords)

    #lon,lat to 28992(DWS)-projection--------------------

    #bathymetry--------
    xct=dsto.lonc.values;  yct=dsto.latc.values #lon,lat units
    xctp,yctp,z = inproj.transform(xct,yct,xct*0.)
    xctp=(xctp)/1e3; yctp=(yctp)/1e3 
    #first projected point to correct the coordinates of model local meter units
    xctp0=xctp[0,0]; yctp0=yctp[0,0]

    #matrix rotation -17degrees-----
    ang=-17*np.pi/180
    angs=np.ones((2,2))
    angs[0,0]=np.cos(ang); angs[0,1]=np.sin(ang)
    angs[1,0]=-np.sin(ang); angs[1,1]=np.cos(ang)


    #local meter model units to 28992(DWS)-projection and lon-lat--------------

    #bathymetry----
    #original topo points in meter
    xct2,yct2=np.meshgrid(dsto.xc.values,dsto.yc.values)
    xy=np.array([xct2.flatten(),yct2.flatten()]).T
    #rotate
    xyp=np.matmul(angs,xy.T).T/1e3
    xyp0=xyp[0,:] #the first point in the bathy data in local meter units=0,0

    #contour0 of DWS------
    #rotate
    bdr_dwsp=np.matmul(angs,bdr_dws.T).T/1e3
    #correct model units:
    #1)substact the first model local point of the topo file, but give tha same as xyp0=[0,0]
    #2)use the first projected point of the case (lon,lat model units to meter)
    bdr_dwsp=bdr_dwsp-xyp0 
    bdr_dwsp[:,0]=bdr_dwsp[:,0]+xctp0; bdr_dwsp[:,1]=bdr_dwsp[:,1]+yctp0
    #get coordinates in lon-lat units (WGS84 ) 
    bdr_dws_lon, bdr_dws_lat, z = proj.transform(bdr_dwsp[:,0]*1e3,bdr_dwsp[:,1]*1e3, bdr_dwsp[:,1]*0.)
    #

    #regular grid of statistics------
    xy=np.array([xgrid.flatten(),ygrid.flatten()]).T
    #rotate
    xyp=np.matmul(angs,xy.T).T/1e3
    #correct model units:
    #1)substact the first model local point of the topo file, but give tha same as xyp0=[0,0]
    #2)use the first projected point of the case (lon,lat model units to meter)
    xyp=xyp-xyp0 
    xyp[:,0]=xyp[:,0]+xctp0; xyp[:,1]=xyp[:,1]+yctp0 
    xyp=np.reshape(xyp,(xgrid.shape[0],xgrid.shape[1],2))
    xgridp=xyp[...,0]; ygridp=xyp[...,1] #km
    
    return xctp,yctp,bdr_dwsp,xgridp,ygridp

def create_cmap(numcolors=11,colors=['blue','white','red'],name='create_cmap'):
    """
    Create a custom diverging colormap
    Default is blue to white to red with 11 colors. Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(N=numcolors,colors=colors,name=name)
    return cmap