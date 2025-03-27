import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import numpy.ma as ma
import math
import pandas as pd
import os
import xarray as xr

# handling paths
homepath = r"C:\Users\nd20983\docs\Simulations\data\rawdata"
expts = ['xpgxa', 'xpgxb', 'xpgxc', 'xpgxx', 'xpgxw', 'xpgxr', 'xpgxs', 
         'xpgxt', 'xpgxh', 'xpgcy',]
for expt in expts:
     exptpath = os.path.abspath(os.path.join(homepath, expt+'a.pdclann.nc'))
     #print(exptpath)
     ds = xr.open_dataset(exptpath)


# home = 'C:/Users/nd20983/OneDrive - University of Bristol/Documents'
# #home = 'C:/Users/Virgil/OneDrive - University of Bristol/Documents'
# adr_xpgxa = home + '/Simulations/data/rawdata/xpgxaa.pdclann.nc'
# adr_xpgxb = home + '/Simulations/data/rawdata/xpgxba.pdclann.nc'
# adr_xpgxc = home + '/Simulations/data/rawdata/xpgxca.pdclann.nc'
# adr_xpgxx = home + '/Simulations/data/rawdata/xpgxxa.pdclann.nc'
# adr_xpgxw = home + '/Simulations/data/rawdata/xpgxwa.pdclann.nc'
# adr_xpgxr = home + '/Simulations/data/rawdata/xpgxra.pdclann.nc'
# adr_xpgxs = home + '/Simulations/data/rawdata/xpgxsa.pdclann.nc'
# adr_xpgxt = home + '/Simulations/data/rawdata/xpgxta.pdclann.nc'
# adr_xpgxh = home + '/Simulations/data/rawdata/xpgxha.pdclann.nc'
# adr_xpecy = home + '/Simulations/data/rawdata/xpecya.pdclann.nc'
# adr_cella = home + '/Simulations/data/rawdata/areacella_fx_HadCM3_historical_r0i0p0.nc'
"""
adr_expts = {'xpgxa': adr_xpgxa,
             'xpgxb': adr_xpgxb,
             'xpgxc': adr_xpgxc,
             'xpgxx': adr_xpgxx,
             'xpgxw': adr_xpgxw,
             'xpgxr': adr_xpgxr,
             'xpgxs': adr_xpgxs,
             'xpgxt': adr_xpgxt,
             'xpgxh': adr_xpgxh,
             'xpecy': adr_xpecy
            }
var_dict = {'evapsea': 'evapsea_mm_srf',
            'precip': 'precip_mm_srf',
           }
Nordic = {{lat_min:56, lat_max:69, lon_min:88, lon_max:96},
          {lat_min:56, lat_max:69, lon_min:0, lon_max:9}
         }
# real coordinates (Nordic Seas): lat_min:50, lat_max:80, lon_min:-30, lon_max:20
Natl = {'lat_min':50, 'lat_max':67, 'lon_min':70, 'lon_max':96}
# real coordinates (N. Atlantic): lat_min:35, lat_max:75, lon_min:260, lon_max:360
Npac = {'lat_min':50, 'lat_max':67, 'lon_min':32, 'lon_max':60}
# lat_min:35, lat_max: 75, lon_min:120, lon_max:220
Tatl_1 = {'lat_min':26, 'lat_max':40, 'lon_min':80, 'lon_max':96}
Tatl_2 = {'lat_min':26, 'lat_max':40, 'lon_min':0, 'lon_max':7}
Tatl_3 = {'lat_min':40, 'lat_max':47, 'lon_min':70, 'lon_max':94}
Tpac_1 = {'lat_min':26, 'lat_max':40, 'lon_min':32, 'lon_max':81}
Tpac_2 = {'lat_min':40, 'lat_max':47, 'lon_min':32, 'lon_max':70}
bound_dict = {'Natl': Natl,
              'Npac': Npac,
              'Tatl_1': Tatl_1,
              'Tatl_2': Tatl_2,
              'Tatl_3': Tatl_3,
              'Tpac_1': Tpac_1,
              'Tpac_2': Tpac_2
             }


# import data
ds_cli = {}
waterflux = {}
ds_cella = nc.Dataset(adr_cella, mode='r')
for expt, expt_adr in adr_expts.items():
    ds_cli[expt] = nc.Dataset(expt_adr, mode='r')
    waterflux[expt] = {}
    for bound, value_b in bound_dict.items():
        waterflux[expt][bound] = {}
        for var, varname in var_dict.items():
            waterflux[expt][bound][var] = {}
            var_temp = ds_cli[expt].variables[varname]
            cella = ds_cella.variables['areacella']
            box = value_b
# the pd file is ordered in atmospheric style while the areacell is ordered in oceanic style,
# why do we keep them the same?
# Because when we apply the mask of the evaporation on the areacell and on the precip and the areacell of the precip, we want to
# keep the masked area the same, do not create additional masked area by mistake.
            flux_ave = var_temp[0,0,(73-box['lat_max']):(73-box['lat_min']),box['lon_min']:box['lon_max']]
            area_b = cella[(73-box['lat_max']):(73-box['lat_min']),box['lon_min']:box['lon_max']]
            if var == 'evapsea':
                    mask = flux_ave
            new_flux_ave = ma.masked_array(flux_ave, mask.mask)
            new_area_b = ma.masked_array(area_b, mask.mask)
            #unit is Sv for flux_tot
            flux_tot = (new_flux_ave*new_area_b)/1000/1000000
            flux_tot.filled(np.nan)
            waterflux[expt][bound][var] = flux_tot.sum()
            

df_waterflux = pd.DataFrame.from_dict({(i,j): waterflux[i][j]
                        for i in waterflux.keys()
                        for j in waterflux[i].keys()
                       },
                       orient='index'
                      )
#df_waterflux.to_csv(r'../Simulations/data/waterflux1.csv', index=True)
waterflux"
"""