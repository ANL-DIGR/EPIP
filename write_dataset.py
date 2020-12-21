import act
import glob
import xarray as xr
import sys
import numpy as np


if __name__ == '__main__':
    """
    Program to run through all the parsed data and create a 
    precipitation best estimate based on clustering
    """
    v_id = {'sgpmetE13.b1_tbrg_precip_total_corr': 'novalynx_tbrg', 'sgpmetE13.b1_org_precip_rate_mean': 'opt_sci_org',
            'sgpmetE13.b1_pwd_precip_rate_mean_1min': 'vaisala_pwd', 'sgpvdisfilteredC1.b1_rain_rate': 'joanneum_vdis',
            'sgpvdisfilteredE13.b1_rain_rate': 'joanneum_vdis', 'sgpdisdrometerC1.b1_rain_rate': 'distromet_disdrometer',
            'sgpdisdrometerE13.b1_rain_rate': 'distromet_disdrometer', 'sgpstamppcpE13.b1_precip': 'texas_elec_tbrg',
            'sgpwbpluvio2C1.a1_intensity_rtnrt': 'ott_pluvio2', 'sgpldE13.b1_precip_rate': 'ott_parsivel2',
            'sgpldC1.b1_precip_rate': 'ott_parsivel2', 'sgpaosmetE13.a1_rain_intensity': 'vaisala_wxt'}

    # Open files and accumulate precipitation
    files = glob.glob('./sgpprecipitationC1.b1/sgpprecipitation*')
    obj = act.io.armfiles.read_netcdf(files)

    files = glob.glob('./sgpprecipitationC1.b1/kmeans*')
    kobj = act.io.armfiles.read_netcdf(files)

    new = xr.merge([obj, kobj])

    da = []
    for v in v_id:
        if 'ancillary_variables' in new[v].attrs:
            del new[v].attrs['ancillary_variables']
        if 'equation' in new[v].attrs:
            del new[v].attrs['equation']
        if 'comment' in new[v].attrs:
            del new[v].attrs['comment']
        if 'valid_min' in new[v].attrs:
            del new[v].attrs['valid_min']
        if 'valid_max' in new[v].attrs:
            del new[v].attrs['valid_max']
        if 'cell_methods' in new[v].attrs:
            del new[v].attrs['cell_methods']
        if 'standard_name' in new[v].attrs:
            del new[v].attrs['standard_name']
        if 'threshold' in new[v].attrs:
            del new[v].attrs['threshold']
        if 'absolute_accuracy' in new[v].attrs:
            del new[v].attrs['absolute_accuracy']
        da.append(new[v])
        #da.append(new[v+'_accumulated'])

    add_vars = ['rr_80_20', 'rr_weighted_5', 'kmeans']
    long_name = ['Rain rate calculated using 80/20 ratio',
                 'Rain rate calculated using derived weights as described in the readme',
                 'Rain rate calculated using K-means clustering of 2 clusters']
    comment = ['80% to weighing and tipping bucket, 20% to everything else','See readme',
               'Rain rate calculated taking average of dominant cluster using k-means 2']
    for i, v in enumerate(add_vars):
        new[v].attrs['units'] = 'mm/hr'
        new[v].attrs['long_name'] = long_name[i]
        new[v].attrs['comments'] = comment[i]
        da.append(new[v])
        #da.append(new[v+'_accumulated'])

    met_vars = ['temp_mean', 'rh_mean', 'wdir_vec_mean', 'wspd_vec_mean', 'atmos_pressure', 'pwd_pw_code_inst', 'lwp', 'pwv']
    instrument = ['sgpmetE13.b1', 'sgpmetE13.b1', 'sgpmetE13.b1', 'sgpmetE13.b1', 'sgpmetE13.b1', 'sgpmetE13.b1', 'sgpmwr3cC1.b1',
                  'sgpmwr3cC1.b1']
    for i, v in enumerate(met_vars):
        if 'ancillary_variables' in new[v].attrs:
            del new[v].attrs['ancillary_variables']
        new[v].attrs['source'] = instrument[i]
        if instrument[i] == 'sgpmetE13.b1':
            new[v].attrs['comment'] = 'If sgpmetE13.b1 was not available, sgpaosmetE13.a1 was used and PWD will not be available.'
        da.append(new[v])

    new = xr.merge(da)

    new.to_netcdf('./sgpprecipitationC1.b1/sgpprecipitationC1.b1.2017_2019.000000.nc')

    dates = np.unique(new.time.dt.strftime('%Y%m%d'))

    for d in dates:
        test = new.sel(time=d)
        filename = 'sgpprecipitationC1.b1.'+d+'.000000.nc'
        test.to_netcdf('./sgpprecipitationC1.b1/daily/'+filename)
