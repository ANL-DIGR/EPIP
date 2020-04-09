import act
import glob
import json
import xarray as xr
import numpy as np
import dask
import pandas as pd

def process_data(cf_ds, d, username, token):
    '''
    Function for processing all the datastreams/variables for a given day
    '''
    out_units = 'mm/hr'
    temp = False
    arm_d = d.strftime('%Y%m%d')
    d = d.strftime('%Y-%m-%d')
    precip = xr.Dataset()
    vmax = []
    precip_var = []
    # Run through each datastream
    for ds in cf_ds:
        # if data not available try and download
        files = glob.glob(''.join(['./', ds, '/*'+arm_d+'*cdf']))
        if len(files) == 0:
            files = glob.glob(''.join(['./', ds, '/*'+arm_d+'*nc']))
            if len(files) == 0:
                try:
                    result = act.discovery.download_data(username, token, ds, d, d)
                except:
                    pass
                files = glob.glob(''.join(['./', ds, '/*'+arm_d+'*cdf']))
                if len(files) == 0:
                    files = glob.glob(''.join(['./', ds, '/*'+arm_d+'*nc']))
        if len(files) == 0:
            continue

        try:
            obj = act.io.armfiles.read_netcdf(files)
        except:
            continue

        # Run through each variable
        for v in cf_ds[ds]['variable']:
            # These lines are used to apply ARM qc or not
            #obj = act.qc.arm.add_dqr_to_qc(obj, variable=v)
            #da = obj[v].where(obj['qc_'+v] == 0)
            da = obj[v]

            # Convert units and add to dataarray list
            units = da.attrs['units']
            if units == 'mm':
                da.attrs['units'] = 'mm/min'
            da.values = act.utils.data_utils.convert_units(da.values, da.attrs['units'], out_units)
            da.attrs['units'] = out_units
            attrs = da.attrs

            # Sample to 1 minute
            da = da.resample(time='1min', keep_attrs=True).mean()#nearest(tolerance='1min')

            # Add attributes back to data array
            da.attrs = attrs

            # Keep running list of variable names
            precip_var.append('_'.join([ds, v]))

            # Add dataarray
            precip['_'.join([ds, v])] = da

            # Add additional data
            variables = ['temp_mean', 'rh_mean', 'wdir_vec_mean', 'wspd_vec_mean', 'atmos_pressure', 'pwd_pw_code_inst']
            if ds == 'sgpaosmetE13.a1' and temp is False:
                aosmet_var = ['temperature_ambient', 'rh_ambient', 'wind_direction', 'wind_speed', 'pressure_ambient']
                for i, var in enumerate(aosmet_var):
                    precip[variables[i]] = obj[var].resample(time='1min').mean()
                temp = True
            if ds == 'sgpmetE13.b1' and temp is False:
                for var in variables:
                    precip[var] = obj[var].resample(time='1min').nearest()
                temp = True
            if ds == 'sgpmwr3cC1.b1':
                mwr3c_vars = ['lwp', 'pwv']
                for var in mwr3c_vars:
                    precip[var] = obj[var].resample(time='1min').nearest()

            da.close()
        obj.close()

    # Only use data when temperature above freezing
    precip = precip.where(precip['temp_mean'] > 2., drop=True)

    if len(precip['time']) == 0:
        return

    precip = precip.fillna(0)

    # Run data through QC method using quantiles
    precip_df = precip.to_dataframe()
    avg = precip_df[precip_var].mean(axis=1)
    std = precip_df[precip_var].std(axis=1)
    upper = np.nanmean(precip_df[precip_var].sum()/60.) + np.nanstd(precip_df[precip_var].sum()/60.)
    for v in precip_var:
        if v == 'time':
            continue
        if precip_df[v].sum()/60. > upper:
            precip_df.iloc[np.where(precip_df[v] > precip_df[precip_var].quantile(0.999, axis=1))[0], [precip_df.columns.get_loc(v)]] = 0
        precip[v].values = precip_df[v].values

    # Create a total precipitation variable to drop times when
    # no instruments are recording precipitation   
    test = precip_df[precip_var].sum(axis=1).to_xarray()
    precip['total_precip'] = test
    precip = precip.where(precip['total_precip'] > 0., drop=True)
    precip = precip.drop('total_precip')

    if len(precip['time']) == 0:
        return

    vmax = [np.nanmax(precip[v].values) for v in precip_var]

    # Count number of instruments recording precip
    # If there's more than 3 instruments recording precip,
    # write out to file
    vsum = sum(i > 0 for i in vmax)
    if vsum > 3:
        precip.to_netcdf('./sgpprecip/sgpprecip.' + arm_d + '.nc')

if __name__ == '__main__':

    # Read in ARM Live Data Webservice Token and Username
    with open('./token.json') as f:
        data = json.load(f)
    username = data['username']
    token = data['token']

    # Specify dictionary of datastreams, variables, and weights
    cf_ds = {'sgpmetE13.b1': {'variable': ['tbrg_precip_total_corr', 'org_precip_rate_mean',
                                           'pwd_precip_rate_mean_1min']},
             'sgpvdisC1.b1': {'variable': ['rain_rate']},
             'sgpvdisE13.b1': {'variable': ['rain_rate']},
             'sgpdisdrometerC1.b1': {'variable': ['rain_rate']},
             'sgpdisdrometerE13.b1': {'variable': ['rain_rate']},
             'sgpstamppcpE13.b1': {'variable': ['precip']},
             'sgpwbpluvio2C1.a1': {'variable': ['intensity_rtnrt']},
             'sgpldE13.b1': {'variable': ['precip_rate']},
             'sgpldC1.b1': {'variable': ['precip_rate']},
             'sgpaosmetE13.a1': {'variable': ['rain_intensity']},
             'sgpmwr3cC1.b1': {'variable': ['rain_intensity']}
             }

    # Specify date for analysis
    startdate = '2017-01-01'
    #startdate = '2019-05-20'
    enddate = '2019-12-31'
    #enddate = '2019-05-20'
    sdate = ''.join(startdate.split('-'))
    edate = ''.join(enddate.split('-'))
    days = act.utils.datetime_utils.dates_between(sdate, edate)

    days = sorted(days)

    #  Run through each day, convert precip to same units, combine all precip
    # into one object and write out to netcdf if more than 5 instruments are
    # recording precipitation.
    # Note the disdrometers routinely record precip in high wind conditions
    # and that was the reason for upping the threshold
    task = []
    for d in days:
        #print(d)
        #result = process_data(cf_ds, d, username, token)
        task.append(dask.delayed(process_data)(cf_ds, d, username, token))

    result = dask.compute(*task)
    print(result)
