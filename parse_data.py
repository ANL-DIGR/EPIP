import act
import glob
import json
import xarray as xr
import numpy as np
import dask


def process_data(cf_ds, d, username, token):
    out_units = 'mm/hr'
    temp = False
    arm_d = d.strftime('%Y%m%d')
    d = d.strftime('%Y-%m-%d')
    precip = xr.Dataset()
    vmax = []
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
            da = obj[v]
            # Convert units and add to dataarray list
            units = da.attrs['units']
            if units == 'mm':
                da.attrs['units'] = 'mm/min'
            da.values = act.utils.data_utils.convert_units(da.values, da.attrs['units'], out_units)
            da.attrs['units'] = out_units
            da = da.resample(time='1min').mean()
            precip['_'.join([ds, v])] = da

            # Add temperature data
            variables = ['temp_mean', 'rh_mean', 'wdir_vec_mean', 'wspd_vec_mean', 'atmos_pressure', 'pwd_pw_code_inst']
            if ds == 'sgpaosmetE13.a1' and temp is False:
                aosmet_var = ['temperature_ambient', 'rh_ambient', 'wind_direction', 'wind_speed', 'pressure_ambient', '']
                for i, var in enumerate(variables), ds, d, d:
                    precip[var] = obj[aosmet_var[i]].resample(time='1min').mean()
                temp = True
            if ds == 'sgpmetE13.b1' and temp is False:
                for var in variables:
                    precip[var] = obj[var]
                temp = True
            if ds == 'sgpmwr3cC1.b1' and temp is False:
                mwr3c_vars = ['lwp', 'pwv']
                for var in mwr3c_vars:
                    precip[var] = obj[var]

            da.close()
        obj.close()

    # Only use data when temperature above freezing
    precip = precip.where(precip['temp_mean'] > 2.)
    vmax = [np.nanmax(precip[v].values) for v in precip]

    # Count number of instruments recording precip
    vsum = sum(i > 0 for i in vmax)
    if vsum > 5:
        precip.to_netcdf('./sgpprecip/sgpprecip.' + arm_d + '.nc')

if __name__ == '__main__':

    # Read in ARM Live Data Webservice Token and Username
    with open('./token.json') as f:
        data = json.load(f)
    username = data['username']
    token = data['token']

    # Specify dictionary of datastreams, variables, and weights
    cf_ds = {'sgpmetE13.b1': {'variable': ['tbrg_precip_total', 'org_precip_rate_mean',
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
    enddate = '2019-12-31'
    #enddate = '2017-01-10'
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
        task.append(dask.delayed(process_data)(cf_ds, d, username, token))

    result = dask.compute(*task)
    print(result)