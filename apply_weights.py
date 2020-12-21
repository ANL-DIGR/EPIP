import act
import glob
import xarray as xr
import dask
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
import numpy as np
from statistics import mode, stdev
import sys
from collections import Counter

# Weights to use
clusters = 5
# Kmeans to use
kclust =2
def proc_data(data, rain_rate, weights, variables, corresponding_id, weight_inst, rr):
    accum_weight = []
    for j, var in enumerate(variables):
        if data[j] == 0:
            accum_weight.append(0.)
            continue
        idx = weight_inst == corresponding_id[var]
        if not any(idx):
            accum_weight.append(0.)
            continue

        index = np.where(idx)[0]
        rr_ind = np.argmin(np.abs(np.array(rain_rate) - rr))
        #rr_ind = np.argmin(np.abs(np.array(rain_rate) - rain_rate[j]))
        wt = weights[index,rr_ind][0]
        #print(var, rain_rate[j], wt, rr)
        accum_weight.append(wt)

    wt_data = (data * accum_weight) / np.nansum(accum_weight)

    idx = np.where(np.array(accum_weight) > 0.)[0]
    if len(idx) <= 1:
        return wt_data * 0

    return wt_data


def calc_80_20(df):
    dummy =  df
    #dummy = df.replace(0, np.NaN)
    #JK new data frames for 80/20 weights
    tb_wb_avg = dummy[['sgpmetE13.b1_tbrg_precip_total_corr', 
                    'sgpwbpluvio2C1.a1_intensity_rtnrt',
                    'sgpstamppcpE13.b1_precip']].mean(axis=1)

    no_catch_avg = dummy[['sgpmetE13.b1_org_precip_rate_mean', 
                       'sgpmetE13.b1_pwd_precip_rate_mean_1min', 
                       'sgpvdisfilteredC1.b1_rain_rate', 
                       'sgpdisdrometerC1.b1_rain_rate',
                       'sgpdisdrometerE13.b1_rain_rate',
                       'sgpldE13.b1_precip_rate', 
                       'sgpldC1.b1_precip_rate',
                       'sgpaosmetE13.a1_rain_intensity'
                       ]].mean(axis=1)
                       #'sgpaosmetE13.a1_rain_intensity']].mean(axis=1)

    weight_80_20 = (tb_wb_avg * 0.8) + (no_catch_avg * 0.2)

    #JK end 
    return weight_80_20

if __name__ == '__main__':
    """
    Program to run through all the parsed data and create a 
    precipitation best estimate based on clustering
    """

    #files = glob.glob('./data_pars_vdis_filtered/sgpprecip*201*')
    files = glob.glob('./sgpprecip/sgpprecip*201*')

    # Open files and accumulate precipitation
    obj = act.io.armfiles.read_netcdf(files)
    obj = obj.fillna(0)
    v_id = {'sgpmetE13.b1_tbrg_precip_total_corr': 'novalynx_tbrg', 'sgpmetE13.b1_org_precip_rate_mean': 'opt_sci_org',
            'sgpmetE13.b1_pwd_precip_rate_mean_1min': 'vaisala_pwd', 'sgpvdisfilteredC1.b1_rain_rate': 'joanneum_vdis',
            'sgpvdisfilteredE13.b1_rain_rate': 'joanneum_vdis', 'sgpdisdrometerC1.b1_rain_rate': 'distromet_disdrometer',
            'sgpdisdrometerE13.b1_rain_rate': 'distromet_disdrometer', 'sgpstamppcpE13.b1_precip': 'texas_elec_tbrg',
            'sgpwbpluvio2C1.a1_intensity_rtnrt': 'ott_pluvio2', 'sgpldE13.b1_precip_rate': 'ott_parsivel2',
            'sgpldC1.b1_precip_rate': 'ott_parsivel2', 'sgpaosmetE13.a1_rain_intensity': 'vaisala_wxt',
            'sgpmwr3cC1.b1_rain_intensity': 'vaisala_wxt'}
    for v in obj:
        if obj[v].attrs['units'] != 'mm/hr':
            continue
        if 'sgpmwr3c' in v:
            obj = obj.drop_vars(v)
            continue
        #if 'sgpldE13' in v:
        #    obj = obj.drop_vars(v)
        #    continue
        #if 'sgpaosmet' in v:
        #    obj = obj.drop_vars(v)
        #    continue
        if 'pwd_precip_rate' in v:
            data = obj[v].sel(time=slice(None, "2017-11-01"))
            obj[v] = data
            #obj[v] = obj[v].fillna(0)
        if 'org_precip_rate_mean' in v:
            odata = obj[v].values
            data = obj[v].sel(time=slice("2017-03-24", None))
            obj[v] = data
            #obj[v] = obj[v].fillna(0)

        # Check DQR System for records
        obj.attrs['_datastream'] = v.split('_')[0]
        dqr_var = '_'.join(v.split('_')[1:])
        obj = act.qc.arm.add_dqr_to_qc(obj, variable=dqr_var, assessment='incorrect', add_qc_variable=v)
        if 'qc_'+v in obj:
            da = obj[v].where(obj['qc_'+v] == 0)
            obj[v] = da

        obj = act.utils.data_utils.accumulate_precip(obj, v, time_delta=60.)

    #weight_file = glob.glob('./weights/cluster_'+str(clusters)+'_max_norm.nc')
    weight_file = glob.glob('./weights/cluster_'+str(clusters)+'_high_qc_sum_norm.nc')
    #weight_file = glob.glob('./weights/cluster_3_max_norm.nc')
    weight_obj = act.io.armfiles.read_netcdf(weight_file)
    weights = weight_obj['weights'].values
    weight_inst = weight_obj['instruments'].values
    rain_rate = weight_obj['rain_rate'].values

    # Convert to pandas dataframe
    df = obj.to_dataframe()
    # Drop any non-rain rate variables
    for d in df:
        if obj[d].attrs['units'] != 'mm/hr':
            df = df.drop(d,1)

    variables = list(df.columns)

    # Create kmeans variables
    #k_file = glob.glob('./kmeans_precip/kmeans_cluster_'+str(clusters)+'.nc')
    k_file = glob.glob('./kmeans_precip/kmeans_cluster_high_qc_sum'+str(kclust)+'.nc')
    k_obj = act.io.armfiles.read_netcdf(k_file)
    k_obj = k_obj.rename({'__xarray_dataarray_variable__': 'kmeans'})
    k_obj = k_obj.sel(time=slice(obj['time'].values[0], obj['time'].values[-1]))
    k_obj = act.utils.data_utils.accumulate_precip(k_obj, 'kmeans')

    task = []
    rr_w = []
    for i in range(len(df.index)):
        data = np.asarray(df.iloc[i])
        rr = k_obj['kmeans'].sel(time=df.index[i]).values * 60.
        #rr_w.append(proc_data(data, rain_rate, weights, variables, v_id, weight_inst, rr))
        task.append(dask.delayed(proc_data)(data, rain_rate, weights, variables, v_id, weight_inst, rr))

    rr_w = list(dask.compute(*task))

    rr = np.nansum(rr_w,axis=1)

    # Create 80/20 Variable
    rr_80_20 = calc_80_20(df)
    atts = {'units': 'mm/hr', 'long_name': 'Precipitation Rate (80/20)'}
    obj['rr_80_20'] = xr.DataArray(rr_80_20, dims=['time'], coords=[obj['time'].values], attrs=atts)
    obj = obj.fillna(0)
    obj = act.utils.data_utils.accumulate_precip(obj, 'rr_80_20')

    # Add rain rate
    atts = {'units': 'mm/hr', 'long_name': 'Precipitation Rate (Weighted)'}
    obj['rr_weighted_'+str(clusters)] = xr.DataArray(rr, dims=['time'], coords=[obj['time'].values], attrs=atts)
    obj = obj.fillna(0)
    obj = act.utils.data_utils.accumulate_precip(obj, 'rr_weighted_'+str(clusters))

    obj.to_netcdf('./sgpprecipitationC1.b1/sgpprecipitationC1.b1.2017_2019.nc')
    k_obj.to_netcdf('./sgpprecipitationC1.b1/kmeans_precipitation_2017_2019.nc')


    # Create plot with accumulations
    labels = ['NovaLynx Tipping Bucket', 'Optical Scientific Optical Rain Gauge', 'Vaisala Present Weather Detector',
              'Joanneum Research Video Disdrometer', 'Distromet Impact Disdrometer',
              'Texas Electronics Tipping Bucket', 'Pluvio 2 Weighing Bucket', 'Parsivel 2', 'WXT-520']
    fig, ax = plt.subplots(nrows=1, figsize=(12,8))
 
    for d in obj:
        if 'accumulated' not in d:
            continue
        lab = d + ': '+ str(round(obj[d].values[-1],2))
        ax.plot(obj['time'], obj[d], label=lab)
    ax.plot(k_obj['time'], k_obj['kmeans_accumulated'], label='K-Means Best Estimate '+str(kclust)+
            ': '+str(round(k_obj['kmeans_accumulated'].values[-1],2)))
    ax.set_xlim([df.index[0], df.index[-1]])
    ax.legend(loc=2)
    plt.show()


    ncols = 3
    nrows = int(np.ceil(len(df.columns) / ncols)) + 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,8))
    hist = []
    x = 0
    y = 0
    for i, d in enumerate(df):
        data = df[d]
        data = data[data > 0]
        h, bins = np.histogram(data, bins=100, range=(0,100))
        center = (bins[:-1] + bins[1:]) / 2
        ax[x,y].bar(center, h, label=d)
        ax[x,y].set_title(d)
        ax[x,y].set_yscale('log')
        ax[x,y].set_ylim(0.1, 3000)
        ax[x,y].grid(True)
        if y == 0:
            ax[x,y].set_ylabel('Counts\n(Log Scale)')
        if x == nrows-1:
            ax[x,y].set_xlabel('Rain Rate')
        y += 1
        if y == ncols:
            x += 1
            y = 0

    y = 0
    x = 4
    data = obj['rr_80_20'].values
    data = data[data > 0]
    h, bins = np.histogram(data, bins=100, range=(0,100))
    center = (bins[:-1] + bins[1:]) / 2
    ax[x,y].bar(center, h, label=d)
    ax[x,y].set_title('80/20 Rain Rates')
    ax[x,y].set_yscale('log')
    ax[x,y].set_ylim(0.1, 3000)
    ax[x,y].grid(True)
    ax[x,y].set_ylabel('Counts\n(Log Scale)')
    ax[x,y].set_xlabel('Rain Rate')

    y += 1
    data = k_obj['kmeans'].values
    data = data[data > 0]
    h, bins = np.histogram(data * 60., bins=100, range=(0,100))
    center = (bins[:-1] + bins[1:]) / 2
    ax[x,y].bar(center, h, label=d)
    ax[x,y].set_title('Kmeans '+str(kclust)+' Rain Rates')
    ax[x,y].set_yscale('log')
    ax[x,y].set_ylim(0.1, 3000)
    ax[x,y].grid(True)
    ax[x,y].set_xlabel('Rain Rate')

    y += 1
    data = obj['rr_weighted_'+str(clusters)].values
    data = data[data > 0]
    h, bins = np.histogram(data, bins=100, range=(0,100))
    center = (bins[:-1] + bins[1:]) / 2
    ax[x,y].bar(center, h, label=d)
    ax[x,y].set_title('Weighted '+str(clusters)+' Rain Rates')
    ax[x,y].set_yscale('log')
    ax[x,y].set_ylim(0.1, 3000)
    ax[x,y].grid(True)
    ax[x,y].set_xlabel('Rain Rate')

    fig.tight_layout()
    plt.show()
    
    #im = ax.pcolormesh(bins, df.columns, hist, cmap='jet', vmin=0)
    #fig.colorbar(im, ax=ax, orientation='vertical')
    #plt.show()     
    #center = (bins[:-1] + bins[1:]) / 2
    #plt.bar(center, hist)
    #plt.show()
    obj.close()
    weight_obj.close()
