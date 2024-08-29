import act
import glob
import xarray as xr
import dask
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
import numpy as np
from statistics import mode, stdev
import sys
from collections import Counter
import pandas as pd
import time

if __name__ == '__main__':
    """
    Program to test out different clustering numbers to determine how the instruments
    compare across rain rates
    """
    t = time.time()
    files = glob.glob('./sgpprecip/sgpprecip*201*')
    #files = glob.glob('./data_pars_vdis_filtered/sgpprecip*201*')
    clusters = 6


    # Open files and accumulate precipitation
    obj = act.io.armfiles.read_netcdf(files, parallel=True)
    ldis_sum = ~np.isnan(obj['sgpldE13.b1_precip_rate'].values) * 1 + ~np.isnan(obj['sgpldC1.b1_precip_rate'].values) * 1
    obj['sgpldE13.b1_precip_rate'].values = np.nansum([obj['sgpldE13.b1_precip_rate'].values,
                                                       obj['sgpldC1.b1_precip_rate'].values], axis=0) / ldis_sum

    data = obj['sgpvdisfilteredE13.b1_rain_rate'].sel(time=slice("2018-06-19", None))
    obj['sgpvdisfilteredE13.b1_rain_rate'] = data
    vdis_sum = ~np.isnan(obj['sgpvdisfilteredE13.b1_rain_rate'].values) * 1 + ~np.isnan(obj['sgpvdisfilteredC1.b1_rain_rate'].values) * 1
    obj['sgpvdisfilteredE13.b1_rain_rate'].values = np.nansum([obj['sgpvdisfilteredE13.b1_rain_rate'].values,
                                                               obj['sgpvdisfilteredC1.b1_rain_rate'].values],
                                                               axis=0) / vdis_sum

    dis_sum = ~np.isnan(obj['sgpdisdrometerE13.b1_rain_rate'].values) * 1 + ~np.isnan(obj['sgpdisdrometerC1.b1_rain_rate'].values) * 1
    obj['sgpdisdrometerE13.b1_rain_rate'].values = np.nansum([obj['sgpdisdrometerE13.b1_rain_rate'].values,
                                                              obj['sgpdisdrometerC1.b1_rain_rate'].values],
                                                              axis=0) / dis_sum

    wxt_sum = ~np.isnan(obj['sgpaosmetE13.a1_rain_intensity'].values) * 1 + ~np.isnan(obj['sgpmwr3cC1.b1_rain_intensity'].values) * 1
    obj['sgpaosmetE13.a1_rain_intensity'].values = np.nansum([obj['sgpaosmetE13.a1_rain_intensity'].values,
                                                              obj['sgpmwr3cC1.b1_rain_intensity'].values],
                                                              axis=0) / wxt_sum
    obj = obj.fillna(0)

    for v in obj:
        if obj[v].attrs['units'] != 'mm/hr':
            continue
        # Removing duplicate instruments
        if 'sgpvdisfilteredC1' in v:
            obj = obj.drop_vars(v)
            continue
        if 'sgpdisdrometerC1' in v:
            obj = obj.drop_vars(v)
            continue
        if 'sgpldC1' in v:
            obj = obj.drop_vars(v)
            continue
        if 'sgpmwr3c' in v:
            obj = obj.drop_vars(v)
            continue
        if 'org_precip_rate_mean' in v:
            data = obj[v].sel(time=slice("2017-03-24", None))
            obj[v] = data
            obj[v] = obj[v].fillna(0)
        if 'pwd_precip_rate' in v:
            data = obj[v].sel(time=slice(None, "2017-11-01"))
            obj[v] = data
            obj[v] = obj[v].fillna(0)

        # Check DQR System for records
        obj.attrs['_datastream'] = v.split('_')[0]
        dqr_var = '_'.join(v.split('_')[1:])
        obj = act.qc.arm.add_dqr_to_qc(obj, variable=dqr_var, assessment='incorrect', add_qc_variable=v)
        if 'qc_'+v in obj:
            da = obj[v].where(obj['qc_'+v] == 0)
            obj[v] = da

        obj = act.utils.data_utils.accumulate_precip(obj, v, time_delta=60.)

    # Convert to pandas dataframe
    df = obj.to_dataframe()
 
    #JK new data frames for 80/20 weights
    tb_wb_avg = df[['sgpmetE13.b1_tbrg_precip_total_corr_accumulated', 
                    'sgpwbpluvio2C1.a1_intensity_rtnrt_accumulated']].mean(axis=1)
    
    no_catch_avg = df[['sgpmetE13.b1_org_precip_rate_mean_accumulated', 
                       'sgpmetE13.b1_pwd_precip_rate_mean_1min_accumulated', 
                       'sgpvdisC1.b1_rain_rate_accumulated', 
                       'sgpvdisE13.b1_rain_rate_accumulated',
                       'sgpdisdrometerC1.b1_rain_rate_accumulated',
                       'sgpdisdrometerE13.b1_rain_rate_accumulated',
                       'sgpstamppcpE13.b1_precip_accumulated', 
                       'sgpwbpluvio2C1.a1_intensity_rtnrt_accumulated',
                       'sgpldE13.b1_precip_rate_accumulated', 
                       'sgpldC1.b1_precip_rate_accumulated', 
                       'sgpaosmetE13.a1_rain_intensity_accumulated',
                       'sgpmwr3cC1.b1_rain_intensity_accumulated']].mean(axis=1)
    
    weight_80_20 = (tb_wb_avg * 0.8) + (no_catch_avg * 0.2)
    
    tb_wb_avg_total = round(tb_wb_avg[-1], 2)
    no_catch_avg_total = round(no_catch_avg[-1], 2)
    weight_80_20_total = round(weight_80_20[-1], 2)
    
    print('Total tb_wb_avg: ', tb_wb_avg_total)
    print('Total no_catch_avg: ', no_catch_avg_total)
    print('Total weight_80_20: ', weight_80_20_total)
    #JK end 
    
    # Drop any non-rain rate variables
    for d in df:
        if obj[d].attrs['units'] != 'mm/hr':
            df = df.drop(d,1)

    columns = df.columns.tolist()
    columns.append(' ')
    bins_0_25 = np.linspace(0,25, 51)
    bins_25_50 = np.linspace(25,50, 26)
    bins_50_75 = np.linspace(50,75, 11)
    bins_75_100 = np.linspace(75,125, 11)

    bins = np.unique(np.concatenate((bins_0_25, bins_25_50, bins_50_75, bins_75_100)))
    grid = np.zeros([len(columns), len(bins)])
      
    # For each time, cluster rain rates and take mean of
    # cluster with most instruments
    prec = []
    cols = df.columns
    for i in range(len(df.index)):
        data = np.asarray(df.iloc[i])

        # If row does not have recorded precip, continue
        z_idx = data > 0
        z_index = np.where(z_idx)[0]

        # Set number of clusters here
        if len(z_index) <= clusters - 1:
            prec.append(0.)
            continue
        if z_index[0] == -1:
            prec.append(0.)
            continue

        # Only run clustering on non-zero data
        data_n0 = data[z_index]

        # Running scipy kmeans, using # clusters
        y, _ = kmeans(data_n0, clusters)

        # Get indice of cluster with most instruments
        cluster_indices, _ = vq(data_n0, y)
        counts = Counter(cluster_indices)
        clust = counts.most_common(1)[0][0]

        # Take mean of cluster
        idx = cluster_indices == clust
        index = np.where(idx)[0]

        if sum(data_n0[index]) == 0 and len(np.where(~idx)[0]) > 1:
            index = np.where(~idx)[0]
        prec.append(np.nanmean(data_n0[index])/60.)

        rr_ind = (np.abs(bins - np.nanmean(data_n0[index]))).argmin()
        
        if np.nanmean(data_n0[index]) > 0.:
            grid[z_index[index], rr_ind] += 1



    # Add BE to object
    atts = {'units': 'mm', 'long_name': 'Best Estimate'}
    da = xr.DataArray(prec, dims=['time'], coords=[obj['time'].values], attrs=atts)
    da.to_netcdf('./kmeans_precip/kmeans_cluster_high_qc_sum'+str(clusters)+'.nc')
    obj['precip_be'] = da
    obj = act.utils.data_utils.accumulate_precip(obj, 'precip_be')

    df = df.sort_index()

    ds = df.to_xarray()
    ds = ds.fillna(0)

    #grid = grid / np.max(grid, axis=0)
    grid = grid / np.sum(grid, axis=0)
    #grid = np.divide(grid, np.reshape(np.sum(grid, axis=1), (-1, 1)))

    # Write data out to netcdf.  Note, weights is technically not correct
    c_id = ['novalynx_tbrg', 'opt_sci_org', 'vaisala_pwd', 'joanneum_vdis', 'distromet_disdrometer',
            'texas_elec_tbrg', 'ott_pluvio2', 'ott_parsivel2', 'vaisala_wxt', 'dummy']
    grid_obj = xr.Dataset({'weights': (['instruments','rain_rate'], grid),
                           'rain_rate': ('rain_rate', bins),
                           'instruments': ('instruments', c_id)})
    grid_obj = grid_obj.fillna(0)


    grid_obj['weights'] = np.round(grid_obj['weights'].rolling(rain_rate=5, min_periods=1, keep_attrs=True).mean(),3)

    grid_obj.to_netcdf('./weights/cluster_'+str(clusters)+'_high_qc_sum_norm.nc')
    grid_obj.close()

    print(time.time() - t)
    # Create plot with accumulations on top and heatmap on bottom
    labels = ['NovaLynx Tipping Bucket', 'Optical Scientific Optical Rain Gauge', 'Vaisala Present Weather Detector',
              'Joanneum Research Video Disdrometer', 'Distromet Impact Disdrometer',
              'Texas Electronics Tipping Bucket', 'Pluvio 2 Weighing Bucket', 'Parsivel 2', 'WXT-520', 'K-Means Best Estimate']
    fig, ax = plt.subplots(nrows=2, figsize=(16,10))
    ct = 0
    for d in obj:
        if 'accumulated' not in d:
            continue
        #if 'precip_be' in d:
        #    continue
        lab = labels[ct] + ': '+ str(round(obj[d].values[-1],2))
        ax[0].plot(obj['time'], obj[d], label=lab)
        ct += 1
    ax[0].set_xlim([df.index[0], df.index[-1]])
    ax[0].legend(loc=2)

    #im = ax[1].pcolormesh(bins, columns, grid, norm=colors.LogNorm(vmin=0.1, vmax=40000), cmap='jet')
    im = ax[1].pcolormesh(bins, columns, grid_obj['weights'].values, vmin=0, cmap='jet')
    for label in ax[1].yaxis.get_ticklabels():
        label.set_verticalalignment('bottom')
    fig.colorbar(im, ax=ax[1], orientation='horizontal', shrink=0.5, pad=0.05, aspect=30)
    fig.tight_layout(h_pad=0.05, w_pad=0.05)
    plt.show()
    obj.close()

    print('Weights Plot')
    fig, ax = plt.subplots(nrows=len(columns)-1, figsize=(16,10), sharex=True, gridspec_kw = {'wspace':0, 'hspace':0}, sharey=True)

    for i,d in enumerate(columns):
        if i == len(columns) - 1:
           continue
        ax[i].plot(bins, grid_obj['weights'].values[i,:], label=labels[i])
        ax[i].legend(loc=1)
    fig.tight_layout()
    plt.show()

    grid_obj.close()
    
#JK Plot the catchment vs. non-catchment averages and 80/20 BE
ax=tb_wb_avg.plot(x = 'time', y = '...', kind = 'line', label = 'TB and WB Avg: {0}'.format(tb_wb_avg_total))
no_catch_avg.plot(ax=ax, label = 'Non-Catchment Avg: {0}'.format(no_catch_avg_total))
weight_80_20.plot(ax=ax, label = '80/20 Weight BE: {0}'.format(weight_80_20_total))
ax.legend()
plt.show()