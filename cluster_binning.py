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

if __name__ == '__main__':
    """
    Program to test out different clustering numbers to determine how the instruments
    compare across rain rates
    """

    files = glob.glob('./sgpprecip/sgpprecip*201*')

    # Open files and accumulate precipitation
    obj = act.io.armfiles.read_netcdf(files)
    obj = obj.fillna(0)
    for v in obj:
        if obj[v].attrs['units'] != 'mm/hr':
            continue
        obj = act.utils.data_utils.accumulate_precip(obj, v, time_delta=60.)

    # Convert to pandas dataframe
    df = obj.to_dataframe()

    # Drop any non-rain rate variables
    for d in df:
        if obj[d].attrs['units'] != 'mm/hr':
            df = df.drop(d,1)

    columns = df.columns.tolist()
    columns.append(' ')
    bins = np.linspace(0,105, 106)
    grid = np.zeros([len(columns), len(bins)])

    # For each time, cluster rain rates and take mean of
    # cluster with most instruments
    prec = []
    for i in range(len(df.index)):
        data = np.asarray(df.iloc[i])

        # If row does not have recorded precip, continue
        z_idx = data != 0
        z_index = np.where(z_idx)

        # Set number of clusters here
        clusters = 3
        if z_index[0][0] == -1 or len(z_index[0]) <= clusters - 1:
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
        
        if np.nanmean(data_n0[index]) > 0:
            grid[index, rr_ind] += 1


    # Add BE to object
    atts = {'units': 'm', 'long_name': 'Best Estimate'}
    da = xr.DataArray(prec, dims=['time'], coords=[obj['time'].values], attrs=atts)
    obj['precip_be'] = da
    obj = act.utils.data_utils.accumulate_precip(obj, 'precip_be')

    df = df.sort_index()

    ds = df.to_xarray()
    ds = ds.fillna(0)

    grid = grid / np.max(grid, axis=0)

    # Write data out to netcdf.  Note, weights is technically not correct
    grid_obj = xr.Dataset({'weights': (['instruments','rain_rate'], grid),
                           'rain_rate': ('rain_rate', bins),
                           'instruments': ('instruments', columns)})
    grid_obj.to_netcdf('./weights/cluster_3_max_norm.nc')
    grid_obj.close()

    # Create plot with accumulations on top and heatmap on bottom
    fig, ax = plt.subplots(nrows=2, figsize=(16,10))
    for d in obj:
        if 'accumulated' not in d:
            continue
        lab = d + ': '+ str(round(obj[d].values[-1],2))
        ax[0].plot(obj['time'], obj[d], label=lab)
    ax[0].set_xlim([df.index[0], df.index[-1]])
    ax[0].legend(loc=2)

    #im = ax[1].pcolormesh(bins, columns, grid, norm=colors.LogNorm(vmin=0.1, vmax=40000), cmap='jet')
    im = ax[1].pcolormesh(bins, columns, grid, vmin=0, cmap='jet')
    for label in ax[1].yaxis.get_ticklabels():
        label.set_verticalalignment('bottom')
    fig.colorbar(im, ax=ax[1], orientation='horizontal', shrink=0.5, pad=0.05, aspect=30)
    fig.tight_layout(h_pad=0.05, w_pad=0.05)
    plt.show()
    obj.close()

    fig, ax = plt.subplots(nrows=len(columns)-1, figsize=(16,10), sharex=True, gridspec_kw = {'wspace':0, 'hspace':0}, sharey=True)
    for i,d in enumerate(columns):
        if i == len(columns) - 1:
           continue
        ax[i].plot(bins, grid[i,:], label=d)
        ax[i].legend(loc=1)
    fig.tight_layout()
    plt.show()
