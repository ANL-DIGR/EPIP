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

if __name__ == '__main__':
    """
    Program to run through all the parsed data and create a 
    precipitation best estimate based on clustering
    """

    files = glob.glob('./sgpprecip/sgpprecip*201905*')

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

    # For each time, cluster rain rates and take mean of
    # cluster with most instruments
    prec = []
    for i in range(len(df.index)):
        data = np.asarray(df.iloc[i])

        # If row does not have recorded precip, continue
        z_idx = data != 0
        z_index = np.where(z_idx)

        # Set number of clusters here
        clusters = 2
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

    atts = {'units': 'm', 'long_name': 'Best Estimate'}
    da = xr.DataArray(prec, dims=['time'], coords=[obj['time'].values], attrs=atts)
    obj['precip_be'] = da
    obj = act.utils.data_utils.accumulate_precip(obj, 'precip_be')

    df = df.sort_index()

    ds = df.to_xarray()
    ds = ds.fillna(0)

    fig, ax = plt.subplots()
    for d in obj:
        if 'accumulated' not in d:
            continue
        lab = d + ': '+ str(round(obj[d].values[-1],2))
        ax.plot(obj['time'], obj[d], label=lab)
    ax.set_xlim([df.index[0], df.index[-1]])
    ax.legend()
    plt.show()
