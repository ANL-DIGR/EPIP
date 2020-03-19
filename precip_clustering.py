import act
import glob
import xarray as xr
import dask
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
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

    files = glob.glob('./sgpprecip/sgpprecip*20190508*')

    obj = act.io.armfiles.read_netcdf(files)
    obj = obj.fillna(0)
    for v in obj:
        if obj[v].attrs['units'] != 'mm/hr':
            continue
        obj = act.utils.data_utils.accumulate_precip(obj, v, time_delta=60.)

    df = obj.to_dataframe()

    for d in df:
        #if 'accumulated' not in d:
        #    df = df.drop(d,1)
        if obj[d].attrs['units'] != 'mm/hr':
            df = df.drop(d,1)


    #y = KMeans(n_clusters=2, random_state=0).fit_predict(df.iloc[0])
    prec = []
    for i in range(len(df.index)):
        data = np.asarray(df.iloc[i])
        z_idx = data != 0
        z_index = np.where(z_idx)
        if z_index[0][0] == -1 or len(z_index[0]) <= 1:
            prec.append(0.)
            continue

        data_n0 = data[z_index]

        y, _ = kmeans(data_n0, 2)

        cluster_indices, _ = vq(data_n0, y)
        counts = Counter(cluster_indices)
        clust = counts.most_common(1)[0][0]

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
