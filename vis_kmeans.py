import matplotlib
matplotlib.use('Agg')

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

    files = glob.glob('./sgpprecip/sgpprecip*20190520*')

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
    names = np.asarray(df.columns)
    prec = []
    fig, ax = plt.subplots()
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
        index = np.where(idx)
        if sum(data_n0[index]) == 0 and len(np.where(~idx)[0]) > 1:
            index = np.where(~idx)

        test_ind = np.asarray(z_index[0][index[0]])
        if np.nanmean(data_n0[index]) > 15:
            print(np.nanmean(data_n0[index]), names[test_ind])
        prec.append(np.nanmean(data_n0[index])/60.)

        color = np.repeat('k', len(data_n0))
        color[index] = 'r'
        ax.scatter(np.repeat(df.index[i], len(data_n0)), data_n0, c=color)
        ax.set_xlim([df.index[0], df.index[i]])
        plt.savefig('/Users/atheisen/Code/EPIP/images/vis_kmeans/'+str(i)+'.png')

    atts = {'units': 'm', 'long_name': 'Best Estimate'}
    da = xr.DataArray(prec, dims=['time'], coords=[obj['time'].values], attrs=atts)
    obj['precip_be'] = da
    obj = act.utils.data_utils.accumulate_precip(obj, 'precip_be')

    df = df.sort_index()

    ds = df.to_xarray()
    ds = ds.fillna(0)

    #for i in range(len(obj['time'].values)):
    #    for d in obj:
    #        if 'accumulated' not in d:
    #            continue
    #        lab = d + ': '+ str(round(obj[d].values[-1],2))
    #        ax.plot(obj['time'][i], obj[d][i])
    #    plt.show()
    #    sys.exit()
    #ax.set_xlim([df.index[0], df.index[-1]])
    #ax.legend()
