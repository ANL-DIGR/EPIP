import xarray as xr
import numpy as np
import numpy as np
from glob import glob

# Filter out DSDs by diameter
min_diameter = 0.5
max_diameter = 5.
def calculate_fall_speed(diameter, density=1000, inplace=False):
    """ Calculate terminal fall velocity for drops.
    Parameters
    ----------
    diameter: array_like[float]
        Array of diameters to calculate fall speed for. 
    density: float, optional
        Density of the air in millibars. Defaults to 1000mb. 
    Returns
    -------
    terminal_fall_speed: array_like[float]
        Array of fall speeds matching size of diameter, adjusted for air density.
    """
    velocity = 9.65 - 10.3 * np.exp(-0.6 * diameter)
    speed_adjustment = (density / 1000.0) ** 0.4  # Based on Yu 2016
    terminal_fall_speed = velocity * speed_adjustment
    return terminal_fall_speed


# Fall speed filters
drops_file = './sgpvdisdropsC1.b1/*'
vdis_psd_file = './sgpvdisC1.b1/*'
vdis_files = glob(vdis_psd_file)    
# Filtered data paths
filtered_file_path = './sgpvdisfilteredC1.b1/'
drops_file_prefix = './sgpvdisdropsC1.b1/sgpvdisdropsC1.b1.'
for vdis_file in vdis_files:
    vdis_psd_ds = xr.open_mfdataset(vdis_file)
    vdis_global_attrs = vdis_psd_ds.attrs
    time = np.datetime_as_string(vdis_psd_ds['time'].values[0], 'D')
    year = time[:4]
    month = time[5:7]
    day = time[-2:]
    search_for = drops_file_prefix + year + month + day + '*'
    my_drop_file = glob(search_for)
    print(vdis_file, my_drop_file)
    if(len(my_drop_file) == 0):
        vdis_psd_ds.to_netcdf(filtered_file_path + vdis_file.split('/')[-1])
        vdis_psd_ds.close()
        continue
    drop_ds = xr.open_dataset(my_drop_file[0])
    drop_diams = drop_ds['equivolumetric_sphere_diameter'].values
    fall_speed = drop_ds['fall_speed'].values
    drop_times = drop_ds['time'].values
    bins = np.arange(0, 10.2, 0.2)
    psd = vdis_psd_ds.num_drops.values
    area = drop_ds.area.values
    volume = drop_ds.drop_volume.values
    volume_over_area = volume/area
    term_vels = calculate_fall_speed(drop_diams)
    good_fspeeds = np.logical_and(fall_speed > 0.5*term_vels, fall_speed < 1.5*term_vels)
    good_diams = np.logical_and(drop_diams > 0.2, drop_diams < 5)
    good_particles = np.logical_and(good_fspeeds, good_diams)
    drop_diams = drop_diams[good_particles]
    volume_over_area = volume_over_area[good_particles]
    drop_times = drop_times[good_particles]
    i = 0
    rain_rate = np.zeros_like(vdis_psd_ds.rain_rate.values)
    for times in vdis_psd_ds.time.values:
        end_time = times + np.timedelta64(1, 'm')
        time_inds = np.logical_and(drop_times >= times, drop_times < end_time)
        diams = drop_diams[time_inds]
        volareas = volume_over_area[time_inds]
        rain_rate[i] = np.sum(volareas) * 60
        hist, bins = np.histogram(diams, bins=bins)
        psd[i] = hist
        i = i + 1
    old_attrs_nd = vdis_psd_ds.num_drops.attrs
    vdis_psd_ds['num_drops'] = xr.DataArray(psd, dims=('time', 'drop_diameter'))
    vdis_psd_ds.attrs = old_attrs_nd
    vdis_psd_rain_attrs = vdis_psd_ds.rain_rate.attrs
    vdis_psd_ds['rain_rate'] = xr.DataArray(rain_rate, dims=('time'))
    vdis_psd_ds['rain_rate'].attrs = vdis_psd_rain_attrs
    vdis_psd_ds.attrs = vdis_global_attrs
    vdis_psd_ds.to_netcdf(filtered_file_path + vdis_file.split('/')[-1])
    drop_ds.close()
    vdis_psd_ds.close()
