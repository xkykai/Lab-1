#%%
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
# import numdifftools as nd
import math

# global variable
RE = 6371.0e3  # Earth's radius
#%%
# calculate derivatives
def pzpx(z: xr.DataArray):
  # calculate zonal derivative
  lon = z.coords['longitude']
  lat = z.coords['latitude']
  dlon = lon[1].values - lon[0].values
  
  # dx varies with lat
  dx = (RE * np.cos(lat * np.pi / 180.) * dlon * np.pi / 180.).values

  field = z.values
  shape_f = list(field.shape) # (time, lat, lon)
  deriv = np.full(shape_f, np.nan)
  deriv[..., :, 1:-1] = (field[..., :, 2:] - field[..., :, :-2]) / (2*dx[None, :, None])
  deriv[..., :, 0] = 2*deriv[..., :, 1] - deriv[..., :, 2]
  deriv[..., :, -1] = 2*deriv[..., :, -2] - deriv[..., :, -3]

  out = xr.DataArray(deriv, dims=z.dims, coords=z.coords)
  return out
  
def pzpy(z: xr.DataArray):
  # calculate meridional derivative
  lon = z.coords['longitude']
  lat = z.coords['latitude']

  # dy does not vary with lat or lon
  dy = RE * (lat[1].values - lat[0].values) * np.pi / 180.

  field = z.values
  shape_f = list(field.shape) # (time, lat, lon)
  deriv = np.full(shape_f, np.nan)
  deriv[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2*dy)
  deriv[..., 0, :] = 2*deriv[..., 1, :] - deriv[..., 2, :]
  deriv[..., -1, :] = 2*deriv[..., -2, :] - deriv[..., -3, :]

  out = xr.DataArray(deriv, dims=z.dims, coords=z.coords)
  return out

def calc_vort(u: xr.DataArray, v: xr.DataArray):
  return pzpx(v) - pzpy(u)
#%%
# data_450 = xr.open_dataset('/home/users/xinkai/MIT/12.843/Data/450hPa.nc')
# data_550 = xr.open_dataset('/home/users/xinkai/MIT/12.843/Data/550hPa.nc')
# data_650 = xr.open_dataset('/home/users/xinkai/MIT/12.843/Data/650hPa.nc')
# data_750 = xr.open_dataset('/home/users/xinkai/MIT/12.843/Data/750hPa.nc')

DATASET_PATH = [
  'C:\\Users\\xinle\\Downloads\\ERA5\\450hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\650hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\550hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\750hPa.nc'
]

lat_slice = ()

data_450 = xr.open_dataset('C:\\Users\\xinle\\Downloads\\ERA5\\450hPa.nc')
data_550 = xr.open_dataset('C:\\Users\\xinle\\Downloads\\ERA5\\650hPa.nc')
data_650 = xr.open_dataset('C:\\Users\\xinle\\Downloads\\ERA5\\550hPa.nc')
data_750 = xr.open_dataset('C:\\Users\\xinle\\Downloads\\ERA5\\750hPa.nc')



# data_650 = xr.open_dataset('/content/drive/MyDrive/12.843/650hPa.nc', decode_times=True)
# data_750 = xr.open_dataset('/content/drive/MyDrive/12.843/750hPa.nc', decode_times=True)

# data_350 = xr.open_dataset('https://engaging-web.mit.edu/~xinkai/12.843/ERA5/350hPa.nc#mode=bytes')
# data_450 = xr.open_dataset('https://engaging-web.mit.edu/~xinkai/12.843/ERA5/450hPa.nc#mode=bytes')
# data_550 = xr.open_dataset('https://engaging-web.mit.edu/~xinkai/12.843/ERA5/550hPa.nc#mode=bytes')
# data_650 = xr.open_dataset('https://engaging-web.mit.edu/~xinkai/12.843/ERA5/450hPa.nc#mode=bytes')

for key in list(data_450.data_vars):
    data_450[key] = data_450[key].expand_dims(level=[450], axis=1)

for key in list(data_550.data_vars):
    data_550[key] = data_550[key].expand_dims(level=[550], axis=1)

for key in list(data_650.data_vars):
    data_650[key] = data_650[key].expand_dims(level=[650], axis=1)

for key in list(data_750.data_vars):
    data_750[key] = data_750[key].expand_dims(level=[750], axis=1)

data = xr.concat([data_450, data_550, data_650, data_750], dim="level")
data.coords["time"].values
# np.array(data.coords["time"].values, dtype='datetime64') - np.timedelta64(70, "Y")

time_offset = np.datetime64(int(data.coords["time"].values[0]), "h") - np.datetime64("2017-07-20T00", "h")
data.coords["time"] = np.array([np.datetime64(int(time), 'h') - time_offset for time in data.coords["time"].values])

del data_450, data_550, data_650, data_750
#%%