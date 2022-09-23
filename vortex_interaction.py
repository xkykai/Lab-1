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
  'C:\\Users\\xinle\\Downloads\\ERA5\\350hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\400hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\450hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\500hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\550hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\600hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\650hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\700hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\750hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\800hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\850hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\875hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\900hPa.nc',
  'C:\\Users\\xinle\\Downloads\\ERA5\\925hPa.nc',
]

levels = [350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 875, 900, 925]

lat_slice = slice(35,0)
lon_slice = slice(210, 300)
time_slice = slice(16, None)

# data_450 = xr.open_dataset('C:\\Users\\xinle\\Downloads\\ERA5\\450hPa.nc')
# data_550 = xr.open_dataset('C:\\Users\\xinle\\Downloads\\ERA5\\650hPa.nc')
# data_650 = xr.open_dataset('C:\\Users\\xinle\\Downloads\\ERA5\\550hPa.nc')
# data_750 = xr.open_dataset('C:\\Users\\xinle\\Downloads\\ERA5\\750hPa.nc')

# data_650 = xr.open_dataset('/content/drive/MyDrive/12.843/650hPa.nc', decode_times=True)
# data_750 = xr.open_dataset('/content/drive/MyDrive/12.843/750hPa.nc', decode_times=True)

# data_350 = xr.open_dataset('https://engaging-web.mit.edu/~xinkai/12.843/ERA5/350hPa.nc#mode=bytes')
# data_450 = xr.open_dataset('https://engaging-web.mit.edu/~xinkai/12.843/ERA5/450hPa.nc#mode=bytes')
# data_550 = xr.open_dataset('https://engaging-web.mit.edu/~xinkai/12.843/ERA5/550hPa.nc#mode=bytes')
# data_650 = xr.open_dataset('https://engaging-web.mit.edu/~xinkai/12.843/ERA5/450hPa.nc#mode=bytes')

ds = []

for i, file in enumerate(DATASET_PATH):

	data = xr.open_dataset(file).sel(latitude=lat_slice).sel(longitude=lon_slice).expand_dims(level=[levels[i]], axis=1)

	time_offset = np.datetime64(int(data.coords["time"].values[0]), "h") - np.datetime64("2017-07-20T00", "h")
	data.coords["time"] = np.array([np.datetime64(int(time), 'h') - time_offset for time in data.coords["time"].values])

	data = data.isel(time=time_slice)

	ds.append(data)

	del data

ds = xr.concat(ds, dim="level")

#%%