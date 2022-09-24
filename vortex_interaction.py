#%%
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
# import numdifftools as nd
import math
from scipy.optimize import curve_fit
import seaborn as sns

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

def earth_distance(lat_1, lon_1, lat_2, lon_2):
  # approximate radius of earth in m
  lat1 = np.radians(lat_1)
  lon1 = np.radians(lon_1)
  lat2 = np.radians(lat_2)
  lon2 = np.radians(lon_2)

  dlon = lon2 - lon1
  dlat = lat2 - lat1

  a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
  c = 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a))

  distance = RE * c

  return distance
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
ds["vorticity"] = calc_vort(ds["u"], ds["v"])
ds["speed"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
#%%
timestep = 27
pressure_level = 925

plt.figure()
ds["u"].isel(time=timestep).sel(level=pressure_level).plot.contourf(x="longitude")
plt.show()

plt.figure()
ds["v"].isel(time=timestep).sel(level=pressure_level).plot.contourf(x="longitude")
plt.show()

plt.figure()
ds["speed"].isel(time=timestep).sel(level=pressure_level).plot.contourf(x="longitude")
plt.show()

plt.figure()
ds["vorticity"].isel(time=timestep).sel(level=pressure_level).plot.contourf(x="longitude")
plt.show()
#%%
def find_local_max_ind(arr: xr.DataArray):
  field = arr.values
  lat_size = arr.coords['latitude'].size
  lon_size = arr.coords['longitude'].size
  mask = np.zeros((lat_size, lon_size))
  for ilat in range(2, lat_size-2):
    for ilon in range(2, lon_size-2):
      if field[ilat, ilon] == np.max(field[ilat-2:ilat+3, ilon-2:ilon+3]):
        mask[ilat, ilon] = 1
  max_ind = np.where(field*mask>0.0002)
  max_ind1 = 0
  max_ind2 = 1
  if len(max_ind[0])>2:
    vort_list = [field[max_ind[0][i], max_ind[1][i]] for i in range(len(max_ind[0]))]
    max_ind1 = vort_list.index(max(vort_list))
    vort_list[max_ind1] = 0
    max_ind2 = vort_list.index(max(vort_list))
  tc1_lat = arr.coords['latitude'][max_ind[0][max_ind1]].values
  tc1_lon = arr.coords['longitude'][max_ind[1][max_ind1]].values
  tc2_lat = arr.coords['latitude'][max_ind[0][max_ind2]].values
  tc2_lon = arr.coords['longitude'][max_ind[1][max_ind2]].values
  return tc1_lat, tc1_lon, tc2_lat, tc2_lon

pressure_level = 925 # locate the TCs at 925hPa
tc1_lat = np.zeros(ds["time"].size)
tc1_lon = np.zeros(ds["time"].size)
tc2_lat = np.zeros(ds["time"].size)
tc2_lon = np.zeros(ds["time"].size)
for nt in range(ds["time"].size):
  local_vort = ds["vorticity"].isel(time=nt).sel(level=pressure_level, latitude=slice(19+10*nt/32., 8+8*nt/32.), longitude=slice(240-20*nt/32., 260-20*nt/32.))
  tc1_lat_tmp, tc1_lon_tmp, tc2_lat_tmp, tc2_lon_tmp = find_local_max_ind(local_vort)
  if nt==0:
    tc1_lat[nt] = tc1_lat_tmp
    tc1_lon[nt] = tc1_lon_tmp
    tc2_lat[nt] = tc2_lat_tmp
    tc2_lon[nt] = tc2_lon_tmp
  else:
    d1 = earth_distance(tc1_lat[nt-1], tc1_lon[nt-1], tc1_lat_tmp, tc1_lon_tmp)
    d2 = earth_distance(tc1_lat[nt-1], tc1_lon[nt-1], tc2_lat_tmp, tc2_lon_tmp)
    if d1 < d2:
      tc1_lat[nt] = tc1_lat_tmp
      tc1_lon[nt] = tc1_lon_tmp
      tc2_lat[nt] = tc2_lat_tmp
      tc2_lon[nt] = tc2_lon_tmp
    else:
      tc1_lat[nt] = tc2_lat_tmp
      tc1_lon[nt] = tc2_lon_tmp
      tc2_lat[nt] = tc1_lat_tmp
      tc2_lon[nt] = tc1_lon_tmp

timestep=32
ds["vorticity"].isel(time=timestep).sel(level=pressure_level).plot.contourf(x="longitude")
plt.plot(tc1_lon, tc1_lat, label='TC1')
plt.plot(tc2_lon, tc2_lat, label='TC2')
plt.legend()
plt.show()
#%%
import matplotlib.animation
from matplotlib.animation import FuncAnimation

plt.rcParams["animation.html"] = "jshtml"
# plt.rcParams['figure.dpi'] = 150  
plt.ioff()
fig, ax = plt.subplots(figsize=(8, 4))

def animate(t):
  shift = 0
  plt.cla()
  ds["vorticity"].isel(time=t+shift).sel(level=925).plot.contourf(x='longitude', levels=np.linspace(-0.001, 0.001, 11), add_colorbar=False)
  # plot track
  plt.plot(tc1_lon[:t+shift+1], tc1_lat[:t+shift+1], label='TC1')
  plt.plot(tc2_lon[:t+shift+1], tc2_lat[:t+shift+1], label='TC2')
  plt.legend()

matplotlib.animation.FuncAnimation(fig, animate, frames=32)

#%%
# plot meridional cross section

ntime = 15
print(tc1_lon[ntime])
ds["vorticity"].isel(time=ntime).sel(longitude=tc1_lon[ntime]).plot.contourf(x='latitude', add_colorbar=True)
plt.gca().invert_yaxis()
plt.show()

ds["vorticity"].isel(time=ntime).sel(longitude=tc2_lon[ntime]).plot.contourf(x='latitude', add_colorbar=True)
plt.gca().invert_yaxis()
plt.show()
#%% 
# plot meridional cross section for potential temperature

ntime = 15

gas_constant = 8.3145 / 28.97
c_p = 1

kappa = gas_constant / c_p

p_reference = 1000

ds["potential temperature"] = ds["temperature"] * (p_reference / ds["level"]) ** kappa

ds["potential temperature"].isel(time=ntime).sel(longitude=tc1_lon[ntime]).plot.contourf(x='latitude', add_colorbar=True)
plt.gca().invert_yaxis()
plt.show()

ds["potential temperature"].isel(time=ntime).sel(longitude=tc2_lon[ntime]).plot.contourf(x='latitude', add_colorbar=True)
plt.gca().invert_yaxis()
plt.show()
#%%
# plot distance (km) between TCs

dis_TC12 = [earth_distance(tc1_lat[i], tc1_lon[i], tc2_lat[i], tc2_lon[i])/1000. for i in range(ds["time"].size)]
plt.figure(figsize=(8, 4))
plt.plot(ds["time"], dis_TC12)
plt.show()
#%%
# plot angular average, max distance = 300km
ntime = 0
level=925
arr = ds["speed"].isel(time=ntime).sel(level=level, latitude=slice(tc1_lat[ntime]+5, tc1_lat[ntime]-5), longitude=slice(tc1_lon[ntime]-5, tc1_lon[ntime]+5))
inv_distance = []
value = []
for i in range(arr.coords['latitude'].size):
  for j in range(arr.coords['longitude'].size):
    dis = earth_distance(tc1_lat[ntime], tc1_lon[ntime], arr.coords['latitude'][i], arr.coords['longitude'][j])
    if dis < 500.0e3 and dis > 0:
      inv_distance.append(1./(dis/1000.))
      value.append(arr[i, j].values)
plt.figure(figsize=(8, 4))
plt.scatter(inv_distance, value)
plt.show()
#%%
levels_used = levels[7:]
levels_used = [925]
average_speeds = []
distances = []

time = ds["time"][0]

vortex_area = ds["speed"].sel(time=time, level=levels_used, 
                              latitude=slice(tc1_lat[i] + 5, tc1_lat[i] - 5), 
                              longitude=slice(tc1_lon[i] - 5, tc1_lon[i] + 5))
for lat in vortex_area["latitude"]:
  for lon in vortex_area["longitude"]:
    r = earth_distance(tc1_lat[i], tc1_lon[i], lat, lon)
    if r < 500e3 and r != 0:
      distances.append(r)
      average_speeds.append(np.mean(ds["speed"].sel(latitude=lat, longitude=lon, level=levels_used, time=time).values))
#%%
log_distances = np.log(np.array(distances))
log_speeds = np.log(np.array(average_speeds))
plt.scatter(log_distances, log_speeds)
plt.show()
#%%
log_distances_fit = []
log_speeds_fit = []

for i in range(len(log_distances)):
  if log_distances[i] > 12:
    log_distances_fit.append(log_distances[i])
    log_speeds_fit.append(log_speeds[i])
  
log_distances_fit = np.array(log_distances_fit)
log_speeds_fit = np.array(log_speeds_fit)

def linear(x, m, c):
  return m * x + c

m_guess = -1
c_guess = 30

p0 = np.array([m_guess, c_guess])

fit, cov = curve_fit(linear, log_distances_fit, log_speeds_fit, p0=p0)

x_fit = np.linspace(11.5, np.amax(log_distances), 100)
y_fit = linear(x_fit, *fit)

#%%
sns.set_style("darkgrid")
sns.scatterplot(x=log_distances, y=log_speeds, label="ERA5 data", alpha=0.5)
sns.lineplot(x=x_fit, y=y_fit, color="orange", label=f"Best fit line, gradient = {np.round(fit[0], 2)}")
plt.xlabel(r"$\log$(Distance from Hurricane Center / m)")
plt.ylabel(r"$\log$(Wind speed / m s$^{-1}$)")
plt.show()
#%%
def distance_to_latlon(initial_lon, initial_lat, displacement):
  final_lat = initial_lat + displacement[1] * 360 / (2 * np.pi * RE)
  final_lon = initial_lon + displacement[0] * 360 / (2 * np.pi * RE * np.cos(math.radians(initial_lat)))
  return final_lon, final_lat

def perpendicular(a):
  b = np.empty_like(a)
  b[0] = -a[1]
  b[1] = a[0]
  return b / np.linalg.norm(a)

def point_vortex_interaction(zeta_1, zeta_2, lon_1, lat_1, lon_2, lat_2, nsteps, dt=600):
  lon_1s = np.zeros(nsteps + 1)
  lat_1s = np.zeros(nsteps + 1)
  lon_2s = np.zeros(nsteps + 1)
  lat_2s = np.zeros(nsteps + 1)

  lon_1s[0] = lon_1
  lat_1s[0] = lat_1
  lon_2s[0] = lon_2
  lat_2s[0] = lat_2

  for i in range(nsteps):
    lat_1_rad = np.radians(lat_1)
    lon_1_rad = np.radians(lon_1)
    lat_2_rad = np.radians(lat_2)
    lon_2_rad = np.radians(lon_2)

    dlon_rad = lon_2_rad - lon_1_rad
    dlat_rad = lat_2_rad - lat_1_rad

    dx_1_to_2 = RE * np.cos(lon_1_rad) * dlon_rad
    dy_1_to_2 = RE * dlat_rad

    r_1_to_2 = np.array([dx_1_to_2, dy_1_to_2])
    r_2_to_1 = - 1 * r_1_to_2

    distance = np.linalg.norm(r_1_to_2)

    u_1_on_2 = perpendicular(r_1_to_2) * zeta_1  / ( 2 * np.pi * distance)
    u_2_on_1 = perpendicular(r_2_to_1) * zeta_2  / ( 2 * np.pi * distance)

    lon_2, lat_2 = distance_to_latlon(lon_2, lat_2, u_1_on_2 * dt)
    lon_1, lat_1 = distance_to_latlon(lon_1, lat_1, u_2_on_1 * dt)

    lon_1s[i+1] = lon_1
    lat_1s[i+1] = lat_1
    lon_2s[i+1] = lon_2
    lat_2s[i+1] = lat_2

  return lon_1s, lat_1s, lon_2s, lat_2s

#%%
levels_used = levels[7:]

zeta_1s = np.zeros(len(ds["time"]))
for i, time in enumerate(ds["time"]):
  zeta_1s[i] = np.mean(ds["vorticity"].sel(level=levels_used, latitude=tc1_lat[i], longitude=tc1_lon[i], time=time)).values

zeta_2s = np.zeros(len(ds["time"]))
for i, time in enumerate(ds["time"]):
  zeta_2s[i] = np.mean(ds["vorticity"].sel(level=levels_used, latitude=tc2_lat[i], longitude=tc2_lon[i], time=time)).values

plt.plot(ds["time"], zeta_1s)
plt.ylabel("Vorticity of TC1")
plt.show()

plt.plot(ds["time"], zeta_2s, label="Vorticity of TC2")
plt.ylabel("Vorticity of TC2")
plt.show()
#%%
# zeta_1_time_average = np.mean(zeta_1s) * 10e11
# zeta_2_time_average = np.mean(zeta_2s) * 10e11

zeta_1_time_average = np.exp(fit[1])
zeta_2_time_average = zeta_1_time_average / 2

# vortex_1_lons, vortex_1_lats, vortex_2_lons, vortex_2_lats = point_vortex_interaction(zeta_1_time_average, zeta_2_time_average, tc1_lon[0], tc1_lat[0], tc2_lon[0], tc2_lat[0], 10000, dt=3600)
vortex_1_lons, vortex_1_lats, vortex_2_lons, vortex_2_lats = point_vortex_interaction(zeta_1_time_average, zeta_2_time_average, 0, 0, 0, 1, 5000, dt=60)

plt.style.use('default')

plt.scatter(vortex_1_lons, vortex_1_lats)
plt.scatter(vortex_2_lons, vortex_2_lats)
plt.show()
#%%