using NCDatasets

era_u_ds = NCDataset("era5_u_843.nc")
era_v_ds = NCDataset("era5_v_843.nc")


era_u_ds["u"][:,:,1,1]