from netCDF4 import Dataset
import numpy as np


def write_data_to_netCDF_file(lats, lons, codes):
    # lats and lons can be Python lists or numpy arrays
    rootgrp = Dataset("radar_data.nc", mode="w", format="NETCDF4")
    lat = rootgrp.createDimension("lat", len(lats))
    lon = rootgrp.createDimension("lon", len(lons))
    cod = rootgrp.createDimension("cod", len(codes))
    latitudes = rootgrp.createVariable("lat", "f4", (lat,))
    longitudes = rootgrp.createVariable("lon", "f4", (lon,))
    color_codes = rootgrp.createVariable("cod", "u1", (cod,))

    latitudes[:] = np.array(lats)
    longitudes[:] = np.array(lons)
    color_codes[:] = np.array(codes)

    rootgrp.close()


if __name__ == "__main__":
    test_latitudes = np.array([39.17629, 41.4082, 39.37977])
    test_longitudes = np.array([-0.25102,  1.88499,  2.78506])
    test_color_codes = np.array([12, 18, 24])
    write_data_to_netCDF_file(test_latitudes, test_longitudes, test_color_codes)
    print("test latitudes: {}".format(test_latitudes))
    print("test longitudes: {}".format(test_longitudes))
    print("test color codes: {}".format(test_color_codes))

    datafile = Dataset("radar_data.nc", mode="r", format="NETCDF4")
    read_latitudes = datafile.variables["lat"][:]
    read_longitudes = datafile.variables["lon"][:]
    read_color_codes = datafile.variables["cod"][:]
    print("read latitudes: {}".format(read_latitudes))
    print("read longitudes: {}".format(read_longitudes))
    print("read color codes: {}".format(read_color_codes))
    datafile.close()
