from image_filter import filter_image
from netCDF_writer import write_data_to_netCDF_file
from numpy import ndenumerate

filtered_pixels = filter_image("aemet_ba_202205310000.gif")
latitudes = []
longitudes = []
codes = []

for index, pixel in ndenumerate(filtered_pixels):
    if pixel.any():  # even one nonzero pixel
        latitudes.append(index[0])
        longitudes.append(index[1])
        codes.append(pixel  )

write_data_to_netCDF_file(latitudes, longitudes, codes)
