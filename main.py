from image_filter import filter_image
from netCDF_writer import write_data_to_netCDF_file
from PIL import Image

filtered_pixels = filter_image("examples/aemet_ba_202205310030.gif")
img = Image.fromarray(filtered_pixels)
img.save("result.png")

latitudes = []
longitudes = []
codes = []

for x, row in enumerate(filtered_pixels):
    for y, pixel in enumerate(row):
        if pixel.any():  # even one nonzero pixel
            latitudes.append(x)
            longitudes.append(y)
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 252:
                codes.append(12)
            elif pixel[0] == 0 and pixel[1] == 148 and pixel[2] == 252:
                codes.append(18)
            elif pixel[0] == 0 and pixel[1] == 252 and pixel[2] == 252:
                codes.append(24)
            elif pixel[0] == 67 and pixel[1] == 131 and pixel[2] == 36:
                codes.append(30)
            elif pixel[0] == 0 and pixel[1] == 223 and pixel[2] == 0:
                codes.append(42)
            elif pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 0:
                codes.append(48)
            elif pixel[0] == 255 and pixel[1] == 187 and pixel[2] == 0:
                codes.append(54)
            else:
                codes.append(0)

write_data_to_netCDF_file(latitudes, longitudes, codes)
