#####################################################################################
#This code includes a callable function that takes a GeoTiff format file, and returns
#pixel count of that file, the could percentage, and the shadow percentage.
#The workflow is
#1. take blue, green, red, and NearInfrared bands as b1,b2,b3,b4.
#2. get the max&min image.  The max image means every pixel takes the max value from b1 through b4 at that location.  
#3. apply a 7x7 max filter on max image, and a 7x7 min filter on min image.
#4. extract the cloud from min image, extract shadow(shadow plus water) from max image, land from NearInfrared.
#Arbitary threshold eas used for extracting cloud and shadows.
#5. get shadow by using land as a mask on shadow plus water image.
#6. count total pixels, cloud, shadow pixels and calculate cloud_perc and shadow_perc
########################################################################################

import gdal
import numpy as np
from scipy import ndimage
import sys
from os import listdir
import copy
import csv

def Cloud_Shadow_Stats(in_name):
    #1 open the tif, take 4 bands, and read them as arrays
    in_ds = gdal.Open(in_name)
    b1 = in_ds.GetRasterBand(1)
    b2 = in_ds.GetRasterBand(2)
    b3 = in_ds.GetRasterBand(3)
    b4 = in_ds.GetRasterBand(4)

    b1_array = b1.ReadAsArray()
    b2_array = b2.ReadAsArray()
    b3_array = b3.ReadAsArray()
    b4_array = b4.ReadAsArray()

    #2. make max image and min image.
    #np.dstack() takes a list of bands and makes a band stack
    #np.amax() find the max along the axis, here 2 means the axis that penetrates through bands in each pixel.
    band_list = [b1_array,b2_array,b3_array,b4_array]
    stacked = np.dstack(band_list)
    max_img = np.amax(stacked,2)
    min_img = np.amin(stacked,2)

    del b1_array, b2_array, b3_array, band_list

    #3. make max 7x7 filtered max and min image
    max7x7_img = ndimage.maximum_filter(max_img, 7)
    min7x7_img = ndimage.minimum_filter(min_img, 7)

    del max_img, min_img

    #4. extract cloud, shadow&water, land
    #The threshold here is based on Sitian and Tammy's test on 11 planet scenes.  It may not welly work for every AOI.
    #Apparently np.where() method will change or lost the datatype, so .astype(np.int16) is used to make sure the datatype is the same as original
    cloud_array = np.where(min7x7_img > 2000, 1, 0).astype(np.int16)
    shadow_and_water_array = np.where(max7x7_img < 1500, 1, 0).astype(np.int16)
    land_array = np.where(b4_array > 1000, 1, 0).astype(np.int16)

    del max7x7_img, min7x7_img, b4_array

    #5. get shadow by masking 
    shadow_array = np.where(land_array == 1, shadow_and_water_array, 0).astype(np.int16)

    #6. Calculate Statistics
    grid_count = np.ma.count(shadow_array)# acutally count all pixels 
    cloud_count = np.count_nonzero(cloud_array ==1)
    shadow_count = np.count_nonzero(shadow_array ==1)

    cloud_perc = cloud_count/grid_count
    shadow_perc = shadow_count/grid_count
    print("The total number of pixel in the grid is {}, the cloud perc is :{}, the shadow perc is {}".format(grid_count, cloud_perc, shadow_perc))

    del cloud_array, shadow_and_water_array, land_array, shadow_array
    return cloud_perc, shadow_perc

#1. List dir and extract filename contains "SR"(surface reflectance)
file_names = listdir(r"D:\Extracting_Clouds\OS\\")
tif_cells = [fn for fn in file_names if "SR" in fn if fn[-3:]=="tif"]

#print("total tif cells: {}\n".format(len(tif_cells)))
#2.extract cloud, shadow percs and calculate stats.
cs_list = []
for fn in tif_cells:
    cs_list.append([fn])

for cell in cs_list:
    cell.extend(Cloud_Shadow_Stats(r"D:\Extracting_Clouds\OS\\"+cell[0]))

#3. Write cloud shadow perc into csv file.
with open(r"D:\Extracting_Clouds\C2000_S1500_OS.csv", "w") as csvfile:
    headers = ["file_name", "cloud_perc", "shadow_perc"]
    
    writer = csv.writer(csvfile, lineterminator='\n')
    writer.writerow(headers)
    for val in cs_list:
        writer.writerow(val)
