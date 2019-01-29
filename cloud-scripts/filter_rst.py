#This code is create for Ron and other people who work with .rst format planet images.
#1. The code works with either large images (no upper limit, e.g., whole scene) or small cell images without run into the memory problem,
#   by processing the image in 640x640 pixel block in each time.
#
#2. Each function read the input .rst image on the disk and write the output image to the disk.
#   The code assumes a planet scene/AOI has four .rst images: Blue(b1), Green(b2), Red(b3), and Near Infrared(b4).
#
#3. This code provides "horizontal" functions, meaning the functions can be used for more purpose
#   than the cloud and shadow extraction algorithm. E.g., 
#       maxmin_img():    given many spectral bands, output one composite max or min image 
#       reclass():       reclass the image given a threshold
#       filter_window(): does spatial filter given an image; this function process multi lines rather than block, due to the edge effect of filter.
#       masks():         mask image given an image and a mask
#  These "horizontal" functions provide flexibility for future change of the algorithm.
#For a "vertical" version (compact, for cloud and shadow extraction purpose only), see filter_callable.py.  Note that filter_callable.py
#does not have any image output, but only numerical output for statistics.


#The cloud-shadow extraction algorithm family codes:
# filter_rst.py: works with .rst format; outputs are images; horizontal functions (flexible if the algorithm is going to changw), process image by block. 
# filter_tif.py: works with .tif format; outputs are images; horizontal functions, process image by block.
# filter_callable.py : works with numpy array; outputs are cloud-shadow fraction statistics; vertical function (work for this algorithm only) 




import gdal
import numpy as np
from scipy import ndimage
import os


def maxmin_img(m,out_name,*args):
    """
    This function creates a maximum or minimum image, which each pixel takes the 
    max or minimum value at that location from the input bands.
    The purpose of this funciton is to help extract shadows and clouds, assuming 
    that the shadow's maximum reflectance is still low, whereas the cloud's minimum 
    reflectance is still high.
    
    m           - specify either "max" or "min"
    out_name    - The out put image name. e.g., D:\output.rst
    *args       - input .rst format images' full path.
    """
    try:
        #1. Get the GeoTransform and Projection from the first band of the input bands
        #(bands should have same geotransform and projection)
        #Also get the columns and rows
        in_ds_template = gdal.Open(args[0])
        in_ds_GeoTransform = in_ds_template.GetGeoTransform()
        in_ds_Projection = in_ds_template.GetProjection()

        cols = in_ds_template.GetRasterBand(1).XSize
        rows = in_ds_template.GetRasterBand(1).YSize

        #2. Create a list of opened bands for later iterations
        bands= [] 
        for arg in args:
            bands.append(gdal.Open(arg)) 

        
        #3. Create an empty output file. The gdal driver will actually create a rst file and a rdc file. 
        #The rst is the image, and the rdc file holds metadata.  We copied the metadata from input band, since they are almost the same,
        # except the max/value and displayed max/min value.
        rst_driver = gdal.GetDriverByName('rst')
        out_ds = rst_driver.Create(out_name, cols, rows,1,3) #here 1 stands for create 1 band, 3 stands for integer data type. 
        out_band = out_ds.GetRasterBand(1)

        #4. use nested for loop to read 640x640 block each time for all input bands, and append these
        #band blocks into a list.  Then use np.max/np.min method to get max/min image along the axis-2,
        # which is the penetration direction of a pixel thorugh a stack of bands. 
        block_size = 640
        for col_start in range(0, cols, block_size): 
            if col_start + block_size < cols:
                col_read = block_size
            else:
                col_read = cols - col_start
            for row_start in range(0, rows, block_size):
                if row_start + block_size < rows:
                    row_read = block_size
                else:
                    row_read = rows - row_start


                bands_blocks = []     #bands in current block
                for band in bands:
                    current_block= band.ReadAsArray(col_start, row_start, col_read, row_read)
                    bands_blocks.append(current_block)
                
                block_stack = np.dstack(bands_blocks)  #stack the bands along axis2(layers of a pixel)
                if m == "max":
                    m_arry = np.amax(block_stack,2).astype(np.int16)
                elif m =="min":
                    m_arry = np.amin(block_stack,2).astype(np.int16)
                out_band.WriteArray(m_arry, col_start, row_start)

        #out_band.FlushCache()
        out_ds.SetGeoTransform(in_ds_GeoTransform)
        out_ds.SetProjection(in_ds_Projection)

    except:
        return "Failed to generate max/min image."
        #with open(out_name.replace(out_name[-3:],"rdc"),"w") as out_rdc:
        #    for line in meta_data:
        #        print(line.rstrip("\n"))
        #        out_rdc.write(line)

        #write rdc for test in Terrset
        #out_rdc = out_name[:-3]+"rdc"
        #with open(out_rdc, "w") as f:

def reclass(cd,in_name,out_name, thre_val):
    """
    Output is a boolin image where 0 is the pixels not meet the condition,
    and 1 means the pixels that meet the condition.
    cd       - ">" or "<"
    in_name  - input image name, with path
    out_name - out put image name, with path
    thre_val - threshold value 
    """
    try:
        #1. Open the input image and get the GeoTransform, Projection, 
        #cols and rows.
        in_ds = gdal.Open(in_name)
        in_ds_GeoTransform = in_ds.GetGeoTransform()
        in_ds_Projection = in_ds.GetProjection()

        in_band = in_ds.GetRasterBand(1)
        cols = in_band.XSize
        rows = in_band.YSize

        #2.Create a output file. The gdal driver will actually create a rst file and a rdc file. 
        #The rdc file holds metadata.  The output metadata will set by using the input metadata, 
        # except the max/min value and display max/min value.
        rst_driver = gdal.GetDriverByName('rst')
        out_ds = rst_driver.Create(out_name, cols, rows,1,3) # here 1 means 1 band in the image,3 means integer datatype
        out_band = out_ds.GetRasterBand(1)

        block_size = 640
        for col_start in range(0, cols, block_size): 
            if col_start + block_size < cols:
                col_read = block_size
            else:
                col_read = cols - col_start
            for row_start in range(0, rows, block_size):
                if row_start + block_size < rows:
                    row_read = block_size
                else:
                    row_read = rows - row_start

                current_block = in_band.ReadAsArray(col_start, row_start, col_read, row_read)
                
                if cd == ">":
                    reclass_arry = np.where(current_block > thre_val, 1, 0).astype(np.int16)
                elif cd =="<":
                    reclass_arry = np.where(current_block < thre_val, 1, 0).astype(np.int16)
                
                out_band.WriteArray(reclass_arry, col_start, row_start)
        
        out_ds.SetGeoTransform(in_ds_GeoTransform)
        out_ds.SetProjection(in_ds_Projection)  

    except:
        return "Failed to reclass image."

def filter_window(in_name, out_name, filter_size,cd):
    """
    in_name     - input image name with path
    out_name    - output image name with path
    filter_size - e.g., 7 stands for a 7x7 filter
    cd          - "max" or "min"
    """
    try:
        #1. Open the input image and get the GeoTransform, Projection, 
        #cols and rows.
        in_ds = gdal.Open(in_name)
        in_ds_GeoTransform = in_ds.GetGeoTransform()
        in_ds_Projection = in_ds.GetProjection()
        
        in_band = in_ds.GetRasterBand(1)
        cols = in_band.XSize
        rows = in_band.YSize
        
        #2.Create a output file. The gdal driver will actually create a rst file and a rdc file. 
        #The rdc file holds metadata.  The output metadata will set by using the input metadata, 
        # except the max/min value and display max/min value.
        rst_driver = gdal.GetDriverByName('rst')
        out_ds = rst_driver.Create(out_name, cols, rows,1,3) # here 1 means 1 band in the image,3 stands for integer datatype

        out_band = out_ds.GetRasterBand(1)

        #3.read 7 lines from the image each time for a 7x7 filter.  Create a current_row, and let numpy read three lines above it,
        #,three lines below it, and itself.  Althogh all the pixel will be processed in this 7 lines, only the centerline(4th line) 
        # will be written as output each time.
        #The scipy's ndimage.maximum_filter deals with the situation when there is no enough neibor for a 7x7 filter,
        #by using as much neighbor as it can.
        #We only need to deal with the first or last three lines in the image when they are the centerline, 
        # so they won't create three lines above or below then thus made the image out of index.
        
        if cd =="max":
            current_row  = 0
            while current_row <= rows-1:
                if current_row -3 < 0:
                    current_block = in_band.ReadAsArray(0, 0, cols, current_row + 4) #col_start, row_start, col read, row read
                    filtered_block = ndimage.maximum_filter(current_block, filter_size)
                    out_band.WriteArray(filtered_block[-4:-3], 0, current_row) #data, col_start, rows_start

                elif current_row -3 >=0 and current_row+3 <= rows-1:
                    current_block = in_band.ReadAsArray(0, current_row-3, cols, 7)
                    filtered_block = ndimage.maximum_filter(current_block, filter_size)
                    out_band.WriteArray(filtered_block[3:4], 0, current_row) #data, col_start, rows_start


                if current_row +3 > rows-1: 
                    current_block = in_band.ReadAsArray(0, current_row-3, cols, rows- 1 - current_row+4)              
                    filtered_block = ndimage.maximum_filter(current_block, filter_size)
                    out_band.WriteArray(filtered_block[3:4], 0, current_row) #data, col_start, rows_start

                current_row +=1

        elif cd =="min":
            current_row  = 0
            while current_row <= rows-1:
                if current_row -3 < 0:
                    current_block = in_band.ReadAsArray(0, 0, cols, current_row + 4) #col_start, row_start, col read, row read
                    filtered_block = ndimage.minimum_filter(current_block, filter_size)
                    out_band.WriteArray(filtered_block[-4:-3], 0, current_row) #data, col_start, rows_start

                elif current_row -3 >=0 and current_row+3 <= rows-1:
                    current_block = in_band.ReadAsArray(0, current_row-3, cols, 7)
                    filtered_block = ndimage.minimum_filter(current_block, filter_size)
                    out_band.WriteArray(filtered_block[3:4], 0, current_row) #data, col_start, rows_start


                if current_row +3 > rows-1: 
                    current_block = in_band.ReadAsArray(0, current_row-3, cols, rows- 1 - current_row+4)              
                    filtered_block = ndimage.minimum_filter(current_block, filter_size)
                    out_band.WriteArray(filtered_block[3:4], 0, current_row) #data, col_start, rows_start

                current_row +=1
        
        out_ds.SetGeoTransform(in_ds_GeoTransform)
        out_ds.SetProjection(in_ds_Projection)
    except:
        return "Failed to apply spatial filter to the input image."

def masks(in_image, mask_img, out_name):
    try:
        #1. open the input image
        in_ds = gdal.Open(in_image)
        in_band = in_ds.GetRasterBand(1)

        #2. Get metadata from input image
        in_ds_GeoTransform = in_ds.GetGeoTransform()
        in_ds_Projection = in_ds.GetProjection()
        cols = in_band.XSize
        rows = in_band.YSize

        #3.Create an empty rst file
        rst_driver = gdal.GetDriverByName('rst')
        out_ds = rst_driver.Create(out_name, cols, rows,1,3) # here 1 means 1 band in the image,3 means integer datatype
        out_band = out_ds.GetRasterBand(1)
        
        #get mask and read as array by block
        block_size = 640
        for col_start in range(0, cols, block_size): 
            if col_start + block_size < cols:
                col_read = block_size
            else:
                col_read = cols - col_start
            for row_start in range(0, rows, block_size):
                if row_start + block_size < rows:
                    row_read = block_size
                else:
                    row_read = rows - row_start
                in_array = in_band.ReadAsArray(col_start, row_start, col_read, row_read)
                mask_ds = gdal.Open(mask_img)
                mask_array = mask_ds.ReadAsArray(col_start, row_start, col_read, row_read)
                #apply the mask array on the input array
                #array_mask = np.ma.make_mask(mask_array)
                img_masked = np.where(mask_array == 1, in_array, 0)

                out_band.WriteArray(img_masked, col_start, row_start)
        
        out_ds.SetGeoTransform(in_ds_GeoTransform)
        out_ds.SetProjection(in_ds_Projection)     
    except:
        return "Failed to return masked image."

def cloud_shadow(cloud_name, shadow_name, out_name):
    try:
        cloud_ds = gdal.Open(cloud_name)
        cloud_band = cloud_ds.GetRasterBand(1)

        shadow_ds = gdal.Open(shadow_name)
        shadow_band = shadow_ds.GetRasterBand(1)
        cols = cloud_band.XSize
        rows = cloud_band.YSize

        #write output image and metadata
        in_ds_GeoTransform = cloud_ds.GetGeoTransform()
        in_ds_Projection = cloud_ds.GetProjection()

        rst_driver = gdal.GetDriverByName('rst')
        out_ds = rst_driver.Create(out_name, cols, rows,1,3) # here 1 means 1 band in the image,3 means integer datatype

        block_size = 640
        for col_start in range(0, cols, block_size): 
            if col_start + block_size < cols:
                col_read = block_size
            else:
                col_read = cols - col_start
            for row_start in range(0, rows, block_size):
                if row_start + block_size < rows:
                    row_read = block_size
                else:
                    row_read = rows - row_start

                cloud_array = cloud_ds.ReadAsArray(col_start, row_start, col_read, row_read)
                shadow_array = shadow_ds.ReadAsArray(col_start, row_start, col_read, row_read)

                cloud_and_shadow = cloud_array*2 + shadow_array

                
                out_band = out_ds.GetRasterBand(1)
                out_band.WriteArray(cloud_and_shadow, col_start, row_start)


        out_ds.SetGeoTransform(in_ds_GeoTransform)
        out_ds.SetProjection(in_ds_Projection)  

        del out_ds
           
    except:
        return "Failed to make cloud_shadow image."

#Examples: 



#1.Set the input band path and names; out put path

in_path = r"D:\Extracting_Clouds\cloud_shadow_test_rst\\"  #input image path

#define a file prefix.
in_name = "20180228_095050_102f_3B_AnalyticMS_SR_"
#The band images can be a group of files using the same prefix, e.g.,
#"20180228_095050_102f_3B_AnalyticMS_SR_b1.rst"
#"20180228_095050_102f_3B_AnalyticMS_SR_b2.rst"
#"20180228_095050_102f_3B_AnalyticMS_SR_b3.rst"
#"20180228_095050_102f_3B_AnalyticMS_SR_b4.rst"



os.makedirs(r"D:\Extracting_Clouds\0726\img\\") #make an output folder
out_path = r"D:\Extracting_Clouds\0726\img\\"

b1 = in_path + in_name + "b1.rst"
b2 = in_path + in_name + "b2.rst"
b3 = in_path + in_name + "b3.rst"
b4 = in_path + in_name + "b4.rst"




#2. Calculate the max and min image, and output to disk
maxmin_img("max", out_path + "max_img.rst",b1,b2,b3,b4)
maxmin_img("min", out_path + "min_img.rst",b1,b2,b3,b4)

filter_window(out_path + "max_img.rst", out_path + "max7x7.rst",7,"max")
filter_window(out_path + "min_img.rst", out_path +"min7x7.rst",7,"min")

#3. Calculate Shadow and Water, then to disk
reclass("<", out_path + "max7x7.rst", out_path + "Shadow_and_Water.rst", 2000) #specify threshold for shadow and water

#4. Cloud, and to disk
reclass(">", out_path + "min7x7.rst", out_path + "Cloud.rst", 2000) #specify threshold for cloud

#5. Land, and to disk
reclass(">", b4, out_path + "Land.rst", 1000) #specify threshold for land

#6. Shadow, and to disk
masks(out_path + "Shadow_and_Water.rst", out_path + "Land.rst", out_path + "Shadow.rst")

#7. put shadow and cloud into one image, and to disk

cloud_shadow(out_path + "Cloud.rst", out_path + "Shadow.rst", out_path + "Cloud_Shadow.rst")



