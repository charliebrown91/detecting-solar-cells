# Name: CopyRaster.py
# Description: Changing tif to png raster dataset
# Source: https://pro.arcgis.com/de/pro-app/latest/tool-reference/data-management/copy-raster.htm

import arcpy
import os

rasterpath = "D:/Model/DOP2014_clipped_tif_20cm" # path to clipped orthophoto mosaic
outFolder = "D:/DataLabeling/DOP2014_clipped_png_20cm_scaled" # new destination path
arcpy.env.workspace = rasterpath

for rasterFile in arcpy.ListRasters("*.tif"):
    outRaster = os.path.join(outFolder,  rasterFile.replace(".tif", ".png")) # take off .tif and replace with .png
    arcpy.CopyRaster_management(in_raster=rasterFile,
                                out_rasterdataset=outRaster,
                                pixel_type="8_BIT_UNSIGNED",
                                scale_pixel_value="NONE",
                                format="PNG")
