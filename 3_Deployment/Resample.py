# Name: Resample.py
# Description: Changing spatial resolution for DOP2017 and DOP2020 to 0.2 m (downsampling)
# Source: https://pro.arcgis.com/de/pro-app/latest/tool-reference/data-management/resample.htm

import arcpy
import os


arcpy.env.workspace = "D:/DOP2017_clipped_tif" # path to clipped orthophoto mosaic
base_path = "D:/DOP2017_clipped_tif_20cm" # new destination path

for inRaster in arcpy.ListRasters("*", "TIF"):
    out_raster = os.path.join(base_path, inRaster.replace(".tif", "_{0}cm.tif".format(20)))
    arcpy.Resample_management(in_raster=inRaster,
                              out_raster=out_raster,
                              cell_size="0.2", # in meters
                              resampling_type="CUBIC") # bicubic