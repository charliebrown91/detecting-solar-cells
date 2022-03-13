# Name: ClipRaster_Fishnet.py
# Description: Cutting out raster dataset (Orthophoto) from Fishnet_Buildings
# Sources: https://pro.arcgis.com/de/pro-app/latest/arcpy/data-access/searchcursor-class.htm, https://pro.arcgis.com/de/pro-app/2.7/tool-reference/data-management/clip.htm

import arcpy
import os

fishnet_buildings = "C:/Users/Thomas/Documents/Privat/Masterarbeit/11_DataPreparation/Solaranlagen.gdb/fishnet_buildings" # path to geodatabase with fishnet_buildings
base_path = "D:/DOP2017_clipped_tif" # destination path
field = ["Shape@"] # taking coordinates
orthophoto = "D:/MosaicDataset/OF17HI10_DS_V1_20171215.gdb/OF17HI10_V1_RAS" # path to orthophoto mosaic
image_no = 1 # start numbering

with arcpy.da.SearchCursor(fishnet_buildings, field) as cursor:
    for row in cursor:
        path_to_tif = os.path.join(base_path, f"{Year}_{row[0].extent.XMax:.4f}_{row[0].extent.YMin:.4f}_{str(image_no)}.tif") # Choose year of flight "Year"
        arcpy.Clip_management(in_raster=orthophoto,
                              out_raster=path_to_tif,
                              in_template_dataset=row[0],
                              clipping_geometry="ClippingGeometry", # using fishnet_buildings
                              maintain_clipping_extent="NO_MAINTAIN_EXTENT")
        image_no = image_no + 1 # adding serial number
del cursor