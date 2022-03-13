# Name: ClipRaster.py
# Description: Cutting out raster dataset (Orthophoto) from solar_buildings
# Sources: https://pro.arcgis.com/de/pro-app/latest/arcpy/data-access/searchcursor-class.htm, https://pro.arcgis.com/de/pro-app/2.7/tool-reference/data-management/clip.htm

import arcpy
import os


fishnet_solar = "C:/data/fishnet_solar2014/fishnet_solar2014.shp" # path to shapefile with fishnet_solarYEAR 
base_path = "D:/Model/DOP2014_clipped_tif_20cm" # destination path
field = ["Shape@", "OID@"] # taking coordinates and OID
orthophoto = "D:/MosaicDataset/OF14HIGH_DS_V1_20140609.gdb/OF14HIGH_V1_RAS"  # path to orthophoto mosaic

with arcpy.da.SearchCursor(fishnet_solar, field) as cursor:
    for row in cursor:
        path_to_tif = os.path.join(base_path, f"{2014}_{row[0].extent.XMax:.4f}_{row[0].extent.YMin:.4f}_FID_{row[1]}.tif")
        arcpy.Clip_management(in_raster=orthophoto,
                              out_raster=path_to_tif,
                              in_template_dataset=row[0],
                              clipping_geometry="ClippingGeometry", # using solar_buildings
                              maintain_clipping_extent="NO_MAINTAIN_EXTENT")

del cursor