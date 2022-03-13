# Name: Fishnet_Buildings.py
# Description: Extract tiles containing buildings
# Sources: https://pro.arcgis.com/de/pro-app/latest/tool-reference/data-management/select-layer-by-attribute.htm, https://pro.arcgis.com/de/pro-app/latest/tool-reference/data-management/select-layer-by-location.htm, 

# import system module
import arcpy
from arcpy import env

# workspace environment - where is data?
arcpy.env.workspace = "C:/data"
outFeatureClass = "fishnet_buildings.shp" # new file name

# filter to buildings
buildings = arcpy.SelectLayerByAttribute_management(in_layer_or_view="AV_Bodenbedeckung.shp",
                                                    selection_type="NEW_SELECTION",
                                                    where_clause='"ART" = 0 And "SHAPE_AREA" > 6')
# make a new layer
fishnet_buildings = arcpy.SelectLayerByLocation_management(in_layer="fishnet_raw.shp",
                                                           overlap_type="INTERSECT",
                                                           select_features=buildings)

arcpy.CopyFeatures_management(fishnet_buildings, outFeatureClass)

