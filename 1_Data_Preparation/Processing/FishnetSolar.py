# Name: FishnetSolar.py
# Description: Extract tiles containing buildings
# Sources: https://pro.arcgis.com/de/pro-app/latest/tool-reference/data-management/select-layer-by-attribute.htm, https://pro.arcgis.com/de/pro-app/latest/tool-reference/analysis/spatial-join.htm

# import system module
import arcpy
from arcpy import env

arcpy.env.workspace = "C:/Users/Thomas/Downloads/Shapes" # workspace environment - where is data?

# filter to buildings
buildings = arcpy.SelectLayerByAttribute_management(in_layer_or_view="AV_Bodenbedeckung.shp",
                                                    selection_type="NEW_SELECTION",
                                                    where_clause='"ART" = 0 And "SHAPE_AREA" > 6')

# join buildings with gt_YEAR (solar data); KEY: EGID
solar_buildings = arcpy.SpatialJoin_analysis(target_features=buildings,
                                             join_features="gt_2014.shp",
                                             out_feature_class=r"memory/solar_buildings2014",
                                             join_operation="JOIN_ONE_TO_MANY",
                                             join_type="KEEP_COMMON")

# delete several columns in solar_buildings
arcpy.DeleteField_management(in_table=solar_buildings,
                             drop_field=["Join_Count", "TARGET_FID", "JOIN_FID", "EGID"])



# filter to buildings containing solar panels
fishnet_solar2014 = arcpy.SpatialJoin_analysis(target_features="fishnet_raw.shp",
                                               join_features=r"memory/solar_buildings2014",
                                               out_feature_class=r"memory/fishnet_solar",
                                               join_operation="JOIN_ONE_TO_MANY",
                                               join_type="KEEP_ALL",
                                               match_option="INTERSECT")

# filter to buildings
fishnet_solar = arcpy.SelectLayerByAttribute_management(in_layer_or_view=fishnet_solar2014,
                                                        selection_type="NEW_SELECTION",
                                                        where_clause='"JOIN_FID" <> -1')

# delete several columns in fishnet_solar
arcpy.DeleteField_management(in_table=fishnet_solar,
                             drop_field=["Join_Count", "Shape_Length", "Shape_Area"])

outfeatureclass = "fishnet_solar2014.shp"
arcpy.CopyFeatures_management(fishnet_solar, outfeatureclass)








