# Name: CreateFishnet.py
# Description: Creates rectangular cells
# Source: https://pro.arcgis.com/de/pro-app/latest/tool-reference/data-management/create-fishnet.htm

# import system module
import arcpy
from arcpy import env

env.workspace = "C:/data/" # workspace environment - where save data?
outFeatureClass = "fishnet_raw.shp" # new file name

# Set coordinate system of the output fishnet
env.outputCoordinateSystem = arcpy.SpatialReference(2056)

# the origin of the fishnet
originCoordinate = '2621775.01960435 1178005.59295473' # bottom left of orthophoto-mosaic
# orientation
yAxisCoordinate = '2621775.01960435 1178015.59295473' # bottom left of orthophoto-mosaic

# width and height
cellSizeWidth = '102.4'
cellSizeHeight = '102.4'

# Enter 0 for numRows / numColumns - these values will be calculated by the tool
numRows = ""
numColumns = ""

oppositeCorner = '2685770.80896241 1238726.74396845' # upper right of orthophoto-mosaic

# Create a point label feature class
labels = "NO_LABELS"

# Extent is set by origin and opposite corner - no need to use a template fc
templateExtent = "#"

# Each output cell will be a polygon
geometryType = 'POLYGON'

arcpy.CreateFishnet_management(outFeatureClass, originCoordinate, yAxisCoordinate, cellSizeWidth, cellSizeHeight, numRows, numColumns, oppositeCorner, labels, templateExtent, geometryType)
