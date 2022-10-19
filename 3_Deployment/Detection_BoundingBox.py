### Installing packages
import os
import sys
import pytz
import datetime
import json
import requests
import pandas as pd
import numpy as np
import time
import tifffile
import tensorflow as tf
from PIL import Image, ImageDraw
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings("ignore")


### Clone github
!git clone https://github.com/matterport/Mask_RCNN.git

%cd Mask_RCNN
!python setup.py install

# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = 'D:/GanzerKt/Deployment/Mask_RCNN/mrcnn/'

# Import mrcnn libraries
from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.visualize
import mrcnn.model as modellib


### Methods
def bbox_shape(dataframe, shapefile_name):
  df= dataframe.rename(columns={0:"Bild",
                         1:"Bounding Box",
                         2:"Maske",
                         3:"Klasse",
                         4:"Score"})

  df = df.drop(columns=["Maske"]) # remove column "Bild"

  df = df[df["Bounding Box"].map(lambda d: len(d)) >0] # remove images without solar panel installations

  df = df.explode(column=["Bounding Box", "Klasse", "Score"]) # divide column data

  df["Bild"] = df["Bild"].str[30:70] # remove image path
  df["Flugjahr"] = df["Bild"].str[0:4] # get flight year of image name
  df["E_Start"] = df["Bild"].str[5:17].astype("float") # get coordinates
  df["N_Start"] = df["Bild"].str[18:30].astype("float") # get coordinates

  df["E_Start_new"] = round(df["E_Start"] - 512*0.2, 4) # get left upper - start point
  df["N_Start_new"] = round(df["N_Start"] + 512*0.2, 4) # get left upper - start point

  df["Nummer"] = df["Bild"].str[31:37] # get image number
  df["Nummer"] = df["Nummer"].map(lambda x: x.rstrip(".tif")) # remove unwanted letters, if necessary

  df = df.drop(columns=["Bild"]) # remove column "Bild"
  df = df.reindex(columns=["Nummer", "Flugjahr", "E_Start", "N_Start", "Klasse", "Bounding Box", "Score", "E_Start_new", "N_Start_new"]) # sort columns

  df["Klasse"] = df["Klasse"].replace(1, "Photovoltaik") # rename values
  df["Klasse"] = df["Klasse"].replace(2, "Solarthermie") # rename values

  df = df.reset_index(drop=True) # reset index

  # define and transform bounding box elements from relative coordinate system to absolute coordinate system
  y_min = []
  x_min = []
  y_max = []
  x_max = []
  for starters_E, starters_N, element in zip(df["E_Start_new"], df["N_Start_new"], df["Bounding Box"]):
    y_minn = round(starters_N - element[0] * 0.2, 4)
    x_minn = round(starters_E + element[1] * 0.2, 4)
    y_maxx = round(starters_N - element[2] * 0.2, 4)
    x_maxx = round(starters_E + element[3] * 0.2, 4)
    y_min.append(y_minn)
    x_min.append(x_minn)
    y_max.append(y_maxx)
    x_max.append(x_maxx)
  df["y_min"] = y_min
  df["x_min"] = x_min
  df["y_max"] = y_max
  df["x_max"] = x_max

  # Collecting all bounding boxes in one list
  polygon_geometries = []
  for index, row in df.iterrows():
    lon_point_list = [row.x_min, row.x_max, row.x_max, row.x_min]
    lat_point_list = [row.y_min, row.y_min, row.y_max, row.y_max]

    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    polygon_geometries.append(polygon_geom)

  # remove some columns
  df = df.drop("y_min", axis=1)
  df = df.drop("x_min", axis=1)
  df = df.drop("y_max", axis=1)
  df = df.drop("x_max", axis=1)
  df = df.drop("E_Start_new", axis=1)
  df = df.drop("N_Start_new", axis=1)

  # define schema
  df["Nummer"] = df["Nummer"].astype("int")
  df["Flugjahr"] = df["Flugjahr"].astype("int")
  df["E_Start"] = df["E_Start"].astype("float")
  df["N_Start"] = df["N_Start"].astype("float")
  df["Klasse"] = df["Klasse"].astype("str")
  df["Bounding Box"] = df["Bounding Box"].astype("str")
  df["Score"] = df["Score"].astype("float")

  # Creating ESRI Shapefile
  df["geometry"] = gpd.GeoDataFrame(crs='epsg:2056', geometry=polygon_geometries)
  gdf = gpd.GeoDataFrame(df,
                        geometry=df["geometry"],
                        crs='epsg:2056')
  gdf.to_file(filename="D:/5_Extraktion_Solar-Geometrien/Bounding_Box/%s.shp" % shapefile_name, driver="ESRI Shapefile")


### Loading model
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join("D:/4_Modell/mask_rcnn_solar_dataset_0072.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


### Hyperparameter Mask-RCNN
class SolarConfig(Config): # histogram matching3
    """Configuration for training on the box_synthetic dataset.
    Derives from the base Config class and overrides specific values.
    """
    # Give the configuration a recognizable name
    NAME = "solar_dataset"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


    # All of our training images are 512x512
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    NUM_CLASSES = 3 # background + pv + thermal
    IMAGE_CHANNEL_COUNT = 3 # changed to tiff
    MEAN_PIXEL = np.array([123.1, 126.2, 107.5]) # calculated by "Loading.tiff.ipynb"

    STEPS_PER_EPOCH = 1090

    DETECTION_MIN_CONFIDENCE = 0.90 # all proposals with less than 0.90 confidence will be ignored
    DETECTION_NMS_THRESHOLD = 0.3

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 545

    BACKBONE = 'resnet50'  # Backbone network architecture.

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256) ## IMPORTANT!!!!!!!!!!!!!!!
    WEIGHT_DECAY = 0.001 ## IMPORTANT!!!!!!!!!!!!!!!
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 78 # maximum number of ground truth instances per one image => max from 3_EDA_after_Labeling
    DETECTION_MAX_INSTANCES = 78 # max number of final detections per one image => max number from 3_EDA_after_Labeling
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000
config = SolarConfig()


### Define the dataset
class SolarDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation_json, images_dir):
        """ Load solars dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()

        # Add the class names using base method from utils.Dataset
        source_name = "solars"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']['counts']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids


### Prepare to run Inference
class InferenceConfig(SolarConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9 # all proposals with less than 0.9 confidence will be ignored; DEFAULT = 0.9
    IMAGE_SHAPE = [512, 512, 3]
    NUM_CLASSES = 3
inference_config = InferenceConfig()

# Recreate the model in inference mode
model_inference = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)


### ToDo: Define path to trained weights
# Get path to saved weights
model_path = "D:/4_Modell/mask_rcnn_solar_dataset_0072.h5"

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model_inference.load_weights(model_path, by_name=True)


### Save Output - Statistics
pd.options.display.float_format = '{:.4f}'.format


### ToDO: Define path to images
# Image-No 0-4999
start_apply = time.time()
real_test_dir = 'D:/3_Kacheln_prozessiert/2014/' # path to images
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.tif', '.jpg', '.png']:
        image_paths.append(os.path.join(real_test_dir, filename))
image_name = []
bboxes = []
maskes = []
classes = []
scoring = []
for image_path in image_paths[0:5000]:
    img = tifffile.imread(image_path)
    img_arr = np.array(img)[:,:,:3]
    results = model_inference.detect([img_arr], verbose=1)
    r = results[0]
    image_name.append(image_path)
    bboxes.append(r['rois'])
    maskes.append(r['masks'].astype('int'))
    classes.append(r['class_ids'])
    scoring.append(r['scores'])
    print(image_path)
lists = [image_name, bboxes, maskes, classes, scoring]
df = pd.concat([pd.Series(x) for x in lists], axis=1)
bbox_shape(df, "Bounding_Box_2014_5000")
end_apply = time.time()
minutes = round((end_apply - start_apply) / 60, 2)
print(f'Detection took {minutes} minutes')

# Image-No 5000-9999
start_apply = time.time()
real_test_dir = 'D:/3_Kacheln_prozessiert/2014/' # path to images
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.tif', '.jpg', '.png']:
        image_paths.append(os.path.join(real_test_dir, filename))
image_name = []
bboxes = []
maskes = []
classes = []
scoring = []
for image_path in image_paths[5000:10000]:
    img = tifffile.imread(image_path)
    img_arr = np.array(img)[:,:,:3]
    results = model_inference.detect([img_arr], verbose=1)
    r = results[0]
    image_name.append(image_path)
    bboxes.append(r['rois'])
    maskes.append(r['masks'].astype('int'))
    classes.append(r['class_ids'])
    scoring.append(r['scores'])
    print(image_path)
lists = [image_name, bboxes, maskes, classes, scoring]
df = pd.concat([pd.Series(x) for x in lists], axis=1)
bbox_shape(df, "Bounding_Box_2014_10000")
end_apply = time.time()
minutes = round((end_apply - start_apply) / 60, 2)
print(f'Detection took {minutes} minutes')

# Image-No 10000-14999
start_apply = time.time()
real_test_dir = 'D:/3_Kacheln_prozessiert/2014/' # path to images
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.tif', '.jpg', '.png']:
        image_paths.append(os.path.join(real_test_dir, filename))
image_name = []
bboxes = []
maskes = []
classes = []
scoring = []
for image_path in image_paths[10000:15000]:
    img = tifffile.imread(image_path)
    img_arr = np.array(img)[:,:,:3]
    results = model_inference.detect([img_arr], verbose=1)
    r = results[0]
    image_name.append(image_path)
    bboxes.append(r['rois'])
    maskes.append(r['masks'].astype('int'))
    classes.append(r['class_ids'])
    scoring.append(r['scores'])
    print(image_path)
lists = [image_name, bboxes, maskes, classes, scoring]
df = pd.concat([pd.Series(x) for x in lists], axis=1)
bbox_shape(df, "Bounding_Box_2014_15000")
end_apply = time.time()
minutes = round((end_apply - start_apply) / 60, 2)
print(f'Detection took {minutes} minutes')

# Image-No 15000-19999
start_apply = time.time()
real_test_dir = 'D:/3_Kacheln_prozessiert/2014/' # path to images
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.tif', '.jpg', '.png']:
        image_paths.append(os.path.join(real_test_dir, filename))
image_name = []
bboxes = []
maskes = []
classes = []
scoring = []
for image_path in image_paths[15000:20000]:
    img = tifffile.imread(image_path)
    img_arr = np.array(img)[:,:,:3]
    results = model_inference.detect([img_arr], verbose=1)
    r = results[0]
    image_name.append(image_path)
    bboxes.append(r['rois'])
    maskes.append(r['masks'].astype('int'))
    classes.append(r['class_ids'])
    scoring.append(r['scores'])
    print(image_path)
lists = [image_name, bboxes, maskes, classes, scoring]
df = pd.concat([pd.Series(x) for x in lists], axis=1)
bbox_shape(df, "Bounding_Box_2014_20000")
end_apply = time.time()
minutes = round((end_apply - start_apply) / 60, 2)
print(f'Detection took {minutes} minutes')

# Image-No: 20000-24999
start_apply = time.time()
real_test_dir = 'D:/3_Kacheln_prozessiert/2014/' # path to images
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.tif', '.jpg', '.png']:
        image_paths.append(os.path.join(real_test_dir, filename))
image_name = []
bboxes = []
maskes = []
classes = []
scoring = []
for image_path in image_paths[20000:25000]:
    img = tifffile.imread(image_path)
    img_arr = np.array(img)[:,:,:3]
    results = model_inference.detect([img_arr], verbose=1)
    r = results[0]
    image_name.append(image_path)
    bboxes.append(r['rois'])
    maskes.append(r['masks'].astype('int'))
    classes.append(r['class_ids'])
    scoring.append(r['scores'])
    print(image_path)
lists = [image_name, bboxes, maskes, classes, scoring]
df = pd.concat([pd.Series(x) for x in lists], axis=1)
bbox_shape(df, "Bounding_Box_2014_25000")
end_apply = time.time()
minutes = round((end_apply - start_apply) / 60, 2)
print(f'Detection took {minutes} minutes')

# Image-No: 25000-29999
start_apply = time.time()
real_test_dir = 'D:/3_Kacheln_prozessiert/2014/' # path to images
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.tif', '.jpg', '.png']:
        image_paths.append(os.path.join(real_test_dir, filename))
image_name = []
bboxes = []
maskes = []
classes = []
scoring = []
for image_path in image_paths[25000:30000]:
    img = tifffile.imread(image_path)
    img_arr = np.array(img)[:,:,:3]
    results = model_inference.detect([img_arr], verbose=1)
    r = results[0]
    image_name.append(image_path)
    bboxes.append(r['rois'])
    maskes.append(r['masks'].astype('int'))
    classes.append(r['class_ids'])
    scoring.append(r['scores'])
    print(image_path)
lists = [image_name, bboxes, maskes, classes, scoring]
df = pd.concat([pd.Series(x) for x in lists], axis=1)
bbox_shape(df, "Bounding_Box_2014_30000")
end_apply = time.time()
minutes = round((end_apply - start_apply) / 60, 2)
print(f'Detection took {minutes} minutes')

# Image-No: 30000-34999
start_apply = time.time()
real_test_dir = 'D:/3_Kacheln_prozessiert/2014/' # path to images
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.tif', '.jpg', '.png']:
        image_paths.append(os.path.join(real_test_dir, filename))
image_name = []
bboxes = []
maskes = []
classes = []
scoring = []
for image_path in image_paths[30000:35000]:
    img = tifffile.imread(image_path)
    img_arr = np.array(img)[:,:,:3]
    results = model_inference.detect([img_arr], verbose=1)
    r = results[0]
    image_name.append(image_path)
    bboxes.append(r['rois'])
    maskes.append(r['masks'].astype('int'))
    classes.append(r['class_ids'])
    scoring.append(r['scores'])
    print(image_path)
lists = [image_name, bboxes, maskes, classes, scoring]
df = pd.concat([pd.Series(x) for x in lists], axis=1)
bbox_shape(df, "Bounding_Box_2014_35000")
end_apply = time.time()
minutes = round((end_apply - start_apply) / 60, 2)
print(f'Detection took {minutes} minutes')

# Image-No: 35000-39999
start_apply = time.time()
real_test_dir = 'D:/3_Kacheln_prozessiert/2014/' # path to images
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.tif', '.jpg', '.png']:
        image_paths.append(os.path.join(real_test_dir, filename))
image_name = []
bboxes = []
maskes = []
classes = []
scoring = []
for image_path in image_paths[35000:40000]:
    img = tifffile.imread(image_path)
    img_arr = np.array(img)[:,:,:3]
    results = model_inference.detect([img_arr], verbose=1)
    r = results[0]
    image_name.append(image_path)
    bboxes.append(r['rois'])
    maskes.append(r['masks'].astype('int'))
    classes.append(r['class_ids'])
    scoring.append(r['scores'])
    print(image_path)
lists = [image_name, bboxes, maskes, classes, scoring]
df = pd.concat([pd.Series(x) for x in lists], axis=1)
bbox_shape(df, "Bounding_Box_2014_40000")
end_apply = time.time()
minutes = round((end_apply - start_apply) / 60, 2)
print(f'Detection took {minutes} minutes')