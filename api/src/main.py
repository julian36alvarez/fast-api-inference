# -*- coding: utf-8 -*-
"""
This module contains the main FastAPI application to serve the model and process the images.
"""

# Standard library imports
import os
import uuid
from io import BytesIO
import zipfile
import json

# Third party imports
import sys

from PIL import Image, ImageEnhance, ImageStat
from fastapi import FastAPI, File, UploadFile
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.io import MemoryFile
from shapely.geometry import shape
from shapely.ops import unary_union
import boto3


import config.constants as const

sys.path.append(os.path.join(os.path.dirname(__file__), const.PARENT_DIRECTORY))
# pylint: disable=wrong-import-position
import packages.utils.get_path as path

app = FastAPI()
s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker-runtime')
endpoint_name = os.environ['ENDPOINT_NAME']
bucket_load = os.environ['BUCKET_LOAD']
bucket_output = os.environ['BUCKET_OUTPUT']


def create_s3_path( *args):
    return '/'.join(args)

MEAN_PATH = create_s3_path(const.DATA_DIR, const.PROCESSED_DIR, const.NORMALIZATION_DIR, const.MEAN_NPY)
STD_PATH = create_s3_path(const.DATA_DIR, const.PROCESSED_DIR, const.NORMALIZATION_DIR, const.STD_NPY)
AVG_BRIGTHNESS = create_s3_path(const.DATA_DIR, const.PROCESSED_DIR, const.NORMALIZATION_DIR, const.AVG_BRIGTHNESS)
AVG_CONTRAST = create_s3_path(const.DATA_DIR, const.PROCESSED_DIR, const.NORMALIZATION_DIR, const.AVG_CONTRAST)
SAVE_LABEL_PATH = create_s3_path(const.DATA_DIR, const.OUTPUT_DIR, const.LABELS_DIR)
SAVE_IMG_PATH = create_s3_path(const.DATA_DIR, const.OUTPUT_DIR)


def load_s3_object_as_numpy(s3, bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    bytestream = BytesIO(response['Body'].read())
    return np.load(bytestream)

def get_s3_image_url(bucket_name, image_path):
    """
    Generate an S3 URL for an image.

    Parameters:
    bucket_name (str): The name of the S3 bucket.
    image_path (str): The path to the image within the bucket.

    Returns:
    str: The S3 URL for the image.
    """
    return f"s3://{bucket_name}/{image_path}"

mean = load_s3_object_as_numpy(s3, bucket_load, MEAN_PATH)
std = load_s3_object_as_numpy(s3, bucket_load, STD_PATH)

def save_to_s3(s3, bucket, key, data):
    s3.put_object(Body=data, Bucket=bucket, Key=key)

# Definir la función para invocar el endpoint
def inference(data):
    serialized_data = json.dumps(data.tolist())
    response = sagemaker.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=serialized_data,
        ContentType='application/json',
    )
    inference = response['Body'].read()
    data = json.loads(inference)
    if isinstance(data, dict):
        data = np.array(list(data.values()))
    return data

def read_image(file_bytes):
    """
    Reads an image from a byte stream.
    This function receives a byte stream, reads it into an image,
    and returns the image, the transformation, and the CRS.
    Args:
        file_bytes: The byte stream to read.
    Returns:
        The image, the transformation, and the CRS.
    """
    with rasterio.open(BytesIO(file_bytes)) as src:
        original_transform = src.transform
        original_crs = src.crs
        img = src.read()
    return img, original_transform, original_crs

async def normalize_image(img):
    """
    Normalizes an image.
    This function receives an image, normalizes it, and returns the normalized image.
    Args:
        img: The image to normalize.
    Returns:
        The normalized image.
    """
    img = np.transpose(img, (1, 2, 0))
    # Normalize the image
    img = (img - mean) / std
    return img

async def make_prediction(img):
    """
    Makes a prediction using a trained model.
    This function receives an image, makes a prediction using a trained model,
    and returns the prediction.
    Args:
        img: The image to make a prediction on.
    Returns:
        The prediction.
    """
    # Make the prediction using the model
    pred = inference(np.expand_dims(img, axis=0))
    return pred

async def process_prediction(pred, filename_without_extension, path_name):
    """
    Processes a prediction.
    This function receives a prediction, processes it, and returns the processed prediction.
    Args:
        pred: The prediction to process.
        path_name: The name of the folder where the processed prediction will be saved.
    Returns:
        The processed prediction.
    """
    # Threshold the image 1 if > 0.89 else 0
    pred = (pred >= 0.89).astype(np.uint8)

    # Convert the prediction to an image
    pred = np.squeeze(pred) * 255
    pred = pred.astype(np.uint8)
    pred = Image.fromarray(pred)

    # Convert the image to RGBA
    pred = pred.convert("RGBA")

    # Create a new image to put the modified pixels into
    new_img = Image.new("RGBA", pred.size)

    # Go through all pixels and turn black (also shades of blacks)
    # (0, 0, 0, 255) into transparency (0, 0, 0, 0)
    for x in range(pred.width):
        for y in range(pred.height):
            r, g, b, a = pred.getpixel((x, y))
            # If it's black or close to black, make it transparent
            if r < 10 and g < 10 and b < 10:
                new_img.putpixel((x, y), (0, 0, 0, 0))
            else:
                new_img.putpixel((x, y), (255, 0, 0, a))  # red color
    #new_img.save(f"{path_name}/{filename_without_extension}{const.IMAGE_PNG_EXTENTION}")
    img_byte_arr = BytesIO()
    new_img.save(img_byte_arr, format=const.IMAGE_PNG_EXTENTION.lstrip('.'))
    img_byte_arr = img_byte_arr.getvalue()
    # Guardar la imagen en S3
    key = f"{path_name}/{filename_without_extension}{const.IMAGE_PNG_EXTENTION}"
    save_to_s3(s3, bucket_output, key, img_byte_arr)

def read_image_as_array(filename):
    """
    Read an image file and return its data as a numpy array.

    Parameters:
    filename (str): The name of the file to read.

    Returns:
    numpy.ndarray: The image data.
    """
    s3_filename = get_s3_image_url(bucket_output, filename)
    with rasterio.open(s3_filename) as src:
        img_array = src.read()
    return img_array

def write_image(
    filename_without_extension, img_array, original_transform, original_crs
):
    """
    Write an image array to a TIFF file with the specified parameters.

    Parameters:
    filename_without_extension (str): The base name of the file to write to.
    img_array (numpy.ndarray): The image data to write.
    original_transform (affine.Affine): The transformation to apply to the image.
    original_crs (dict or str): The coordinate reference system of the image.

    Returns:
    None
    """

    with rasterio.Env():
        with MemoryFile() as memfile:
            with rasterio.open(
                memfile,
                "w",
                driver="GTiff",
                height=img_array.shape[1],
                width=img_array.shape[2],
                count=img_array.shape[0],
                dtype=str(img_array.dtype),
                crs=original_crs,
                transform=original_transform,
            ) as dst:
                dst.write(img_array)
            
            print("Image written to memory file")
            memfile.seek(0)
            save_to_s3(s3, bucket_output, filename_without_extension, memfile.read())
            
def image_to_shapefile(filename):
    """
    Converts a georeferenced image to a shapefile.

    This function opens a georeferenced image, reads it into an array,
    and converts the array into a set of polygons.
    It then converts the polygons into a single multipolygon and saves it as a shapefile.
    Finally, it creates a zip file containing the shapefile.

    Args:
        filename: The path to the georeferenced image file.

    Returns:
        None. The function saves the shapefile to disk and does not return anything.
    """
    # Open the georeferenced image
    with rasterio.open(filename) as src:
        # Read the image
        img_array = src.read(1)

        # Get the transform and CRS
        transform = src.transform
        crs = src.crs

        # Convert the image array into a set of polygons
        # Only consider red pixels (255, 0, 0, 255)
        polygons = list(shapes(img_array, mask=(img_array == 255), transform=transform))

        # Convert the polygons into a single multipolygon
        multipolygon = unary_union([shape(polygon[0]) for polygon in polygons])

        # Create a GeoDataFrame with the multipolygon
        gdf = gpd.GeoDataFrame({"geometry": [multipolygon]})

        # Set the CRS of the GeoDataFrame
        gdf.crs = crs

        # Save the GeoDataFrame as a shapefile
        shapefile_name = filename.replace(".tif", ".shp")
        gdf.to_file(shapefile_name)

        # Create a ZipFile object
        with zipfile.ZipFile(shapefile_name.replace(".shp", ".zip"), "w") as zipf:
            # Add the shapefile components to the zip file
            for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                file_name = shapefile_name.replace(".shp", ext)
                zipf.write(file_name)
                os.remove(file_name)
        # Save the zip file to S3
        with open(shapefile_name + ".zip", "rb") as data:
            save_to_s3(s3, bucket_output, shapefile_name + ".zip", data.read())
        os.remove(shapefile_name + ".zip")

def image_to_geojson(filename):
    """
    Converts a georeferenced image to a GeoJSON file.

    This function opens a georeferenced image, reads it into an array,
    and converts the array into a set of polygons.
    It then converts the polygons into a single multipolygon and saves it as a GeoJSON file.

    Args:
        filename: The path to the georeferenced image file.

    Returns:
        GeoJSON: The GeoJSON representation of the image as a multipolygon json object.
    """
    # Open the georeferenced image
    s3_filename = get_s3_image_url(bucket_output, filename)
    with rasterio.open(s3_filename) as src:
        # Read the image
        img_array = src.read(1)

        # Get the transform and CRS
        transform = src.transform
        crs = src.crs

        # Convert the image array into a set of polygons
        # Only consider red pixels (255, 0, 0, 255)
        polygons = list(shapes(img_array, mask=(img_array == 255), transform=transform))

        # Convert the polygons into a single multipolygon
        multipolygon = unary_union([shape(polygon[0]) for polygon in polygons])

        # Create a GeoDataFrame with the multipolygon
        gdf = gpd.GeoDataFrame({"geometry": [multipolygon]})

        # Set the CRS of the GeoDataFrame
        gdf.crs = crs
        geojson_str = gdf.to_json()

        # Save the GeoDataFrame as a GeoJSON file
        geojson_name = filename.replace(".tif", ".geojson")
        geojson_dir = os.path.dirname(geojson_name)
        os.makedirs(geojson_dir, exist_ok=True) 
        gdf.to_file(geojson_name, driver="GeoJSON")
        # Save the GeoJSON file to S3
        with open(geojson_name, "rb") as data:
            save_to_s3(s3, bucket_output, geojson_name, data.read())
        os.remove(geojson_name)

        return geojson_str

def adjust_brightness_contrast(image):
    """
    Adjusts the brightness and contrast of an image.
    Args:
        image: The image to adjust.
        brightness: The brightness adjustment. A value of 0.0 means no adjustment.
        contrast: The contrast adjustment. A value of 1.0 means no adjustment.
    Returns:
        The adjusted image.
    """
    contrast = load_s3_object_as_numpy(s3, bucket_load, AVG_CONTRAST)
    brightness = load_s3_object_as_numpy(s3, bucket_load, AVG_BRIGTHNESS)
    adjusted_image = np.clip((image - np.average(image)) * (contrast / np.std(image)) + brightness, 0, 255)
    return adjusted_image.astype(np.uint8)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predicts the class of an uploaded image using a trained model.
    This function receives an image file, checks if it's a TIFF file,
    and reads it into a numpy array. It then checks the shape of the image,
    normalizes it, and uses a trained model to make a prediction.
    The prediction is thresholded, converted back to an image,
    and saved to disk. Finally, the function calls `image_to_shapefile`
    to convert the image to a shapefile.

    Args:
        file: The uploaded image file.

    Returns:
        A dictionary with the filename and a success message.
    """
    # function body
    # Load the image from the request
    file_bytes = await file.read()

    # Get the filename
    filename_without_extension, file_extension = os.path.splitext(file.filename)

    if file_extension != const.TIF_EXTENTION:
        return {const.ERROR_MESSAGE.format(const.ERROR_FORMAT)}

    img, original_transform, original_crs = read_image(file_bytes)

    # Check the image
    if img.shape[0] != 3 or img.shape[1] != 256 or img.shape[2] != 256:
        return {"error": "La imagen debe ser de 3 canales (RGB) y de 256 x 256"}
    folder_name = filename_without_extension[-36:]
    full_path = os.path.join(SAVE_LABEL_PATH, folder_name)
    #os.makedirs(full_path, exist_ok=True)
    img = await normalize_image(img)
    pred = await make_prediction(img)
    await process_prediction(pred, filename_without_extension, full_path)

    save_img_file = f"{full_path}/{filename_without_extension}"
    img_array = read_image_as_array(f"{save_img_file}{const.IMAGE_PNG_EXTENTION}")
    
    tiff_file = f"{full_path}/{filename_without_extension}{file_extension}"
    
    write_image(
        tiff_file,
        img_array,
        original_transform,
        original_crs,
    )
    #image_to_shapefile(f"{save_img_file}{file_extension}")
    geojson = json.loads(image_to_geojson(tiff_file))
    path_result = get_s3_image_url(bucket_output, tiff_file)
    return {"message": "Success", "filename": f"{path_result}", "geojsonData": geojson}

@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

@app.get("/")
def health2():
    """
    Health check endpoint.
    """
    return {"status": "ok"}
