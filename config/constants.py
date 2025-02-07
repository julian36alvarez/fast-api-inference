# -*- coding: utf-8 -*-
"""
This module defines constants used throughout the project.

The constants include directory names, image properties, and data split ratios.
"""

DATA_DIR = "data"
RAW_DIR = "0_raw"
IMAGES_DIR = "images"
LABELS_DIR = "labels"
IMAGE_SIZE = 256
IMAGE_CHANNELS = 3
IMAGE_RBG = "RGB"
IMAGE_GRAY = "L"
IMAGE_JPG_EXTENTION = ".jpg"
IMAGE_PNG_EXTENTION = ".png"
NORMALIZATION_DIR = "0_normalization"
INTERMEDIATE_DIR = "1_intermediate"
EXTERNAL_DIR = "0_external"
GOOGLE_EARTH_DIR = "google_earth"
MODEL_DIR = "models"
PROCESSED_DIR = "2_processed"
TRAIN_DIR = "1_train"
VALIDATION_DIR = "2_validation"
EVALUATION_DIR = "3_evaluation"
IMAGE_LABEL_END = "_label"
TRAIN_SIZE = 0.6  # Split the data into training (60%) and test (40%) sets
TEST_SIZE = 0.4  # Split the data into training (60%) and test (40%) sets
EVALUATION_SIZE = 0.5  # Split the data into training (60%) and test (40%)
LABEL_TITLE = "Label"
IMAGE_TITLE = "Image RBG"
IMAGE_NAME = "Image"
WIDTH_NAME = "Width"
HEIGHT_NAME = "Height"
PIXEL_NAME = "Pixels"
AUGMENTATION_DIR = "augmentation"
CMAP_GRAY = "gray"
RANDOM_STATE = 42
TRAIN_RGB_NPY = "train_rgb.npy"
TRAIN_LABEL_NPY = "train_label.npy"
VAL_RGB_NPY = "validation_rgb.npy"
VAL_LABEL_NPY = "validation_label.npy"
EVAL_RGB_NPY = "evaluation_rgb.npy"
EVAL_LABEL_NPY = "evaluation_label.npy"
MEAN_NPY = "mean.npy"
STD_NPY = "std.npy"
MIN_NPY = "min.npy"
MAX_NPY = "max.npy"
REPORT_DIR = "reports"
FILE_CKPT = "CP3.ckpt"
BEST_MODEL_PATH = "4.28032024-UNET.h5"
PARENT_DIRECTORY = ".."
IMAGE_ROTATE_180 = 180
OUTPUT_DIR = "3_output"
AVG_BRIGTHNESS = "avg_brightness.npy"
AVG_CONTRAST = "avg_contrast.npy"
NORMALIZATION_DIR_IMAGES = "normalized"
TIF_EXTENTION = ".tif"
ERROR_MESSAGE = "Error: {}"
ERROR_FORMAT = "El archivo debe ser un archivo TIFF (.tif)"
BASE_URL = "http://127.0.0.1:8000"
SPLIT_ENDPOINT = "/split"
PREDICT_ENDPOINT = "/predict"
MERGED_DIR = "merged"