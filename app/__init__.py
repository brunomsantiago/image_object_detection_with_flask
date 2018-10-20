import os
from flask import Flask
from config import Config
from app.tf_object_detector import ObjectDetector

app = Flask(__name__)
app.config.from_object(Config)

models_directory = r'C:\Users\BRUNO\nn_models\tensorflow'
model_subdirectory = r'ssd_mobilenet_v1_coco_2018_01_28'
model_filename = 'frozen_inference_graph.pb'
model_filepath = os.path.join(models_directory,
                              model_subdirectory,
                              model_filename)

labels_directory = r'C:\Users\BRUNO\nn_models\tensorflow'
labels_filename = r'mscoco_label_map.pbtxt'
labels_filepath = os.path.join(labels_directory, labels_filename)

detector = ObjectDetector(model_filepath, labels_filepath)

from app import routes
