# Simple Obect Detection Inference with Flask and Tensorflow

This project is a study on how to deploy deep learning object detection models.
It is a very simple and unpolished flask website which the user can upload a file and see the boxes of the detected objects. Hovering the mouse over the boxes shows the label and score.


## 1. Demonstration

![Demonstration](https://github.com/brunomsantiago/simple_object_detection_with_flask/raw/master/demo/flask_detection_demo.gif "Simple Obect Detection Inference with Flask and Tensorflow")

## 2. Installation

#### 2.1. Clone the repository

#### 2.2. Install dependecies
- flask
- flask-wtf
- pillow
- tensorflow (cpu version is fine)
- werkzeug

#### 2.3. Build or download a tensorflow object detection model and labels
For the demo I used:
- **Model**: [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- **Labels**: [mscoco_label_map.pbtxt
](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt)

#### 2.4. Configure model on __init__.py
Adjust the model_filepath and labels_filepath variables

```python
models_directory = r'C:\Users\BRUNO\nn_models\tensorflow'
model_subdirectory = r'ssd_mobilenet_v1_coco_2018_01_28'
model_filename = 'frozen_inference_graph.pb'
model_filepath = os.path.join(models_directory,
                              model_subdirectory,
                              model_filename)

labels_directory = r'C:\Users\BRUNO\nn_models\tensorflow'
labels_filename = r'mscoco_label_map.pbtxt'
labels_filepath = os.path.join(labels_directory, labels_filename)
```

#### 2.5 Confirgure flask enviroment variables
- On windows type on prompt: `set FLASK=inference.py`
- On linux type on shell: `export FLASK=inference.py`

#### Go to project direcory root and star flask
- Type on prompt/shell: `flask run`
