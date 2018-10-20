import numpy as np
import tensorflow as tf


class ObjectDetector(object):

    def __init__(self, model_path, labels_path):
        self.labels = self._load_labels(labels_path)
        self.detection_graph = self._build_graph(model_path)
        self.session = tf.Session(graph=self.detection_graph)

    def _build_graph(self, filepath):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(filepath, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_labels(self, filepath):
        with open(filepath) as f:
            txt = f.read()
        lines = txt.split('\n')
        ids = []
        display_names = []
        for l in lines:
            if l.find(' id:') > 0:
                ids.append(int(l[6:]))
            if l.find(' display_name:') > 0:
                display_names.append(l[17:-1])
        if len(ids) == len(display_names):
            labels = dict(zip(ids, display_names))
        return labels

    def _pil_to_numpy(self, image_pil):
        width, height = image_pil.size
        np_temp = np.array(image_pil.getdata())
        image_np = np_temp.reshape((height, width, 3)).astype(np.uint8)
        return image_np

    def _make_note(self, im_width, im_height, bbox, text):
        y_min, x_min, y_max, x_max = bbox
        box_top = int(y_min * im_height)
        box_left = int(x_min * im_width)
        box_width = int((x_max - x_min) * im_width)
        box_height = int((y_max - y_min) * im_height)
        note = {'top': box_top,
                'left': box_left,
                'width': box_width,
                'height': box_height,
                'text': text,
                'id': 'e7f44ac5-bcf2-412d-b111-6dbb8b19ffbe'}
        return note

    def detect_and_get_notes(self, image_pil, min_score=0.5, max_results=10):
        im_width, im_height = image_pil.size
        boxes, scores, classes = self.detect(image_pil)
        notes = []
        for i in range(max_results):
            if scores[i] > min_score:
                label = self.labels[classes[i]]
                score = scores[i]
                text = '{}. {} ({:.0%})'.format(i, label, score)
                box = boxes[i]
                note = self._make_note(im_width, im_height, box, text)
                notes.append(note)
        return notes

    def detect(self, image_pil):
        image_np = self._pil_to_numpy(image_pil)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        graph = self.detection_graph
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        boxes = graph.get_tensor_by_name('detection_boxes:0')
        scores = graph.get_tensor_by_name('detection_scores:0')
        classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.session.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        boxes, scores, classes, num_detections = map(
            np.squeeze, [boxes, scores, classes, num_detections])

        return boxes, scores, classes.astype(int)
