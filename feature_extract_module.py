import cv2
import io
import json
import numpy as np
import random
import threading
from datetime import datetime
from flask import Flask, request, Response
from PIL import Image
from urllib.parse import urlparse, parse_qs

from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
from deepface.commons import functions
from deepface.detectors import FaceDetector
from deepface.detectors import OpenCvWrapper
from scipy import rand

import tensorflow as tf
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])

physical_devices = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tf_major_version == 1:
	import keras
	from keras.preprocessing.image import load_img, save_img, img_to_array
	from keras.applications.imagenet_utils import preprocess_input
	from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    from tensorflow.keras.preprocessing import image

model = None
input_shape = None
face_detector = None

#detector stored in a global variable in FaceDetector object.
#this call should be completed very fast because it will return found in memory
#it will not build face detector model in each call (consider for loops)
detector_backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']


app = Flask(__name__)


def load_model(model_name='VGGFace'):
    """
    Load Model. Use VGGFace as default
    """
    # soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # resource.setrlimit(resource.RLIMIT_AS, (max_memory_size, hard))

    global model, input_shape, face_detector

    model_list = {
        'VGGFace': VGGFace.loadModel,
        'OpenFace': OpenFace.loadModel,
        'Facenet': Facenet.loadModel,
        'FbDeepFace': FbDeepFace.loadModel
    }

    if model_name in model_list:
        model = model_list[model_name]()
        
    else:
        model = model_list['VGGFace']()

    input_shape = model.layers[0].input_shape[0][1:3]

    face_detector = FaceDetector.build_model(detector_backends[0])


def parse_face_region(face_region):
    """
    Since the http request could not contain numpy int32-format variable, face_detector_module converts them to string to send http request
    Here, we will convert them into int-format variable again
    """
    return {
        'x': int(face_region[0]),
        'y': int(face_region[1]),
        'w': int(face_region[2]),
        'h': int(face_region[3])
    }


def get_detected_face(base_img_bytes, face_region):
    """
    Since the request from face_detector module contains the full image and the face region, so we will process the following tasks:
    1. Convert full image from bytes to numpy array
    2. According to the face_region, we will get the face image(numpy arr) from full image
    Return numpy array - detected face
    """
    global face_detector
    try:
        # convert bytes data to PIL Image object
        base_img = Image.open(io.BytesIO(base_img_bytes))

        # PIL image object to numpy array
        base_img_arr = np.asarray(base_img)

        # Crop the detected face by using face_region and full image
        region_scale = parse_face_region(face_region)
        face_img = base_img_arr[int(region_scale['y']):int(region_scale['y'] + region_scale['h']), int(region_scale['x']):int(region_scale['x'] + region_scale['w'])]

        # Align face image
        face_img_arr = OpenCvWrapper.align_face(face_detector["eye_detector"], face_img)

        return True, face_img_arr

    except Exception as e:
        print(f'Crop Exception: {e}')
        return False, str(e)


def preprocess_face(face_img, target_size=(224, 224), grayscale = False):
    """
    Preprocess the face image before the feature extraction
    1. Resize the face_img with the target size
    2. Normalizing the image pixels
    """
    # Resize image to expected shape
    if face_img.shape[0] > 0 and face_img.shape[1] > 0:
        factor_0 = target_size[0] / face_img.shape[0]
        factor_1 = target_size[1] / face_img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(face_img.shape[1] * factor), int(face_img.shape[0] * factor))
        face_img = cv2.resize(face_img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - face_img.shape[0]
        diff_1 = target_size[1] - face_img.shape[1]
        if grayscale == False:
            # Put the base image in the middle of the padded image
            face_img = np.pad(face_img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
        else:
            face_img = np.pad(face_img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

    # Double check: if target image is not still the same size with target
    if face_img.shape[0:2] != target_size:
        face_img = cv2.resize(face_img, target_size)
    
    # Normalizing the image pixels
    face_img_pixels = image.img_to_array(face_img) # What this line doing? must?
    face_img_pixels = np.expand_dims(face_img_pixels, axis = 0)
    face_img_pixels /= 255 # Normalize input in [0, 1]

    return face_img_pixels


def get_face_feature(base_img_bytes, face_region, face_features):
    """
    Get the feature vector from the face region on the base image
    1. According to the rectangle data, crop the detected face from the base image
    2. Resize and normalizing
    3. Extract the feature vector
    This method will be run in parallel by threads
    """
    global model, input_shape

    face_id = face_region['id']
    face_rect = face_region['rectangle']

    # Get detected face(numpy array)
    status, face_img_arr = get_detected_face(base_img_bytes, face_rect)

    if not status:
        face_features.append({
            'id': face_id,
            'feature_vector': None
        })
        return

    # Process the face, get the representation of the detected face
    try:
        # Resize and normalizing
        face_img_pixels = preprocess_face(face_img_arr, input_shape)

        # Extract the feature vector
        face_img_representation = model.predict(face_img_pixels)[0,:]

        vector_str = face_img_representation.tolist()

        face_features.append({
            'id': face_id,
            'vector': vector_str
        })
        
    except Exception as e:
        face_features.append({
            'id': face_id,
            'vector': None
        })


@app.route('/', methods=['POST'])
def process_detected_faces():
    """
    All requests from the face detection node will be arrived at here
    """
    # Parse request
    body_data = request.form.getlist('face_regions')
    face_regions = json.loads(body_data[0])
    base_img_list = request.files.getlist('image')

    face_features = []
    thread_pool = []

    for i in range(len(face_regions)):
        face = face_regions[i]
        base_img = base_img_list[i].read()

        start_time = datetime.now()
        get_face_feature(base_img, face, face_features)
        print(f"{face['rectangle'][2]}, {face['rectangle'][3]}, {(datetime.now() - start_time).total_seconds() * 1000}")

    #     th = threading.Thread(target=get_face_feature, args=(base_img_bytes, face, face_features))
    #     thread_pool.append(th)
    #     th.start()

    # for th in thread_pool:
    #     th.join()
	
    return Response(json.dumps(face_features), status=200)


if __name__ == '__main__':
    # load_model('VGGFace')
    load_model('Facenet')
    # load_model('OpenFace')
    # load_model('FbDeepFace')

    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
