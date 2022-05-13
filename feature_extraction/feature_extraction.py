import cv2
import io
import json
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from flask import Flask, request, Response
from PIL import Image
from urllib.parse import urlparse, parse_qs

from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
from deepface.detectors import FaceDetector


tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
	import keras
	from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow import keras
    from tensorflow.keras.preprocessing import image

os.environ["CUDA_VISIBLE_DEVICES"]="0"
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = None
input_shape = None
face_detector = None

#detector stored in a global variable in FaceDetector object.
#this call should be completed very fast because it will return found in memory
#it will not build face detector model in each call (consider for loops)
detector_backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

app = Flask(__name__)


@app.before_first_request
def load_model(model_name='Facenet'):
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


def get_face_feature(face_img, face_id, face_features):
    """
    Get the feature vector from the face image
    1. Resize and normalizing
    2. Extract the feature vector
    This method will be run in parallel by threads
    """
    global model, input_shape

    start_time = datetime.now()

    # Process the face, get the representation of the detected face
    try:
        # Resize and normalizing
        face_img_pixels = preprocess_face(face_img, input_shape)
        
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

    print(f"{(datetime.now() - start_time).total_seconds() * 1000}")


@app.route('/', methods=['GET'])
def welcome():
    """
    Welcome page
    """
    return "<h1 style='color:blue'>Feature extraction server is running!</h1>"


@app.route('/', methods=['POST'])
def process_detected_faces():
    """
    All requests from the face detection node will be arrived at here
    """
    # Parse request
    body_data = request.form.getlist('face_id_list')
    face_id_list = json.loads(body_data[0])
    base_img_list = request.files.getlist('image')

    face_features = []
    for i in range(len(face_id_list)):
        face_id = face_id_list[i]
        base_img = base_img_list[i].read()  # this is bytes data of detected face image

        # Convert the bytes to numpy array
        face_img = np.asarray(Image.open(io.BytesIO(base_img)))

        get_face_feature(face_img, face_id, face_features)
	
    return Response(json.dumps(face_features), status=200)


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
