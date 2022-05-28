from curses import ERR
import cv2
import io
import json
from importlib_metadata import metadata
import numpy as np
import os
import requests
import sqlite3
import tensorflow as tf
import threading
from datetime import datetime
from flask import Flask, request, Response, jsonify, make_response
from PIL import Image as im
from requests import RequestException, ConnectionError

from deepface.commons import functions
from deepface.detectors import FaceDetector

tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])

os.environ["CUDA_VISIBLE_DEVICES"]="0"
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
    )
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


HOSTNAME = '0.0.0.0'
PORT = 6337
FEATURE_EXTRACTION_URL = 'http://127.0.0.1:5000/'
DIR_PATH = "dataset/"
SAMPLE_FACE_VECTOR_DATABASE = []
FEATURE_EXTRACT_BATCH_SIZE = 10

IMAGE_PROCESS_OK = 100
IMAGE_PROCESS_ERR = 101
EXTRACT_SAMPLE_VECTOR_OK = 200
EXTRACT_SAMPLE_VECTOR_ERR = 201
UPDATE_SAMPLE_FACES_OK = 202
UPDATE_SAMPLE_FACES_ERR = 203
FEATURE_EXTRACTION_SERVER_CONNECTION_ERR = 204
FEATURE_EXTRACTION_REQUEST_ERR = 205
FEATURE_EXTRACTION_SERVER_RESPONSE_OK = 206
FEATURE_EXTRACTION_SERVER_RESPONSE_PARSE_ERR = 207
FACE_DETECTION_OK = 210
FACE_DETECTION_ERR = 211
NO_FACE_DETECTED_ERR = 212
CALC_DISTANCE_OK = 220
CALC_DISTANCE_ERR = 221
NO_SAMPLE_VECTOR_ERR = 222
NO_SUCH_FILE_ERR = 230
INVALID_REQUEST_ERR = 231
INVALID_IMAGE_ERR = 232
UNKNOWN_ERR = 500

ERR_MESSAGES = {
    IMAGE_PROCESS_OK: 'The image is processed successfully.',
    IMAGE_PROCESS_ERR: 'The image process has been failed.',
    UPDATE_SAMPLE_FACES_OK: 'Sample vector database has been updated successfully.',
    UPDATE_SAMPLE_FACES_ERR: 'Failed to update the sample vector database.',
    FEATURE_EXTRACTION_SERVER_CONNECTION_ERR: 'Feature extraction node is not running.',
    FEATURE_EXTRACTION_REQUEST_ERR: 'Bad request to feature extraction node.',
    FEATURE_EXTRACTION_SERVER_RESPONSE_OK: 'Successfully received a response from feature extraction node.',
    FEATURE_EXTRACTION_SERVER_RESPONSE_PARSE_ERR: 'Failed to parse a response from feature extraction node.',
    FACE_DETECTION_OK: 'Faces are successfully detected from the input image.',
    FACE_DETECTION_ERR: 'Failed to detect face from the input image',
    NO_FACE_DETECTED_ERR: 'No face detected from the input image.',
    CALC_DISTANCE_OK: 'Calculation of vector distance has been suceeded.',
    CALC_DISTANCE_ERR: 'Failed to calculate the vector distance.',
    NO_SAMPLE_VECTOR_ERR: 'There is no sample face data.',
    NO_SUCH_FILE_ERR: 'No such file.',
    INVALID_REQUEST_ERR: 'Invalid request.',
    INVALID_IMAGE_ERR: 'Invalid image has input. Could not read the image data.',
    UNKNOWN_ERR: 'Unknown error has occurred.'
}

app = Flask(__name__)


def get_db_connection():
    """
    Get sqlite connection
    """
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


def rescale_image(img, dsize_width=600):
    """
    Rescale the image to reduce the image size
    """
    original_width = img.shape[0]
    original_height = img.shape[1]

    original_ratio = original_height / original_width

    dsize = (int(dsize_width * original_ratio), dsize_width)

    rescaled_img = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)

    return rescaled_img, dsize_width / original_width


def recover_face_region(detected_region, scaled_ratio):
    """
    Since I rescaled the original image to reduce the face detection time, I will recover the detected region.
    """
    face_region = []
    for i in range(4):
        face_region.append(
            str(int(np.int32(detected_region[i]).item() / scaled_ratio))
        )
    
    return face_region


def detect_faces(img, detector_backend = 'retinaface', align = False):
    """
    Detect the faces from the base image
    """
    #detector stored in a global variable in FaceDetector object.
    #this call should be completed very fast because it will return found in memory
    #it will not build face detector model in each call (consider for loops)
    face_detector = FaceDetector.build_model(detector_backend)

    if isinstance(img, str):
        if len(img) == 0:
            return None, None
        elif len(img) > 11 and img[0:11] == "data:image/":
            img = functions.load_image(img)
        else:
            img = functions.load_image(DIR_PATH + img)

    elif isinstance(img, bytes):
        img = np.array(im.open(io.BytesIO(img)))

    else:
        return None, None

    if not isinstance(img, np.ndarray):
        return None, None

    # rescale the image to the smaller size
    img, scaled_ratio = rescale_image(img)

    # faces stores list of detected_face and region pair
    faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align)

    # no face is detected
    if len(faces) == 0:
        return None, None

    return faces, scaled_ratio


def call_feature_extractor(face_list):
    """
    Send request to feature extraction node. Request will contain list of face ids and detected face image
    Returns error code, and result string
    """
    success_feature_vectors = []
    failure_feature_vectors = []

    try:
        face_data = []
        image_files = []
        for f in face_list:
            face_data.append({
                'id': f['id']
            })

            # Convert the numpy array to bytes
            face_pil_img = im.fromarray(f['img'])
            byte_io = io.BytesIO()
            face_pil_img.save(byte_io, 'png')
            byte_io.seek(0)

            image_files.append((
                'image', byte_io
            ))

        # Send request to feature extraction node
        response = requests.post(FEATURE_EXTRACTION_URL, data={'face_data': json.dumps(face_data)}, files=image_files)

        # Parse the response and get the feature vectors
        try:
            feature_list = json.loads(response.text)

            # Determine which one is success, which one is failure
            for fe in feature_list:
                if len(fe['vector']) == 0:  # If feature extraction is failed
                    failure_feature_vectors.append({
                        'id': fe['id']
                    })
                else:   # If feature extraction is suceed
                    success_feature_vectors.append({
                        'id': fe['id'],
                        'vector': np.array(fe['vector'])
                    })

        except:
            return FEATURE_EXTRACTION_SERVER_RESPONSE_PARSE_ERR, None, None

    except ConnectionError:
        return FEATURE_EXTRACTION_SERVER_CONNECTION_ERR, None, None

    except RequestException:
        return FEATURE_EXTRACTION_REQUEST_ERR, None, None

    return FEATURE_EXTRACTION_SERVER_RESPONSE_OK, success_feature_vectors, failure_feature_vectors


def feature_extraction_thread(face_list, extract_success_list, extract_failure_list):
    """
    Call feature extraction module. this function will be run in multi-threads
    """
    # Prepare the MetaData's Map, this will be useful to determine which faces are success to extract features, and failure
    metadata_map = {}
    for f in face_list:
        metadata_map[f['id']] = f

    # Call API of feature extraction server
    res_code, success_face_features, failure_face_features = call_feature_extractor(face_list)
            
    if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
        # Add all faces to failed list
        for f in face_list:
            extract_failure_list.append({
                'id': f['id']
            })
        return

    # Treat the success faces, add meta data
    for face in success_face_features:
        # If could not find meta data of this face, move it to failed list
        if face['id'] not in metadata_map:
            failure_face_features.append(face)
            continue

        # Add meta data
        meta_data = metadata_map[face['id']]
        face['name'] = meta_data['name']
        face['metadata'] = meta_data['metadata']
        face['action'] = meta_data['action']

    # Append to result arrays
    extract_success_list += success_face_features
    extract_failure_list += failure_face_features


def extract_sample_feature_vector(data_list):
    """
    Extract the feature vector from the sample images
    Return code, extract_success_list, extract_failure_list
    """
    face_list = []
    extract_success_list = []
    extract_failure_list = []
    thread_pool = []

    # Main loop, each element will contain one image and its metadata
    for data in data_list:
        try:
            sample_id = data['id']
            name = data['name']
            img = data['image']
            metadata = data['metadata']
            action = data['action']
        except:
            return INVALID_REQUEST_ERR, None, None

        # Detect face from sample image
        detected_faces, scaled_ratio = detect_faces(img)

        # No face detected
        if detected_faces is None:
            continue

        # Get the first face from the detected faces list. Suppose that the sample image has only 1 face
        face = detected_faces[0]    # tuple (np.ndarray, list) - np.adarray is image. list is face region, e.x. [x, y, w, h]

        # # Get face region from the base image(profile image)
        # face_region = recover_face_region(face[1], scaled_ratio)

        face_list.append({
            'id': sample_id,
            'img': face[0],
            'name': name,
            'metadata': metadata,
            'action': action
        })

        if len(face_list) == FEATURE_EXTRACT_BATCH_SIZE:
            th = threading.Thread(target=feature_extraction_thread, args=(face_list, extract_success_list, extract_failure_list))
            th.start()
            thread_pool.append(th)

            face_list = []

    if len(face_list) > 0:
        th = threading.Thread(target=feature_extraction_thread, args=(face_list, extract_success_list, extract_failure_list))
        th.start()
        thread_pool.append(th)

    # Wait until all threads are finished
    for th in thread_pool:
        th.join()

    return EXTRACT_SAMPLE_VECTOR_OK, extract_success_list, extract_failure_list


def save_sample_database(sample_vectors, db_type='sqlite'):
    """
    Save the sample face feature vector into the database
    """
    if db_type == 'sqlite':
        # Prepare db connection
        conn = get_db_connection()

        # Run query
        for vector_data in sample_vectors:
            sample_id = vector_data['id']
            name = vector_data['name']
            metadata = vector_data['metadata']
            action = vector_data['action']
            vector = vector_data['vector']

            # Delete original vector
            sql_query = f'DELETE FROM sample_face_vectors WHERE sample_id = "{sample_id}";'
            conn.execute(sql_query)

            # Save new vector
            sql_query = f'INSERT INTO sample_face_vectors (sample_id, name, metadata, action, vector) ' \
                f'VALUES ("{sample_id}", "{name}", "{metadata}", "{action}", "{json.dumps(vector.tolist())}");'
            conn.execute(sql_query)

        conn.commit()
        conn.close()

    else:   # Use global variable
        global SAMPLE_FACE_VECTOR_DATABASE

        SAMPLE_FACE_VECTOR_DATABASE = []

        for vector_data in sample_vectors:
            SAMPLE_FACE_VECTOR_DATABASE.append({
                'id': vector_data['id'],
                'name': vector_data['name'],
                'metadata': vector_data['metadata'],
                'action': vector_data['action'],
                'feature_vector': vector_data['vector']
            })
    
    return UPDATE_SAMPLE_FACES_OK


def get_sample_database(db_type='sqlite'):
    """
    Read sample feature vector from database
    """
    if db_type == 'sqlite':
        sample_vectors = []

        conn = get_db_connection()
        sample_vector_list = conn.execute('SELECT * FROM sample_face_vectors').fetchall()

        for vector_data in sample_vector_list:
            vector = np.array(json.loads(vector_data['vector']))
            sample_vectors.append({
                'id': vector_data['sample_id'],
                'name': vector_data['name'],
                'metadata': vector_data['metadata'],
                'action': vector_data['action'],
                'vector': vector
            })

        if conn:
            conn.close()
            
        return sample_vectors

    else:   # Use global variable
        global SAMPLE_FACE_VECTOR_DATABASE

        return SAMPLE_FACE_VECTOR_DATABASE


def find_face(face_feature_vectors, min_distance):
    """
    Find the closest sample by comparing the feature vectors
    """
    # Read sample database
    sample_vectors = get_sample_database()

    if len(sample_vectors) == 0:
        return NO_SAMPLE_VECTOR_ERR, None

    candidates = []
    for vector_data in face_feature_vectors:
        face_feature_vector = vector_data['vector']

        # Initialize variables
        closest_id = ''
        closest_name = ''
        closest_metadata = ''
        closest_distance = -1
        
        # Compare with sample vectors
        for i in range(len(sample_vectors)):
            sample = sample_vectors[i]
            sample_vector = sample['vector']

            try:
                # Calculate the distance between sample and the detected face.
                distance_vector = np.square(face_feature_vector - sample_vector)
                distance = np.sqrt(distance_vector.sum())

                if (closest_id == '' or abs(distance) < abs(closest_distance)) and abs(distance) < min_distance:
                    closest_distance = abs(distance)
                    closest_id = sample['id']
                    closest_name = sample['name']
                    closest_metadata = sample['metadata']

            except Exception as e:
                print(e)
                pass
        
        # If not find fit sample, skip
        if closest_id == '':
            continue

        # Add candidate for this face
        candidates.append({
            'id': closest_id,
            'name': closest_name,
            'metadata': closest_metadata,
            'bbox': vector_data['bbox']
        })
        
    return CALC_DISTANCE_OK, candidates


def process_image(img, min_distance):
    """
    Face recognition
    """
    # Detect the faces from the image that is dedicated in the path or bytes
    try:
        faces, scaled_ratio = detect_faces(img)
    except:
        return FACE_DETECTION_ERR, None

    if len(faces) == 0:
        return NO_FACE_DETECTED_ERR, None

    bound_box_map = {}
    face_list = []
    face_feature_vector_list = []

    # Send request to feature_extraction module
    for i in range(len(faces)):
        face = faces[i]     # tuple (np.ndarray, list) - np.adarray is image. list is face region, e.x. [x, y, w, h]

        # Need to cast the int32 to str
        face_region = recover_face_region(face[1], scaled_ratio)

        # Prepare bound box map
        bound_box_map[i] = ', '.join(face_region)

        # Make the face list, I will send bunch of faces to Feature Extraction Server at once
        face_list.append({
            'id': i,
            'img': face[0]
        })

        if len(face_list) == FEATURE_EXTRACT_BATCH_SIZE:
            # Call the api to extract the feature from the detected faces
            res_code, success_face_features, failure_face_features = call_feature_extractor(face_list)

            if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
                return res_code, None

            face_feature_vector_list += success_face_features

            face_list = []

    if len(face_list) > 0:
        # Call the api to extract the feature from the detected faces
        res_code, success_face_features, failure_face_features = call_feature_extractor(face_list)

        if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
            return res_code, None

        face_feature_vector_list += success_face_features
    
    # Add bound box for each face feature vector
    vector_list = []
    for f in face_feature_vector_list:
        if int(f['id']) not in bound_box_map:
            continue
        
        f['bbox'] = bound_box_map[int(f['id'])]
        vector_list.append(f)

    # Find candidates by comparing feature vectors between detected face and samples
    status, candidates = find_face(vector_list, min_distance)

    if status != CALC_DISTANCE_OK:
        return status, None

    return IMAGE_PROCESS_OK, candidates


def update_sample_database(data_list):
    """
    Update the database that contains sample face vectors
    """
    start_time = datetime.now()

    # Extract the feature vector
    res, success_sample_vectors, failure_sample_vectors = extract_sample_feature_vector(data_list)

    if res != EXTRACT_SAMPLE_VECTOR_OK:
        return res, success_sample_vectors, failure_sample_vectors

    # Save the sample feature vector into database
    res = save_sample_database(success_sample_vectors)

    print(f"{(datetime.now() - start_time).total_seconds() * 1000}")

    return res, success_sample_vectors, failure_sample_vectors


@app.route('/update-samples', methods=['GET', 'POST'])
def update_samples():
    # GET request
    if request.method == 'GET':
        return Response('Face detection server is running.', status=200)

    # POST request
    data_list = request.json

    ## Try to extract features from samples, and update database
    res_code, success_list, failure_list  = update_sample_database(data_list)
    if res_code != UPDATE_SAMPLE_FACES_OK:
        response = {
            'error': ERR_MESSAGES[res_code]
        }
        return make_response(jsonify(response), 400)

    ## Make response
    response = {
        'success': [f['id'] for f in success_list],
        'fail': [f['id'] for f in failure_list]
    }
    return make_response(jsonify(response), 200)


@app.route('/face-recognition', methods=['GET', 'POST'])
def face_recognition():
    if request.method == 'GET':
        return Response('Face detection server is running.', status=200)

    # POST
    # Read image data
    img_data = request.json
    if 'image' not in img_data:
        response = {
            'error': ERR_MESSAGES[INVALID_REQUEST_ERR]
        }
        return make_response(jsonify(response), 400)

    # min_distance is optional parameter in request
    min_distance = 9  # default threshold for facenet, 0.4 for vgg-face model
    if 'min_distance' in img_data:
        min_distance = float(img_data['min_distance'])

    # Process image
    res_code, candidates = process_image(img_data['image'], min_distance)

    if res_code != IMAGE_PROCESS_OK:
        response = {
            'error': ERR_MESSAGES[res_code]
        }
        return make_response(jsonify(response), 500)

    # Return candidates
    return make_response(jsonify(candidates), 200)


if __name__ == '__main__':
    # Run app in debug mode on port 6337
    app.run(debug=True, host='0.0.0.0', port=6337, threaded=True)