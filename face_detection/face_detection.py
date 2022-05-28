import cv2
import io
import json
import numpy as np
import requests
import sqlite3
import threading
from datetime import datetime
from flask import Flask, request, Response
from PIL import Image as im
from requests import RequestException, ConnectionError

from deepface.commons import functions
from deepface.detectors import FaceDetector

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
    Rescaled the original image to reduce the face detection time. This will not affect the detection too much.
    So I will rescale the detected region.
    """
    face_region = []
    for i in range(4):
        face_region.append(
            int(np.int32(detected_region[i]).item() / scaled_ratio)
        )
    
    return face_region


def detect_faces(img, detector_backend = 'opencv', align = False):
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
    try:
        face_id_list = []
        image_files = []
        for face_data in face_list:
            face_id_list.append(face_data['id'])

            # Convert the numpy array to bytes
            face_pil_img = im.fromarray(face_data['img'])
            byte_io = io.BytesIO()
            face_pil_img.save(byte_io, 'png')
            byte_io.seek(0)

            image_files.append((
                'image', byte_io
            ))

        # Send request to feature extraction node
        response = requests.post(FEATURE_EXTRACTION_URL, data={'face_id_list': json.dumps(face_id_list)}, files=image_files)

        # Parse the response and get the feature vectors
        feature_vector_data = []
        try:
            feature_list = json.loads(response.text)
            for face in feature_list:
                feature_vector_data.append({
                    'id': face['id'],
                    'vector': np.array(face['vector'])
                })
        except:
            return FEATURE_EXTRACTION_SERVER_RESPONSE_PARSE_ERR, None

    except ConnectionError:
        return FEATURE_EXTRACTION_SERVER_CONNECTION_ERR, None

    except RequestException:
        return FEATURE_EXTRACTION_REQUEST_ERR, None

    return FEATURE_EXTRACTION_SERVER_RESPONSE_OK, feature_vector_data


def feature_extraction_thread(face_list, face_feature_vector_list):
    """
    Call feature extraction module. this function will be run in multi-threads
    """
    res_code, face_feature_data = call_feature_extractor(face_list)
            
    if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
        return res_code, None

    for i in range(len(face_feature_data)):
        face_feature_data[i]['sample_name'] = face_list[i]['sample_name']

    face_feature_vector_list += face_feature_data


def extract_sample_feature_vector(data_list):
    """
    Extract the feature vector from the sample images
    """
    global SAMPLE_FACE_VECTOR_DATABASE

    face_list = []
    face_feature_vector_list = []

    thread_pool = []
    for data in data_list:
        try:
            img = data['image']
            metadata = data['metadata']
        except:
            return INVALID_REQUEST_ERR, None

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
            'id': 0,
            'img': face[0],
            'sample_name': metadata
        })

        if len(face_list) == FEATURE_EXTRACT_BATCH_SIZE:
            th = threading.Thread(target=feature_extraction_thread, args=(face_list, face_feature_vector_list))
            th.start()
            thread_pool.append(th)

            face_list = []

    if len(face_list) > 0:
        th = threading.Thread(target=feature_extraction_thread, args=(face_list, face_feature_vector_list))
        th.start()
        thread_pool.append(th)

    # Wait until all threads are finished
    for th in thread_pool:
        th.join()

    return EXTRACT_SAMPLE_VECTOR_OK, face_feature_vector_list


def save_sample_database(sample_vectors, db_type='sqlite'):
    """
    Save the sample face feature vector into the database
    """
    if db_type == 'sqlite':
        # Prepare db connection
        conn = get_db_connection()

        # Run query
        for vector_data in sample_vectors:
            name = vector_data['sample_name']
            vector = vector_data['vector']

            # Delete original vector
            sql_query = f'DELETE FROM sample_face_vectors WHERE name = "{name}";'
            conn.execute(sql_query)

            # Save new vector
            sql_query = f'INSERT INTO sample_face_vectors (name, vector) VALUES ("{name}", "{json.dumps(vector.tolist())}");'
            conn.execute(sql_query)

        conn.commit()
        conn.close()

    else:   # Use global variable
        global SAMPLE_FACE_VECTOR_DATABASE

        SAMPLE_FACE_VECTOR_DATABASE = []

        for vector_data in sample_vectors:
            SAMPLE_FACE_VECTOR_DATABASE.append({
                'id': vector_data['sample_name'],
                'feature_vector': vector_data['vector']
            })
    
    return UPDATE_SAMPLE_FACES_OK, len(sample_vectors)


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
                'name': vector_data['name'],
                'vector': vector
            })

        if conn:
            conn.close()
            
        return sample_vectors

    else:   # Use global variable
        global SAMPLE_FACE_VECTOR_DATABASE

        return SAMPLE_FACE_VECTOR_DATABASE


def find_face(face_feature_vectors):
    """
    Find the closest sample by comparing the feature vectors
    """
    # Read sample database
    sample_vectors = get_sample_database()

    if len(sample_vectors) == 0:
        return NO_SAMPLE_VECTOR_ERR, None

    candidates = []
    for vector_data in face_feature_vectors:
        image_id = vector_data['id']
        face_feature_vector = vector_data['vector']

        # Initialize variables
        closest_sample_name = ''
        closest_distance = -1
        
        # Compare with sample vectors
        for i in range(len(sample_vectors)):
            sample = sample_vectors[i]
            sample_vector = sample['vector']

            try:
                # Calculate the distance between sample and the detected face.
                distance_vector = np.square(face_feature_vector - sample_vector)
                distance = np.sqrt(distance_vector.sum())

                if closest_sample_name == '' or abs(distance) < abs(closest_distance):
                    closest_distance = abs(distance)
                    closest_sample_name = sample['name']

            except Exception as e:
                print(e)
                pass
        
        # Add candidate for this face
        candidates.append({
            'image_id': image_id,
            'sample_id': closest_sample_name,
            'distance': closest_distance
        })
        
    return CALC_DISTANCE_OK, candidates


def process_image(img):
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

    # Send request to feature_extraction module
    face_list = []
    face_feature_vector_list = []
    for i in range(len(faces)):
        face = faces[i]     # tuple (np.ndarray, list) - np.adarray is image. list is face region, e.x. [x, y, w, h]

        # # Need to cast the int32 to int, because int32 is not allowed to be included in http request
        # face_region = recover_face_region(face[1], scaled_ratio)

        face_list.append({
            'id': i,
            'img': face[0]
        })

        if len(face_list) == FEATURE_EXTRACT_BATCH_SIZE:
            # Call the api to extract the feature from the detected faces
            res_code, face_feature_data = call_feature_extractor(face_list)

            if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
                return res_code, None

            face_feature_vector_list += face_feature_data

            face_list = []

    if len(face_list) > 0:
        # Call the api to extract the feature from the detected faces
        res_code,  face_feature_data = call_feature_extractor(face_list)

        if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
            return res_code, None

        face_feature_vector_list += face_feature_data

    # Find candidates by comparing feature vectors between detected face and samples
    status, candidates = find_face(face_feature_vector_list)

    if status != CALC_DISTANCE_OK:
        return status, None

    return IMAGE_PROCESS_OK, candidates


def update_sample_database(data_list):
    """
    Update the database that contains sample face vectors
    """
    start_time = datetime.now()
    # Extract the feature vector
    res, sample_vectors = extract_sample_feature_vector(data_list)

    if res != EXTRACT_SAMPLE_VECTOR_OK:
        return ERR_MESSAGES[res]

    # Save the sample feature vector into database
    res, sample_count = save_sample_database(sample_vectors)

    print(f"{(datetime.now() - start_time).total_seconds() * 1000}")

    if res == UPDATE_SAMPLE_FACES_OK:
        return f'There are {sample_count} sample faces.'

    return ERR_MESSAGES[res]


@app.route('/update-samples', methods=['GET', 'POST'])
def update_samples():
    if request.method == 'GET':
        return Response('Face detection server is running.', status=200)

    # POST
    data_list = request.json
    response_text = update_sample_database(data_list)
    return Response(response_text, status=200)


@app.route('/face-recognition', methods=['GET', 'POST'])
def face_recognition():
    if request.method == 'GET':
        return Response('Face detection server is running.', status=200)

    # POST
    # Read image data
    img_data = request.json
    if 'image' not in img_data:
        return Response(ERR_MESSAGES[INVALID_REQUEST_ERR], status=400)

    # Process image
    res_code, candidates = process_image(img_data['image'])

    if res_code != IMAGE_PROCESS_OK:
        return Response(ERR_MESSAGES[res_code], status=500)

    else:
        response_text = ''
        for candidate in candidates:
            response_text += f"{candidate['image_id']} is detected as {candidate['sample_id']}, the distance is {candidate['distance']}.\n"

        return Response(response_text, status=200)


if __name__ == '__main__':
    # Run app in debug mode on port 6337
    app.run(debug=True, host='0.0.0.0', port=6337, threaded=True)