import cgi
import cv2
import io
import json
import os
import random
from charset_normalizer import detect
import numpy as np
import requests
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from flask import Flask, request, Response
from urllib.parse import urlparse, parse_qs
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
UPDATE_SAMPLE_FACES_OK = 200
UPDATE_SAMPLE_FACES_ERR = 201
FEATURE_EXTRACTION_SERVER_CONNECTION_ERR = 202
FEATURE_EXTRACTION_REQUEST_ERR = 203
FEATURE_EXTRACTION_SERVER_RESPONSE_OK = 204
FEATURE_EXTRACTION_SERVER_RESPONSE_PARSE_ERR = 205
FACE_DETECTION_OK = 210
FACE_DETECTION_ERR = 211
NO_FACE_DETECTED_ERR = 212
CALC_DISTANCE_OK = 220
CALC_DISTANCE_ERR = 221
NO_SUCH_FILE_ERR = 230
INVALID_IMAGE_ERR = 231
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
    NO_SUCH_FILE_ERR: 'No such file.',
    INVALID_IMAGE_ERR: 'Invalid image has input. Could not read the image data.',
    UNKNOWN_ERR: 'Unknown error has occurred.'
}


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
        img = functions.load_image(DIR_PATH + img)
    elif isinstance(img, bytes):
        img = np.array(im.open(io.BytesIO(img)))

    # rescale the image to the smaller size
    img, scaled_ratio = rescale_image(img)

    # faces stores list of detected_face and region pair
    faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align)

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


def update_feature_vector_database():
    """
    Save the feature vector database with the sample face images
    """
    global SAMPLE_FACE_VECTOR_DATABASE

    # Read sample images in the dataset directory
    valid_images = [".jpg",".gif",".png"]
    face_list = []
    face_feature_vector_list = []
    for f in os.listdir(DIR_PATH):
        filename = os.path.splitext(f)[0]
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue

        # Detect face from sample image
        detected_faces, scaled_ratio = detect_faces(filename + ext)

        # Get the first face from the detected faces list. Suppose that the sample image has only 1 face
        face = detected_faces[0]    # tuple (np.ndarray, list) - np.adarray is image. list is face region, e.x. [x, y, w, h]

        # # Get face region from the base image(profile image)
        # face_region = recover_face_region(face[1], scaled_ratio)

        face_list.append({
            'id': 0,
            'img': face[0],
            'sample_name': filename + ext
        })

        if len(face_list) == FEATURE_EXTRACT_BATCH_SIZE:
            res_code, face_feature_data = call_feature_extractor(face_list)
            
            if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
                return res_code, None

            for i in range(len(face_feature_data)):
                face_feature_data[i]['sample_name'] = face_list[i]['sample_name']

            face_feature_vector_list += face_feature_data

            face_list = []

    if len(face_list) > 0:
        res_code, face_feature_data = call_feature_extractor(face_list)
        
        if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
            return res_code, None

        for i in range(len(face_feature_data)):
            face_feature_data[i]['sample_name'] = face_list[i]['sample_name']

        face_feature_vector_list += face_feature_data

    # Save the vectors of sample faces into global variable
    SAMPLE_FACE_VECTOR_DATABASE = []
    for feature_data in face_feature_vector_list:
        SAMPLE_FACE_VECTOR_DATABASE.append({
            'id': feature_data['sample_name'],
            'feature_vector': feature_data['vector']
        })

    return UPDATE_SAMPLE_FACES_OK, len(SAMPLE_FACE_VECTOR_DATABASE)


def find_face(face_feature_vectors):
    """
    Find the closest sample by comparing the feature vectors
    """
    global SAMPLE_FACE_VECTOR_DATABASE

    candidates = []
    for vector_data in face_feature_vectors:
        image_id = vector_data['id']
        face_feature_vector = vector_data['vector']

        # Initialize variables
        closest_sample_id = ''
        closest_distance = -1
        
        # Compare with sample vectors
        for i in range(len(SAMPLE_FACE_VECTOR_DATABASE)):
            sample = SAMPLE_FACE_VECTOR_DATABASE[i]
            sample_vector = sample['feature_vector']

            try:
                # Calculate the distance between sample and the detected face.
                distance_vector = np.square(face_feature_vector - sample_vector)
                distance = np.sqrt(distance_vector.sum())

                if closest_sample_id == '' or abs(distance) < abs(closest_distance):
                    closest_distance = abs(distance)
                    closest_sample_id = sample['id']

            except Exception as e:
                print(e)
                pass
        
        # Add candidate for this face
        candidates.append({
            'image_id': image_id,
            'sample_id': closest_sample_id,
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

    if status == CALC_DISTANCE_ERR:
        return status, None

    return IMAGE_PROCESS_OK, candidates


def update_samples():
    """
    Update the database that contains sample face vectors
    """
    res, faces_count = update_feature_vector_database()

    if res == UPDATE_SAMPLE_FACES_OK:
        return f'There are {faces_count} sample faces.'

    return ERR_MESSAGES[res]


class HttpServerHandler(SimpleHTTPRequestHandler):

    def parse_url(self):
        full_path = f'http://{HOSTNAME}:{PORT}{self.path}'
        parsed_url = urlparse(full_path)
        query = parse_qs(parsed_url.query)

        return query

    def do_GET(self):
        if self.path.startswith('/update-samples'):
            response_text = update_samples()
            self.send_successs_response(response_text)

        elif self.path.startswith('/face-recognition'):
            # Parse Query
            query = self.parse_url()
            if 'image_name' not in query:
                self.send_bad_request_response()

            # Process the dedicated image
            res_code, candidates = process_image(query['image_name'][0])

            response_text = ''
            if res_code != IMAGE_PROCESS_OK:
                response_text = ERR_MESSAGES[res_code]
            else:
                for candidate in candidates:
                    response_text += f"{candidate['image_id']} is detected as {candidate['sample_id']}, the distance is {candidate['distance']}.\n"

            self.send_successs_response(response_text)

    def do_POST(self):
        if self.path.startswith('/face-recognition'):
            ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
            pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
            pdict['CONTENT-LENGTH'] = int(self.headers['Content-Length'])
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD':'POST', 'CONTENT_TYPE':self.headers['Content-Type'], })
            response_text = ''
            try:
                # if isinstance(form['image'], list):
                #     for record in form["image"]:
                #         image_data = record.file.read()

                if isinstance(form['image'], cgi.FieldStorage):
                    image_data = form['image'].file.read()

                    res_code, candidates = process_image(image_data)

                    if res_code != IMAGE_PROCESS_OK:
                        response_text = ERR_MESSAGES[res_code]
                    else:
                        for candidate in candidates:
                            response_text += f"{candidate['image_id']} is detected as {candidate['sample_id']}, the distance is {candidate['distance']}.\n"

            except IOError:
                response_text = ERR_MESSAGES[INVALID_IMAGE_ERR]

            self.send_successs_response(response_text)
        
        else:
            self.send_bad_request_response()

    def send_successs_response(self, msg):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(msg, "utf-8"))

    def send_bad_request_response(self, message=None):
        self.send_response(400)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        if message:
            self.wfile.write(bytes(f"Bad Request: {message}", "utf-8"))
        else:
            self.wfile.write(bytes(f"Bad Request", "utf-8"))


if __name__ == '__main__':
    
    webServer = HTTPServer((HOSTNAME, PORT), HttpServerHandler)
    print("Feature Comparing Server started http://%s:%s" % (HOSTNAME, PORT))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")