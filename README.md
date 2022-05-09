# Face recognition
This is a face recognition work based on [DeepFace](https://github.com/serengil/deepface).  
This project contains three modules in 2 Python scripts, the first one is for the face detection from the input image, the second one is for the feature extraction from the detected face and the third one is for comparing the feature vectors between dataset and detected face.  
We will divide the whole project to two parts: face detection and feature comparation, and feature extraction module. So we have two Python scripts for them.  
Now, let's try to run the project.
1. Prerequisite  
Please create virtual env and install the Python packages described in requirements.txt.  
`python3 -m venv venv`  
`source ./venv/bin/activate`  
`pip install -r requirements.txt`

2. First, run the feature detection module.  
`python3 feature_extraction_module.py`  
This will run the Flask http server on 127.0.0.1:5000.

3. Run the main module.  
`python3 face_detection_module.py`  
This will run the http server on 0.0.0.0:6337.

Now, this project provides two endpoints like:
- /update-samples  
  type: GET  
  This will read the image files from /dataset directory, and extract the all the features from the detected faces in those files. We will save these features in the global variable.
- /face-recognition  
  type: POST  
  example request:  
  `curl --location --request POST 'http://(domain or ip address):6337/face-recognition' --form 'image=@"/image_path/image.jpg"'`  
  This will return the verify the faces inside the image.jpg, and try to find the closest face among the /dataset directory, and return the filename.
