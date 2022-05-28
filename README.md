# Face recognition
This is a face recognition work based on [DeepFace](https://github.com/serengil/deepface).  
This project contains three modules in 2 Python scripts, the first one is for the face detection from the input image, the second one is for the feature extraction from the detected face and the third one is for comparing the feature vectors between dataset and detected face.  
We will divide the whole project to two parts: face detection and feature comparation, and feature extraction module. So we have two Python scripts for them.  
Now, let's try to run the project on ubuntu.
1. Prerequisite  
   - Install cuda and cudnn.  
     I installed cuda11.2 and libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb downloaded from [Nvidia site](https://developer.nvidia.com/cudnn).  
     If you are using the different version of cuda and cudnn, you need to set the suitable version of tensorflow in [requirements.txt](https://github.com/JamesWanglf/face_recognition/blob/main/requirements.txt)  
   - Please create virtual env and install the Python packages described in requirements.txt.  
     `python3 -m venv venv`  
     `source ./venv/bin/activate`  
     `pip install -r requirements.txt`
   - Install mod_wsgi  
     `sudo apt-get install libapache2-mod-wsgi-py3`

2. First, run the feature detection module.  
`cd ./feature_extraction`  
`gunicorn -w 2 -b 0.0.0.0:5000 wsgi:app`  
This will run the Flask http server on 0.0.0.0:5000. When you can access this url from your browser, it will show you "Feature extraction server is running!"  

3. Run the main module.  
`cd ./face_detection`  
`python3 face_detection.py`  
This will run the http server on 0.0.0.0:6337.

Now, the main project provides two endpoints like:
- http://0.0.0.0:6337/update-samples  
  **type: POST**  
  **Request**:  
  ```
  curl --location --request POST 'http://0.0.0.0:6337/update-samples'
  --header 'Content-Type: application/json'
  --data-raw '[
    {
      "id": "person1",
      "name: "person1",
      "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD....",
      "metadata": "mydomain.com/myobject1",
      "action": "embedlink"
    },
    {
      "id": "person2",
      "name: "person2",
      "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD....",
      "metadata": "mydomain.com/myobject2",
      "action": "embedlink"
    }
  ]'
  ```  
  You can send base64-encoded image to this endpoint. This will read the image data, and extract the feature vector from the detected face in the original image. We will save these features in the database(sqlite) or global variable.(this is optional)  
  
  **Response**:  
  status_code: 200  
  response:  
  ```
  {
      "success": [all success ids],
      "fail": [list of failed ids]
  }
  ```
  
  status_code: 400  
  response:  
  ```
  {
      "error": "Invalid request."
  }
  ```
  
- http://0.0.0.0:6337//face-recognition  
  This will return the verified faces inside the image.jpg, and try to find the closest face among the sample data, and return its id, name, metadata and bounding box.    
  **type: POST**  
  **Request:**  
  ```
  curl --location --request POST 'http://0.0.0.0:6337/face-recognition'
  --header 'Content-Type: application/json'
  --data-raw '{
      "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD....",
      "min_distance": 9
   }'
   ```  
   "min_distance" is optional field. The value of this field is related with the feature extraction model.  
     For example, it will be in [0, 1] for "vgg-face" model, [0, 20] for "facenet" model.  
     Since we are using "facenet" model as default, the ideal threshold is 9.(0.4 for "vgg-face" model)
  
   **Response**:  
   status_code: 200  
   response:  
   ```
   [
      {
         "bbox": "x, y, w, h", 
         "id": "", 
         "metadata": "", 
         "name": ""
      },
      {
         "bbox": "x, y, w, h", 
         "id": "", 
         "metadata": "", 
         "name": ""
      },
   ]
   ```
   
   status_code: 400  
   response:  
   ```
   {
      "error": "Invalid request."
   }
   ```
   
   status_code: 500  
   response:  
   ```
   {
      "error": "error message"
   }
   ```
  
  You can check the Python test code in test_request.py.
