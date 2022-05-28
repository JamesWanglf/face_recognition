import base64
import json
import os
import requests

hostname = 'http://127.0.0.1:6337'
dir_path = os.path.join(os.path.dirname(__file__), 'dataset')


def update_sample_database_test():
    valid_images = ['.jpg', '.gif', '.png']

    data_list = []
    for f in os.listdir(dir_path):
        filename = os.path.splitext(f)[0]
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        file_path = os.path.join(dir_path, f'{filename + ext}')
        with open(file_path, 'rb') as img_file:
            base64_img = base64.b64encode(img_file.read()).decode('utf-8')

            data_list.append({
                'id': filename,
                'name': filename + ext,
                'image': f'data:image/{ext[1:]};base64,{base64_img}',
                'metadata': filename + ext,
                'action': 'embedlink'
            })
    
    url = f'{hostname}/update-samples'
    headers = {
        'Content-Type': 'application/json'
    }
    payload = json.dumps(data_list)

    response = requests.post(url, headers=headers, data=payload)

    print(response.text)


def face_recognition_test(number):
    file_path = os.path.join(dir_path, f'img{number}.jpg')
    with open(file_path, 'rb') as img_file:
        base64_img = base64.b64encode(img_file.read()).decode('utf-8')

        img_data = {
            'image': f'data:image/jpeg;base64,{base64_img}',
            'min_distance': 9
        }

    url = f'{hostname}/face-recognition'
    headers = {
        'Content-Type': 'application/json'
    }
    payload = json.dumps(img_data)

    response = requests.post(url, headers=headers, data=payload)

    print(response.text)


if __name__ == '__main__':
    # Update sample database with the images in /dataset directory
    # update_sample_database_test()

    # Test face recognition with "img1.jpg"
    face_recognition_test(1)
