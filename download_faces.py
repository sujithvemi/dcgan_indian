"""
    This script downloads the images from 'https://ruralindiaonline.org/categories/faces'
"""
import requests
import os
import json
import re

FACE_JSON = './data/faces.json'

with open(FACE_JSON) as f:
    data = json.load(f)

try:
    os.mkdir('./data/images')
except:
    pass

for i, photo in enumerate(data):
    url = photo['photo']
    file_name = photo['photo'].split('/')[-1]
    print(i)
    with open('./data/images/' + file_name, 'wb') as handle:
        response = requests.get(url, stream=True)
        if not response.ok:
            print(response)
        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)