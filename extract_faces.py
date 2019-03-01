"""
    This script extracts the faces from images and saves them in separate files. This script uses 'Haar Cascades'
"""
import os
import cv2
import numpy as np
import shutil

DATA_DIR = './data/images/'
FACES_DIR = './data/faces/'

try:
    print("Making 'faces' directory")
    os.mkdir(FACES_DIR)
except:
    print("Deleting 'faces' directory and all its contents and making a fresh directory")
    shutil.rmtree(FACES_DIR)
    os.mkdir(FACES_DIR)

images = os.listdir(DATA_DIR)
images = [image for image in images if not image.startswith('.')]

face_cascade = cv2.CascadeClassifier('detectors/haarcascade_frontalface_default.xml')

for i, image in enumerate(images):
    try:
        img_color = cv2.imread(DATA_DIR + image)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        height, width = img_gray.shape[0:2]
        faces = face_cascade.detectMultiScale(img_gray, 1.1, 5)
        # print(i, '-', image, '-', len(faces))
        try:
            if len(faces) <= 0:
                cv2.imwrite(FACES_DIR + image, img_color)
            else:
                for j in range(len(faces)):
                    (x, y, w, h) = faces[j]
                    left = max(x-75, 0)
                    top = max(y-125, 0)
                    right = min(x+w+75, width)
                    bottom = min(y+h+125, height)
                    new_face = img_color[top:bottom, left:right, :]
                    cv2.imwrite(FACES_DIR + str(j) + '-' + image, new_face)
        except Exception as e:
            print("Failed to save face in image " + image)
            print(e)
            pass
    except Exception as e:
        print("Failed to read image or faces in " + image)
        print(e)
        pass