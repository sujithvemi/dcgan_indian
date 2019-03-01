import os
import matplotlib.pyplot as plt

lfw_source = './data/lfw-deepfunneled/'
lfw_fun_dest = './data/LFW_funneled/'
lfw_fun_india_dest = './data/LFW_funneled_Indian/'

if not os.path.isdir(lfw_fun_dest):
    os.mkdir(lfw_fun_dest)

if not os.path.isdir(lfw_fun_india_dest):
    os.mkdir(lfw_fun_india_dest)

lfw_names = os.listdir(lfw_source)
lfw_names = [name for name in lfw_names if not name.startswith('.')]
for person in lfw_names:
    images = os.listdir(lfw_source + person)
    images = [image for image in images if not image.startswith('.')]
    for image in images:
        img = plt.imread(lfw_source + person + '/' + image)
        plt.imsave(fname=lfw_fun_dest+image, arr=img)
        plt.imsave(fname=lfw_fun_india_dest+image, arr=img)

indian_faces_src = './data/indian_faces/'
ind_face_imgs = os.listdir(indian_faces_src)
ind_face_imgs = [image for image in ind_face_imgs if not image.startswith('.')]

for image in ind_face_imgs:
    img = plt.imread(indian_faces_src + image)
    plt.imsave(fname=lfw_fun_india_dest+image, arr=img)