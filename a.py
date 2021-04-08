import os
import numpy as  np
import cv2 
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Add the directory here to where the images are stored use double spaces i.e \\
dir = 'C:\\Users\\otweyo brian wilson\\Desktop\\Faces\\dataset'
categories =['freakles', 'glasses', 'hair_color','hair_top', 'wrinkles']
# define freakles by zero and glasses by one same for features and so on

data = []
for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        face_img = cv2.imread(imgpath, 0)
        cv2.imshow('image',face_img)
        break
    break
cv2.waitKey(0)
cv2.destroyAllWindows()