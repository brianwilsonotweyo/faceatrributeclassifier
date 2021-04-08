import os
import numpy as  np
import cv2 
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# # Add the directory here to where the images are stored use double spaces i.e \\
# dir = 'C:\\Users\\otweyo brian wilson\\Desktop\\Faces\\dataset'
# categories =['freakles', 'glasses', 'hair_color','hair_top', 'wrinkles']
# # define cat by zero and dog by one same for features

# data = []
# for category in categories:
#     path = os.path.join(dir, category)
#     label = categories.index(category)
#     for img in os.listdir(path):
#         imgpath = os.path.join(path, img)
#         face_img = cv2.imread(imgpath, 0)
#         try:
#             face_img = cv2.resize(face_img,(50,50))
#             image = np.array(face_img).flatten()

#             data.append([image, label])
#         except Exception as e:
#             pass  

# pick_in = open('data1.pickle', 'wb')

# pickle.dump(data,pick_in)
# pick_in.close()


# Loading the data
pick_in = open('data1.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

#dividing the data into features and labels
random.shuffle(data)
features = []
labels =[]

for feature , label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.98)

model = SVC(C=1,gamma='auto',kernel='poly')
model.fit(xtrain, ytrain)

pick = open('model.sav','wb')
pickle.dump(model,pick)
pick.close()


# prediction = model.predict(xtest)
# accuracy = model.score(xtest, ytest)

# categories =['freakles', 'glasses', 'hair_color','hair_top', 'wrinkles']

# print('Accuracy: ', accuracy)
# print('Prediction is: ', categories[prediction[0]])

# myface = xtest[0].reshape(50,50)
# plt.imshow(myface, cmap='gray')
# plt.show(myface, cmap='gray')