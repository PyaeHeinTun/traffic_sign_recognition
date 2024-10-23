import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
np.random.seed(42)

data_dir = "/media/leo/Data/Thesis/Datasets"
# Resizing the images to 30x30x3
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

# Load Model
model_path = "model.h5"
model = tf.keras.models.load_model(model_path)

test = pd.read_csv(data_dir + '/Test' + '/Test.csv')

labels = test["ClassId"].values
imgs = test["Path"].values
imgs_array = imgs.tolist()
ImagesFilePath=data_dir+"/Test"
ImageNamePath=os.listdir(ImagesFilePath)
predict_number = 0

predict_compare = []
data =[]

for img in ImageNamePath:
    try:
        image = cv2.imread(data_dir+"/Test/" + img)
        image_path = "Test/"+img
        getted_label = labels[imgs_array.index(image_path)]
        actual_label = 0
        if getted_label == 14:
            actual_label = 0
        elif getted_label == 0:
            actual_label = 1
        elif getted_label == 1:
            actual_label = 2
        elif getted_label == 2:
            actual_label = 3
        elif getted_label == 3:
            actual_label = 4
        elif getted_label == 4:
            actual_label = 5
        elif getted_label == 5:
            actual_label = 6
        elif getted_label == 7:
            actual_label = 7
        elif getted_label == 8:
            actual_label = 8
        elif getted_label == 17:
            actual_label = 9
        predict_compare.append(actual_label)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((IMG_HEIGHT,IMG_WIDTH))
        data.append(np.array(resize_image))
        predict_number = predict_number + 1
    except:
        print("Error in " + img)

X_test = np.array(data)
X_test = X_test/255
# pred = model.predict_classes(X_test)
predicted_class = model.predict(X_test)
result = []
for data in predicted_class:
    result.append(data.argmax())
print(accuracy_score(predict_compare,result)*100)