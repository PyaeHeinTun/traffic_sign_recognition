import numpy as np
import cv2 as cv
from tensorflow import keras
import tensorflow as tf
import os
from PIL import Image


# Load Model
model_path = "model.h5"
model = tf.keras.models.load_model(model_path)

# Video Capture Or Image Source
video_path = ""
cap =  cv.VideoCapture(video_path)
ImagesFilePath='/media/leo/Data/Thesis/TestImage'
ImageNamePath=os.listdir(ImagesFilePath)

def readImage(imagePath):
    img = cv.imread(ImagesFilePath+'/'+imagePath,1)
    img = cv.resize(img,(500,400))
    return img

def show(img,name):
	cv.imshow(name,img)
	cv.waitKey(0)
	cv.destroyAllWindows()

# Image Detection Alogrithm
labelToText={
            0:"Stop",
    		1:"(20)",
    		2:"(30)",
    		3:"(50)",
            4:"(60)",
            5:"(70)",
            6:"(80)",
            7:"(100)",
            8:"(120)",
            9:"No Entry",
            }

# Threshold Value
# ==== Red ====
low_thresh1 =  (165,100,40)
high_thresh1 = (179,255,255)

low_thresh2 = (0,160,40)
high_thresh2 = (10,255,255)


# ==== Blue ====
low_thresh3 = (100,150,40)
high_thresh3 = (130,255,255)



# HSV Image
def returnHSV(img):
    blur = cv.GaussianBlur(img,(11,11),0)
    hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
    return hsv

# Binary The Image From HSV Range
def binaryImage(img):

    image1 = img.copy()
    image2 = img.copy()
    image_blue = img.copy()

    hsv1 = returnHSV(image1)
    hsv2 = returnHSV(image2)
    hsvBlue = returnHSV(image_blue)

    b_img1 = cv.inRange(hsv1,low_thresh1,high_thresh1)
    b_img2 = cv.inRange(hsv2,low_thresh2,high_thresh2)
    # binarize red sign image
    # 0*0 - 0
    # 1*0 - 1
    # 1*1 - 1
    # 0*1 - 1
    b_img_red = cv.bitwise_or(b_img1,b_img2)

    # binarize blue sign image
    b_img_blue = cv.inRange(hsvBlue,low_thresh3,high_thresh3)
    return b_img_red, b_img_blue


# findng contour
def findContour(img):
    contours, hierarchy = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    return contours

def findBiggestContour(contours):
    c=[cv.contourArea(i) for i in contours]
    return contours[c.index(max(c))]

def placeLabelText(text,box,img):
    cv.putText(img,text,(int(box[0]+(box[2]/2)-5), int(box[1]-10)), cv.FONT_HERSHEY_SIMPLEX , 1 , (0,200,0) , 3 )

def boundaryBox(img,contours):
    x,y,w,h=cv.boundingRect(contours)
    box=cv.boundingRect(contours)
    img=cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    sign=img[y:(y+h) , x:(x+w)]
    return img,sign,box
    
# preprocessing image
def preprocessingImageToClassifier(image=None,imageSize=28,mu=89.77428691773054,std=70.85156431910688):
    image = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    image = cv.resize(image,(imageSize,imageSize))
    image = (image - mu) / std
    image = image.reshape(1,imageSize,imageSize,1)
    return image

# finding red sign
def findRedSign(frame):
    b_img_red, _ = binaryImage(frame)
    contours = findContour(b_img_red)
    # For Checking Size
    big_contour = findBiggestContour(contours)
    area = cv.contourArea(big_contour)
    (a,b),r = cv.minEnclosingCircle(big_contour)

    # checking round shape
    if((area>0.42*np.pi*r*r)):
        img,sign,box = boundaryBox(frame,big_contour)
        # Preprocess IMG TO Predict
        image_fromarray = Image.fromarray(sign, 'RGB')
        resize_image = image_fromarray.resize((30, 30))
        expand_input = np.expand_dims(resize_image,axis=0)
        input_data = np.array(expand_input)
        input_data = input_data/255
        predicted_class = model.predict(input_data)
        result = predicted_class.argmax()
        label = labelToText[result]
        # box = np.asarray(box)
        return 'sign' , 'box' , label
       

cap=cv.VideoCapture(0)

while(True):
    _, frame = cap.read()
    # detection code run
    label = "Unknown"  #Traffic Sign Label (CNN Network)
    try:
        sign,box , label = findRedSign(frame)
        print(label)
        # placeLabelText(label,box,frame)
    except:
        cv.imshow('frame',frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()