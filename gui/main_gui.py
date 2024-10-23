from tkinter import *
from PIL import Image,ImageTk
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk
import numpy as np
import cv2 as cv
from tensorflow import keras
import tensorflow as tf
import os
from PIL import Image
from pathlib import Path


# Load Model
source_path = "/home/leo/Desktop/PyaeHeinTun/Thesis/Code/gui/"
model_path = source_path+"model.keras"
model = tf.keras.models.load_model(model_path)

def readImage(imagePath):
    predictimg = cv.imread(imagePath)
    predictimg = cv.resize(predictimg,(500,400))
    return predictimg

def show(predictimg,name):
	cv.imshow(name,predictimg)
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

GUIlabelToText={
            0:"Stop Signs",
    		1:"20 Speed Limit Signs",
    		2:"30 Speed Limit Signs",
    		3:"50 Speed Limit Signs",
            4:"60 Speed Limit Signs",
            5:"70 Speed Limit Signs",
            6:"80 Speed Limit Signs",
            7:"100 Speed Limit Signs",
            8:"120 Speed Limit Signs",
            9:"No Entry Signs",
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
def returnHSV(predictimg):
    blur = cv.GaussianBlur(predictimg,(11,11),0)
    hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
    return hsv

# Binary The Image From HSV Range
def binaryImage(predictimg):

    image1 = predictimg.copy()
    image2 = predictimg.copy()
    image_blue = predictimg.copy()

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
def findContour(predictimg):
    contours, hierarchy = cv.findContours(predictimg,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    return contours

def findBiggestContour(contours):
    c=[cv.contourArea(i) for i in contours]
    return contours[c.index(max(c))]

def placeLabelText(text,box,predictimg):
    predictimg = cv.putText(predictimg,text,(int(box[0]+(box[2]/2)-5), int(box[1]-10)), cv.FONT_HERSHEY_SIMPLEX , 1 , (0,200,0) , 3 )
    return predictimg

def boundaryBox(predictimg,contours):
    x,y,w,h=cv.boundingRect(contours)
    box=cv.boundingRect(contours)
    predictimg=cv.rectangle(predictimg,(x,y),(x+w,y+h),(0,255,0),2)
    sign=predictimg[y:(y+h) , x:(x+w)]
    return predictimg,sign,box
    
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
        box = np.asarray(box)
        label = label
        return sign , box , label , result


def findRedSignRealTime(realFrame):
    b_img_real_red, _ = binaryImage(realFrame)
    real_contours = findContour(b_img_real_red)
    # For Checking Size
    big_real_contour = findBiggestContour(real_contours)
    real_area = cv.contourArea(big_real_contour)
    (_,_),r_real = cv.minEnclosingCircle(big_real_contour)

    # checking round shape
    if((real_area>0.42*np.pi*r_real*r_real)):
        img_real,sign_real,box_real = boundaryBox(realFrame,big_real_contour)
        # Preprocess IMG TO Predict
        image_fromarray_real = Image.fromarray(sign_real, 'RGB')
        resize_image_real = image_fromarray_real.resize((30, 30))
        expand_input_real = np.expand_dims(resize_image_real,axis=0)
        input_data_real = np.array(expand_input_real)
        input_data_real = input_data_real/255
        predicted_class_real = model.predict(input_data_real)
        result_real = predicted_class_real.argmax()
        label_real = labelToText[result_real]
        # box = np.asarray(box)
        return 'sign' , 'box' , label_real
       

#make a window
ws = Tk()
ws.title("Traffic Sign Detectino And Recognition")


#get wigth & height of screen
width= ws.winfo_screenwidth()
height= ws.winfo_screenheight()

#set screensize as fullscreen and not resizable
ws.geometry("%dx%d" % (width/2, height/2))
ws.resizable(False,False)

# Frame Option

mainFrame = Frame(ws)
singleFrame = Frame(ws,bg='#45aaf2')
realFrame = Frame(ws,bg='#45aaf2')

def change_frame(frame):
    frame.tkraise()

for frame in (mainFrame, singleFrame, realFrame):
    frame.grid(row=0, column=0, sticky='news')


# =================== Main Page ====================#
# Create Canvas
Maincanvas = Canvas(
    mainFrame,
    width = width, 
    height = height
) 


# put image in a label and place label as background
image = Image.open(source_path+"assets/mtu_gate.png")
image = image.resize((int(width/2), int(height/2)))
img = ImageTk.PhotoImage(image=image)

Maincanvas.create_image(
    0,
    0, 
    anchor=NW, 
    image=img,
)

Maincanvas.create_text(
	width/4, 
	height/15, 
	font=('Arial', 42),
    text="Traffic Sign Detection And Recognition",
    fill="red"
)

# Button Section
# =========BTN REALTIME
# Function
def RealImageRecognition():
    change_frame(realFrame)

btnRealtime = Button(
	mainFrame, 
    command=RealImageRecognition,
	text = 'Realtime Recognition',
	width=20,
	height=2,
	relief=SOLID,
	font=('arial', 18)
	)

btn_canvas_realtime = Maincanvas.create_window(
	width/3, 
	height/3, 
	anchor = "nw",
	window = btnRealtime,
	)

# =======BTN SINGLE IMAGE
# Function
def singleImageRecognition():
    change_frame(singleFrame)


# UI
btnSingleImage = Button(
	mainFrame, 
	text = 'Single Image Recognition',
    command=singleImageRecognition,
	width=20,
	height=2,
	relief=SOLID,
	font=('arial', 18)
	)

btn_canvas_singleimage = Maincanvas.create_window(
	30, 
	height/3, 
	anchor = "nw",
	window = btnSingleImage,
	)
  
Maincanvas.pack() 
change_frame(mainFrame)


#=================== Single Image Recognition ============
OriginalImageBorder = Canvas(
    singleFrame,
    width=350,
    height=250,
)
ResultImageBorder = Canvas(
    singleFrame,
    width=350,
    height=250,
)
original_image_show = Label(singleFrame)
result_image_show = Label(singleFrame)
# ====Function========
def selectPic():
    global original_img
    global filename
    filename = filedialog.askopenfilename(initialdir="/images", title="Select Image",
                           filetypes=(("jpg images","*.jpg"),("png images","*.png")))
    original_img = Image.open(filename)
    original_img = original_img.resize((350,250))
    original_img = ImageTk.PhotoImage(original_img)
    original_image_show['image'] = original_img
    OriginalImageBorder.destroy()
    original_image_show.place(x=20,y=height/6+10)
    

def goToHome():
    change_frame(mainFrame)

def recognizeImage():
    global result_img
    global filenameResult
    testCase = readImage(filename)
    sign,box,label,result = findRedSign(testCase)
    # result = 8
    sign = placeLabelText(label,box,testCase)
    cv.imwrite(source_path+'temp/1.png',sign)
    filenameResult = source_path+'temp/1.png'
    result_img = Image.open(filenameResult)
    result_img = result_img.resize((350,250))
    result_img = ImageTk.PhotoImage(result_img)
    result_image_show['image'] = result_img
    ResultImageBorder.destroy()
    result_image_show.place(x=int(width/3.3),y=height/6+10)
    messagebox.showinfo(title="Predicted Output",message='Predicted Output ===== \n\n'+GUIlabelToText[result])
    

# ==== UI =========
# UI
OriginalImageBorder.create_rectangle(350,350, 1, 1, outline='green')
OriginalImageBorder.place(x=20,y=height/6+10)

ResultImageBorder.create_rectangle(350,350, 1, 1, outline='green')
ResultImageBorder.place(x=int(width/3.3),y=height/6+10)

Button(
	singleFrame, 
	text = 'SELECT',
    command=selectPic,
	width=20,
	height=2,
	relief=SOLID,
	font=('arial', 18)
	).place(x=20,y=20)

Button(
	singleFrame, 
	text = 'RECOGNIZE',
    command=recognizeImage,
	width=20,
	height=2,
	relief=SOLID,
	font=('arial', 18)
	).place(x=int(width/3),y=20)

Button(
	singleFrame, 
	text = 'HOME',
    command=goToHome,
	width=20,
	height=2,
	relief=SOLID,
	font=('arial', 18)
	).place(x=int(width/6),y=450)

Label(singleFrame, text='ORIGINAL', padx=25, pady=25,font=('verdana',18), bg='#45aaf2').place(x=105,y=height/10)
Label(singleFrame, text='RESULT', padx=25, pady=25,font=('verdana',18), bg='#45aaf2').place(x=width/3+50,y=height/10)

#=================== Realtime Image Recognition ============
# Value
Realcanvas = Canvas(
    realFrame,
    width = width, 
    height = height
)

OriginalRealImageBorder = Canvas(
    realFrame,
    width=350,
    height=250,
)
ResultRealImageBorder = Canvas(
    realFrame,
    width=350,
    height=250,
)

original_image_real_show = Label(realFrame)
result_image_real_show = Label(realFrame)
result_text_real_show = Label(realFrame)
# Function
global cam

def startVideoServer():
    global frame
    global cam
    cam = cv.VideoCapture(0)
    OriginalRealImageBorder.destroy()
    original_image_real_show.place(x=20,y=height/6+10)
    result_text_real_show.place(x=430,y=350)
    result_image_real_show.place(x=int(width/3.3),y=height/6+10)
    while(True):
        ret, frame = cam.read()

        #Update the image to tkinter...
        frame = cv.resize(frame,(350,250))
        real_original_frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        img_update = ImageTk.PhotoImage(Image.fromarray(real_original_frame))
        original_image_real_show.configure(image=img_update)
        original_image_real_show.image = img_update
        original_image_real_show.update()
        try:
            sign , box , label , result = findRedSign(frame)
            predict_img_real = placeLabelText(label,box,frame)
            result_text_real_show.configure(text=label,font=('arial',30))
            result_text_real_show.text = label
            result_text_real_show.update()

            #Update the image to tkinter...
            frame_result = cv.resize(predict_img_real,(350,250))
            frame_result = cv.cvtColor(frame_result,cv.COLOR_BGR2RGB)
            img_update_result = ImageTk.PhotoImage(Image.fromarray(frame_result))
            result_image_real_show.configure(image=img_update_result)
            result_image_real_show.image = img_update_result
            result_image_real_show.update()

        except:
            result_image_real_show.configure(image=img_update)
            result_image_real_show.image = img_update
            result_image_real_show.update()
        k = cv.waitKey(1)
        if k%256 == 27:
            cam.release()
            cv.destroyAllWindows()
            break
        

def stopVideoServer():
    global cam
    cam.release()
    cv.destroyAllWindows()
    print("Stopped!")


# UI
OriginalRealImageBorder.create_rectangle(350,350, 1, 1, outline='green')
OriginalRealImageBorder.place(x=20,y=height/6+10)

ResultRealImageBorder.create_rectangle(350,350, 1, 1, outline='green')
ResultRealImageBorder.place(x=int(width/3.3),y=height/6+10)

Realcanvas.create_image(
    0,
    0, 
    anchor=NW, 
    image=img,
)

Button(
	realFrame, 
	text = 'START',
    command=startVideoServer,
	width=20,
	height=2,
	relief=SOLID,
	font=('arial', 18)
	).place(x=20,y=20)

Button(
	realFrame, 
	text = 'STOP',
    command=stopVideoServer,
	width=20,
	height=2,
	relief=SOLID,
	font=('arial', 18)
	).place(x=int(width/3),y=20)

Button(
	realFrame, 
	text = 'HOME',
    command=goToHome,
	width=20,
	height=2,
	relief=SOLID,
	font=('arial', 18)
	).place(x=int(width/6),y=450)

Realcanvas.pack()
 
ws.mainloop()
