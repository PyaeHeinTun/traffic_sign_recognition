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


source_path = "/home/leo/Desktop/PyaeHeinTun/Thesis/Code/gui/"
p = Path(source_path)
model_path = source_path+"model.keras"
model = tf.keras.models.load_model(model_path)