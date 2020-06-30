from __future__ import print_function
import numpy as np
import os
import sys
import logging
#hello
import tkinter
#from collections import defaultdict
import cv2
import os
print("Some imports done")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)
# Just disables the warning shown at the end, doesn't enable AVX/FMA(Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
#import tensorflow as tf
sys.path.append("..")

#from utils import label_map_util
#from utils import visualization_utils as vis_util
from utils import backbone
# api import object_counting_api
#import json



from predictor import cumulative_object_counting_x_axis

input_video = "mouse.mp4"

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
#detection_graph, category_index = backbone.set_model('oldtestped_inference_graph', 'object-detection.pbtxt')

is_color_recognition_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
import tkinter as tk
from tkinter import *
from tkinter import Image as im
from savecsv import savetocsv
def click(key):
    
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    # print the key that was pressed
    print( "\n" + str(key.char))
    x = int(x_min_form.get())
    y = int(y_min_form.get())
    w = int(x_max_form.get())
    h = int(y_max_form.get())
    
    

    #if backspacew then remove1
    print("l"+str(((key.char).isalpha()))+"l")
    if (key.char).isalpha() or str(key.char) == '4' or str(key.char) == '.':
        vid = video_form.get() + key.char#gets the previous text before letter , the u add the letter
        
    else:
        vid = video_form.get()[0:len(video_form.get())-2]
        
    print("\n" + vid + "\n")
    try:
        
        cap = cv2.VideoCapture(vid)
        for i in range(0,1):#to get 1 frame
            ret, frame = cap.read()

            if not ret:
                print("end of the video file...")
                break
                
            crop_frame = frame[y:y + h, x:x + w]
            
            cv2.imshow("window",crop_frame)
            cv2.waitkey(0)
    except:
        pass

def preview(vid,x,y,w,h):
    print("\n" + vid + "\n")
    try:
        
        cap = cv2.VideoCapture(vid)
        for i in range(0,1):#to get 1 frame
            ret, frame = cap.read()

            if not ret:
                print("end of the video file...")
                break
                
            crop_frame = frame[y:y + h, x:x + w]
            
            cv2.imshow("window",crop_frame)
            cv2.waitkey(0)
    except:
        pass
#do for each number entry
def reshow_x_min(key):
    try:
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        # print the key that was pressed
        print( "\n" + str(key.char))
        if (key.char).isdigit():
            x = int(x_min_form.get()+str(key.char))
        else:
            x = int(x_min_form.get()[0:len(x_min_form.get())-2])#to actually have a backspace bew romved on a backspace

        y = int(y_min_form.get())
        w = int(x_max_form.get())
        h = int(y_max_form.get())
        
        

        vid = video_form.get()
        preview(vid,x,y,w,h)
    except:
        print("cant display yet")


#do for each number entry
def reshow_y_min(key):
    try:
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        # print the key that was pressed
        print( "\n" + str(key.char))

        x = int(x_min_form.get())
        if (key.char).isdigit():
            y = int(y_min_form.get()+str(key.char))
        else:
            y = int(y_min_form.get()[0:len(y_min_form.get())-2])#to actually have a backspace bew romved on a backspace

        
        w = int(x_max_form.get())
        h = int(y_max_form.get())
        
        

        vid = video_form.get()
            
        preview(vid,x,y,w,h)
    except:
        print('cant display yet')
    
#do for each number entry
def reshow_x_max(key):
    try:
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        # print the key that was pressed
        print( "\n" + str(key.char))

        x = int(x_min_form.get())
        if (key.char).isdigit():
            w = int(x_max_form.get()+str(key.char))
        else:
            w = int(x_max_form.get()[0:len(x_max_form.get())-2])#to actually have a backspace bew romved on a backspace

        
        y = int(y_min_form.get())
        h = int(y_max_form.get())
        
        

        vid = video_form.get()
            
        preview(vid,x,y,w,h)
    except:
        print('cant display yet')

def reshow_y_max(key):
    try:
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        # print the key that was pressed
        print( "\n" + str(key.char))

        x = int(x_min_form.get())
        if (key.char).isdigit():
            h = int(y_max_form.get()+str(key.char))
        else:
            h = int(y_max_form.get()[0:len(y_max_form.get())-2])#to actually have a backspace bew romved on a backspace

        
        y = int(y_min_form.get())
        w = int(x_max_form.get())
        
        

        vid = video_form.get()
            
        preview(vid,x,y,w,h)
    except:
        print("Cant display yet")
   
master = Tk()

master.title("Pedestrain Counter")



master.geometry('600x600')

video_label = Label(master, text='Video file\nFile name cannot contain anything other than ".mp4" and the letters of the alphabet' )
video_label.pack()

video_form = Entry(master)
video_form.pack()
# Bind entry to any keypress
video_form.bind("<Key>", click)


x_form_label = Label(master, text="Top left(norm 700) or 820 now")
x_form_label.pack()

x_min_form = Entry(master)
x_min_form.pack()
x_min_form.bind("<Key>", reshow_x_min)

y_form_label = Label(master, text="Top right(norm 66)")
y_form_label.pack()

y_min_form = Entry(master)
y_min_form.pack()
y_min_form.bind("<Key>", reshow_y_min)


x_max_label = Label(master, text="Width(norm 468)")
x_max_label.pack()

x_max_form = Entry(master)
x_max_form.pack()
x_max_form.bind("<Key>", reshow_x_max)


y_max_label = Label(master, text="Height(norm 800)")
y_max_label.pack()

y_max_form = Entry(master)
y_max_form.pack()
y_max_form.bind("<Key>", reshow_y_max)

from datetime import datetime
 

def clicked():
    global use_coco
    
    if use_coco:
        label = "person"
        print("Using the coco model")
        detection_graph, category_index = backbone.set_model('coco', 'coco-labelmap.pbtxt')
    else:
        print("Using default")
        detection_graph, category_index = backbone.set_model('oldtestped_inference_graph', 'object-detection.pbtxt')
        label = "pedestrian"
        
    input_video = str(video_form.get())
    try:
        x = int(x_min_form.get())
        y = int(y_min_form.get())
        w = int(x_max_form.get())
        h = int(y_max_form.get())
    except ValueError:
        print("Values must be only number")
    #w = x_max - x_min
    #h = y_max - y_min
    print(x,y,w,h)


    print(label)
    cv2.destroyAllWindows()
    total, framecount , vidnameused = cumulative_object_counting_x_axis(input_video, detection_graph, category_index,is_color_recognition_enabled,x,y,w,h,label ,write=True, display=True, brighten = False)
    print("Your total is: " + str(total))
    total_label = Label(master, text="Around "+str(total) + " people walked by in " + str(vidnameused))
    total_label.pack()
    master.update()

    #getting current time
    currentDT = datetime.now()
    print (str(currentDT))
    #making a csv file with the data collected
    row = [str(currentDT), str(total),str(framecount), str(vidnameused)]
    header = ["Date and Time","Amount_past", "Frame_count", "VidName"]
    print("sending csv command")
    savetocsv("output.csv", row , header)


def highlightform(form):
    pass


btn = Button(master, text="Submit Me", command=clicked)
#btn.grid(column=1, row=0)
btn.pack()
use_coco = False
def toggle_coco():
    global use_coco
    use_coco = not(use_coco)
    print("here")
    print(use_coco)
    
    
    


#-----------

menubar = Menu(master)
show_all = BooleanVar()
show_all.set(True)


view_menu = Menu(menubar, tearoff=False)
view_menu.add_checkbutton(label="Toggle COCO", onvalue=False, offvalue=True, variable=show_all, command=toggle_coco)
menubar.add_cascade(label='View', menu=view_menu)
master.config(menu=menubar)



imgicon = PhotoImage(file='myicon.gif')
master.tk.call('wm', 'iconphoto', master._w, imgicon)  
tk.mainloop()
