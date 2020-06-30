

#  Pedestrian-Counting-Python

This is a pedestrian detector in python. It is an app that detects how many people walk past a doorway or other place.


# Installation
To get the models into their folders:
Follow the installation instructions from those folders

 - My model: [Link](https://github.com/leomet07/Pedestrian-Counting-Python/blob/master/ssd_mobilenet_v1_coco_11_06_2017/README.md)
 - COCO model: [Link](https://github.com/leomet07/Pedestrian-Counting-Python/blob/master/coco/README.md)

Clone this repo and run withgui.py with python 3.5
Required libraries:
 1. Tensorflow
 2. Opencv
 3. Numpy
 
 Also needed
 -  TensorFlow Object Detection API(this is to run in the object_detection folder)
-   Protobuf see the T.O.D.A. object detection folder for more installation help

Just copy these files/folders into the T.O.D.A object_detection folder. If needed to replace then replace.
**Make sure that this is running in the object detection folder of the tensorflow/models/research**



# Usage
After the program boots, it should look like this:
![Normal veiw](https://raw.githubusercontent.com/leomet07/Pedestrian-Counting-Python/master/examples/normal.png)

Then type in the file name and type in the top left corner x value for the first form, the top lefty corner y value ,the width(pixels rightward) for the third form, and the height (pixels downward) for the last form.

A window should pop-up displaying your video with only the part within the coordinates being displayed.



# Models
This app has two models to run off:
A normal model (trained by me)
And a COCO model (not recommended)
To switch between these models, go to the veiw tab(shown in image) and toogle to the model of your choice
![The model toogle menu](https://raw.githubusercontent.com/leomet07/Pedestrian-Counting-Python/master/examples/menu.png)



