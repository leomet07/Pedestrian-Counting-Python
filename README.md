# Pedestrian-Counting-Python

This is a pedestrian detector in python. It is an app that detects how many people walk past a doorway or other place.

# Installation

Install the Tensorflow Object Detetion API

1 `git clone https://github.com/tensorflow/models`

cd into research/. Downlaod my custom library and rename it to "object_detection" and override the old research/object_detection folder.

2 `git clone https://github.com/leomet07/tensorflow-object-detection-api-custom`

Install the main code in the research/ dir

3 `git clone https://github.com/leomet07/Pedestrian-Counting-Python`

Run withgui.py with python 3.5+ < 3.7

Required libraries:

1. Tensorflow

2. Opencv2

3. Numpy

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
