from __future__ import print_function
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import logging
import tensorflow as tf

import cv2
import traceback
sys.path.append("../")
# Object detection imports
from utils import label_map_util
#from utils import old_visualization_utils as old_vis_util
from utils import visualization_utils as vis_util
from utils import backbone

from object_detection.utils import ops as utils_ops


def run_inference_for_single_image(image,sess):
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: image})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]


    return output_dict

from datetime import datetime
def cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled,x,y,w,h,label_to_look_for,
                                       write=True, display = False, location_of_text=(40, 100), pixel_v = 22, brighten = False):

    print("Starting")
    print(label_to_look_for)

    status = True
    last_frame_status = True
    false_in_a_row_count = 0
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    frame_count = 0
    total_passed_vehicle = 0


    # input video
    print("capturinmg vid")
    cap = cv2.VideoCapture(input_video)
    print("Video opened.")

    
    
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if write:
        output_movie = cv2.VideoWriter('the_output.mp4', fourcc, fps, (w, h))

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:


            # for all the frames that are extracted from input video
            #
            while (True):
                try:
                    status = True
                

                    # Get handles to input and output tensors
                    ops = tf.get_default_graph().get_operations()

                    tensor_dict = {}
                    for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes']:
                        tensor_name = key + ':0'

                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("vid no recieved")
                        last_frame_status = False
                        false_in_a_row_count += 1
                        status = False

                        if false_in_a_row_count >= 1:
                            print("\n\nNEED TO QUIT\n\n")
                            return total_passed_vehicle, status, frame_count , input_video
                    
                        #print("end of the video file...")
                        #break
                    else:
                        
                        print("image is solid")
                        
                        last_frame_status = True
                        false_in_a_row_count = 0
                    
                    print("READ A FRAME")
                    input_frame = frame
                    #input_frame = None

                   

                    #image throws eroor if u try to compare it to being null
                    


                    #resize img down
                    #input_frame = cv2.resize(input_frame,None,fx=0.5,fy=0.5)
                    #crop img
                    if brighten:
                        print("brightening img")
                        
                        brightness = 52
                        if brightness > 0:
                            shadow = brightness
                            highlight = 255
                        else:
                            shadow = 0
                            highlight = 255 + brightness
                        alpha_b = (highlight - shadow)/255
                        gamma_b = shadow

                        input_frame = cv2.addWeighted(input_frame, alpha_b, input_frame, 0, gamma_b)
                    input_frame = input_frame[y:y + h, x:x + w]
                    image_height, image_width, _ = input_frame.shape

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(input_frame, axis=0)

                    print("Detecting")
                    output_dict  = run_inference_for_single_image(image_np_expanded, sess)


                    output = []
                    threshold = 0.5
                    for index, score in enumerate(output_dict['detection_scores']):

                        if score < threshold:
                            continue

                        # Get data(label, xmin, ymin, xmax, ymax)
                        label = category_index[output_dict['detection_classes'][index]]['name']
                        ymin, xmin, ymax, xmax = output_dict['detection_boxes'][index]

                        output.append((label, int(xmin * image_width), int(ymin * image_height), int(xmax * image_width),
                                    int(ymax * image_height)))


                    
                    peopleboxes = []
                    for l, x_min, y_min, x_max, y_max in output:
                        if l == label_to_look_for:

                            
                            peopleboxes.append([ x_min, y_min, x_max, y_max])

                            
                            #print(image_width/2)
                            if (x_min >int(round((image_width/2) ))) and (x_min < int(round((image_width/2) )) + pixel_v):
                                if y_max < int(round(image_height/5) * 4):
                                    total_passed_vehicle += 1
                                else:
                                    print("Dont count too low")





                    #peopleboxes = nps(peopleboxes,0.3)
                    #visualization of detected people:
                    for person in peopleboxes:
                        cv2.rectangle(input_frame, (int(person[0]), int(person[1])), (int(person[2]), int(person[3])), (46,37,93), 2)






                    # insert information text to video frame
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    cv2.rectangle(input_frame, (location_of_text[0]+355, location_of_text[1]- 23), (location_of_text[0]-4, location_of_text[1]+4), (0, 0, 0), cv2.FILLED)
                    cv2.putText( input_frame,'Detected Pedestrians: ' + str(round(total_passed_vehicle)),location_of_text,font,0.8,(0, 0xFF, 0),2,cv2.FONT_HERSHEY_SIMPLEX,)

                    


                    frame_count += 1
                    if frame_count % 200 ==  0:
                        #once 200 frames have passed, save
                        time = datetime.now()
                        with open("totals.txt","a") as file:
                            file.write("\n-----\n" + "Total so far (not acounting for restarts)" + str(total_passed_vehicle) + " " + str(time) +  "\n------\n")
                    if write:
                        output_movie.write(input_frame)
                        print(" writing frame...")

                    if display:
                        cv2.imshow('object counting',input_frame)

                    #print("here")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
                except  KeyboardInterrupt:
                    print("keyboard interrupt")
                    #print(e)
                    print(traceback.format_exc())
                    return total_passed_vehicle,status, frame_count , input_video
                

                
                
            
    print("total pedestrains passed" + str(total_passed_vehicle))
    return total_passed_vehicle,status, frame_count , input_video


