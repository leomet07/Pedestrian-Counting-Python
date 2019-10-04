from __future__ import print_function
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import logging
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import cv2
sys.path.append("../")
# Object detection imports
from utils import label_map_util
#from utils import old_visualization_utils as old_vis_util
from utils import visualization_utils as vis_util
from utils import backbone

from object_detection.utils import ops as utils_ops
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)
# Just disables the warning shown at the end, doesn't enable AVX/FMA(Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# This is needed since the notebook is stored in the object_detection folder.
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


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
from alg import nps

def cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled,x,y,w,h,label_to_look_for,
                                       write=True, display = False, location_of_text=(40, 100), pixel_v = 12):

    print("Starting")
    print(label_to_look_for)

    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    i = 0
    total_passed_vehicle = 0


    # input video
    cap = cv2.VideoCapture(input_video)
    print("Video opened.")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if write:
        output_movie = cv2.VideoWriter('the_output.mp4', fourcc, fps, (w, h))

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:


            # for all the frames that are extracted from input video
            while (cap.isOpened()):

                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()

                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes']:
                    tensor_name = key + ':0'

                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break

                input_frame = frame
                #crop img
                input_frame = input_frame[y:y + h, x:x + w]
                image_height, image_width, _ = input_frame.shape

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)


                output_dict  = run_inference_for_single_image(image_np_expanded, sess)


                output = []
                threshold = 0.6
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

                        i += 1
                        peopleboxes.append([ x_min, y_min, x_max, y_max])

                        '''
                        print(x_min , image_width/2)
                        if (x_min >int(round(image_width/3 ))) and (x_min < int(round(image_width/3)) + 6):
                            total_passed_vehicle += 0.5

                        if (x_min >int(round((image_width/3) * 2 ))) and (x_min < int(round((image_width/2) * 2)) + 6):
                            total_passed_vehicle += 0.5
                        '''
                        '''
                        if x_min > 200 and x_min < 217 and y_min < 350:
                            total_passed_vehicle +=0.5
                        
                        if x_min > 350 and x_min < 370 and y_min < 350:
                            total_passed_vehicle +=0.5
                        '''
                        #print(image_width/2)
                        if (x_min >int(round((image_width/2) ))) and (x_min < int(round((image_width/2) )) + pixel_v):
                            total_passed_vehicle += 1





                #peopleboxes = nps(peopleboxes,0.3)
                #visualization of detected people:
                for person in peopleboxes:
                    cv2.rectangle(input_frame, (int(person[0]), int(person[1])), (int(person[2]), int(person[3])), (46,37,93), 2)






                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.rectangle(input_frame, (location_of_text[0]+355, location_of_text[1]- 23), (location_of_text[0]-4, location_of_text[1]+4), (0, 0, 0), cv2.FILLED)
                cv2.putText( input_frame,'Detected Pedestrians: ' + str(round(total_passed_vehicle)),location_of_text,font,0.8,(0, 0xFF, 0),2,cv2.FONT_HERSHEY_SIMPLEX,)

                if write:
                    output_movie.write(input_frame)
                    print(" writing frame...")

                if display:
                    cv2.imshow('object counting',input_frame)



                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
    print("total pedestrains passed" + str(total_passed_vehicle))
    return total_passed_vehicle



