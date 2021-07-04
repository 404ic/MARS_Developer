import json
import os
import numpy as np
import random
from pprint import pprint
import pose_annotation_tools.tfrecord_util as tfrecord_util

def correct_box(xmin, xmax, ymin, ymax, stretch_const=0.04, stretch_factor=0.30, useConstant=True):
    # Code to modify bounding boxes to be a bit larger than the keypoints.

    if useConstant:
        stretch_constx = stretch_const
        stretch_consty = stretch_const
    else:
        stretch_constx = (xmax - xmin) * stretch_factor  # of the width
        stretch_consty = (ymax - ymin) * stretch_factor

    # Calculate the amount to stretch the x by.
    x_stretch = np.minimum(xmin, abs(1 - xmax))
    x_stretch = np.minimum(x_stretch, stretch_constx)

    # Calculate the amount to stretch the y by.
    y_stretch = np.minimum(ymin, abs(1 - ymax))
    y_stretch = np.minimum(y_stretch, stretch_consty)

    # Adjust the bounding box accordingly.
    xmin -= x_stretch
    xmax += x_stretch
    ymin -= y_stretch
    ymax += y_stretch
    return xmin,xmax,ymin,ymax

def format_for_tfrecord(data_path, color, annotator):
    '''
    This function is for formatting the data collected by the expert annotators found
    on the website https://data.caltech.edu/records/2011 into a form where we can convert
    into tfrecords.
    '''
    v_info = []
    D = []
    with open(os.path.join(data_path, 'MARS_raw_keypoints_top.json')) as json_file:
        D = json.load(json_file)
    random.shuffle(D)
    v_info = []
    for i in range(len(D)):
        X = D[i]['coords'][color]['x'][annotator]
        Y = D[i]['coords'][color]['y'][annotator]
        width = D[i]['width']
        height = D[i]['height']
        Bxmin = min(X) / width
        Bxmax = max(X) / width
        Bymin = min(Y) / height
        Bymax = max(Y) / height
        Bxmin, Bxmax, Bymin, Bymax = correct_box(Bxmin, Bxmax, Bymin, Bymax)
        Barea = abs(Bxmax - Bxmin) * abs(Bymax - Bymin) * width * height
        i_frame = {'filename': os.path.join(data_path, 'MARS_pose_top', D[i]['filename']),
                "class": {
                    "label": 0,
                    "text": '',
                },
                'id': format(i, '06d'),
                'width': width,
                'height': height,
                'object': {'area': Barea,
                            'bbox': {
                                'xmin': Bxmin,
                                'xmax': Bxmax,
                                'ymin': Bymin,
                                'ymax': Bymax,
                                'label': [0],
                                'count': 1}}}
        v_info.append(i_frame)
    # with open('result.json', 'w') as fp:
    #     json.dump(v_info, fp)
    return v_info