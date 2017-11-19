# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains all functions related to operation of bounding box
#import __init__paths__
import numpy as np
import math, time, pdb
import util

def bboxcheck(bbox, debug=True):
    '''
    check the input to be a bounding box 

    parameter:
        bbox:   N x 4 numpy array, TLBR format
    
    return:
        True or False
    '''    
    return util.isnparray(bbox) and bbox.shape[1] == 4 and bbox.shape[0] > 0

def pts_conversion_back_bbox(pts_array, bbox, debug=True):
    '''
    convert pts in the cropped image to the pts in the original image 

    parameters:
        bbox:       1 X 4 numpy array, TLBR or TLWH format
        pts_array:  2(3) x N numpy array, N should >= 1
    '''

    if debug:
        assert is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array), 'the input points should have shape: 2 or 3 x num_pts vs %d x %s' % (pts_array.shape[0], pts_array.shape[1])
        assert bboxcheck(bbox), 'the input bounding box is not correct'
    
    pts_array[0, :] = pts_array[0, :] + bbox[0]
    pts_array[1, :] = pts_array[1, :] + bbox[1]

    return pts_array

# this function takes boxes in and process it to make them inside the image
# input format could be a 1x1 cell, which contains Nx4 
# or input boxes could be a Nx4 matrix or Nx5 matrix
# input format: TLWH (x, y)
# output format: TLWH (x, y)
def clip_bboxes_TLWH(boxes, im_width, im_height, debug_mode=True):

    # x1 >= 1 & <= im_width

    boxes[0] = max(min(boxes[0], im_width), 1);
    # y1 >= 1 & <= im_height
    boxes[1] = max(min(boxes[1], im_height), 1);
    
    # width
    boxes[2] = max(min(boxes[2], im_width - boxes[0] + 1), 1);
    # height
    boxes[3] = max(min(boxes[3], im_height - boxes[1] + 1), 1);

    return boxes