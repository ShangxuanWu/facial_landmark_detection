# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this function is to crop the image around a specific center with padded value around the empty area, all images in this function are floating images
# parameters:
#   img:        a floating image
#   rect:       an array, which defines how to crop the image around center
#                   1. rect with WH format, then crop around the center of image
#                   2. rect with XYWH format, then crop around (X, Y) with given height and width
# Note that if the cropped region is out of boundary, we pad gray value around outside
# Note that the cropping is right aligned, which means, if the crop width or height is even, we crop one more pixel right to the center

import math, pdb, cv2
import toolbox.util
import toolbox.bbox
import numpy as np


# this function is to pad given value to an image in provided region, all images in this function are floating images
# parameters:
#   img:        a floating image
#   pad_rect:   4 element array, which describes where to pad the value. The order is [left, top, right, bottom]
#   pad_value:  a scalar defines what value we should pad
def pad_around(img, pad_rect, pad_value):

    [im_height, im_width, channel] = img.shape
    
    # calculate the new size of image
    pad_left    = pad_rect[0];
    pad_top     = pad_rect[1];
    pad_right   = pad_rect[2];
    pad_bottom  = pad_rect[3];
    new_height  = im_height + pad_top + pad_bottom;
    new_width   = im_width + pad_left + pad_right;
    
    # pad
    padded = np.zeros([new_height, new_width, channel]);
    padded[:] = pad_value;
    padded[pad_top: new_height-pad_bottom, pad_left: new_width-pad_right, :] = img;
    return padded

def crop_center(img1, rect, pad_value):
# rect is XYWH

    if not pad_value:
        pad_value = 0.5

    # calculate crop rectangles
    [im_height, im_width, im_channel] = img1.shape
    im_size = [im_height, im_width]

    rect = [int(x) for x in rect]
    if len(rect) == 4:            # crop around the given center and width and height
        center_x = rect[0]
        center_y = rect[1]
        crop_width = rect[2]
        crop_height = rect[3]
    else:                            # crop around the center of the image
        center_x = math.ceil(im_width/2)
        center_y = math.ceil(im_height/2)   
        crop_width = rect[0]
        crop_height = rect[1]

    # calculate cropped region
    xmin = int(center_x - math.ceil(crop_width/2) + 1) -1
    ymin = int(center_y - math.ceil(crop_height/2) + 1) -1
    xmax = int(xmin + crop_width - 1)
    ymax = int(ymin + crop_height - 1)
    
    crop_rect = [xmin, ymin, crop_width - 1, crop_height - 1]
    
    # if original image is not enough to cover the crop area, we pad value around outside after cropping
    if (xmin < 0 or ymin < 0 or xmax > im_width-1 or ymax > im_height-1):
        #pad_left    = max(1 - xmin, 0)
        #pad_top     = max(1 - ymin, 0)
        #pad_right   = max(xmax - im_width, 0)
        #pad_bottom  = max(ymax - im_height, 0)
        #pad_rect    = [pad_left, pad_top, pad_right, pad_bottom]

        # padding
        #cropped = pad_around(cropped, pad_rect, pad_value)

        x_diff = crop_width - img1.shape[1]
        y_diff = crop_height - img1.shape[0]

        cropped_padded = np.ones((crop_height, crop_width, img1.shape[2])) * pad_value
        cropped_padded[0:im_height,0:im_width,:] = img1
        cropped = cropped_padded

    else:
        # crop if everything is fine
        cropped = img1[ymin:(ymin + crop_height), xmin:(xmin+crop_width)]

    # TODO: with padding
    [im_height, im_width, im_channel] = img1.shape
    
    crop_rect_ori = toolbox.bbox.clip_bboxes_TLWH(crop_rect, im_width, im_height)
    return cropped, crop_rect, crop_rect_ori

if __name__ == "__main__":
    pass