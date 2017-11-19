import numpy as np
import math, pdb

def padHeight(img, padValue, bbox):
    

    pad = [0,0,0,0]
    h = img.shape[0]
    w = img.shape[1]
    h = min(bbox[0],h)
    bbox[0] = int(math.ceil(bbox[0]/8)*8)
    bbox[1] = int(max(bbox[1], w))
    bbox[1] = int(math.ceil(bbox[1]/8)*8)
    pad[0] = int(math.floor((bbox[0]-h)/2)) # up
    pad[1] = int(math.floor((bbox[1]-w)/2)) # left
    pad[2] = int(bbox[0]-h-pad[0]) # down
    pad[3] = int(bbox[1]-w-pad[1]) # right

    # TODO there is some problems here
    if pad == [0,0,0,0]:
        return img, pad

    img_padded = img
    pad_up = np.tile(img_padded[0,:,:], (pad[0],1,1))*0 + padValue
    img_padded = np.concatenate((pad_up,img_padded))
    pad_left = np.tile(img_padded[:,0,:], (1,pad[1],1))*0 + padValue
    img_padded = np.concatenate((pad_left,img_padded), axis=1)
    pad_down = np.tile(img_padded[-1,:,:], (pad[2],1,1))*0 + padValue
    img_padded = np.concatenate((img_padded,pad_down))
    pad_right = np.tile(img_padded[:,-1,:], (1,pad[3],1))*0 + padValue
    img_padded = np.concatenate((img_padded,pad_right), axis=1)
    #cropping if needed
    return img_padded, pad