# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# find peak locations and scores for convex blob in heatmap
# note that this function is very strict. No smooth for peak points have been applied
# if there are multiple local peak in a same blob, all of them will be returned. 

import numpy as np
import pdb

def find_peaks(heatmap, thre):
    #filter = fspecial('gaussian', [3 3], 2)
    #map_smooth = conv2(map, filter, 'same')
    
    # variable initialization    

    map_smooth = np.array(heatmap)
    map_smooth[map_smooth < thre] = 0.0


    map_aug = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    map_aug1 = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    map_aug2 = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    map_aug3 = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    map_aug4 = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    
    # shift in different directions to find peak, only works for convex blob
    map_aug[1:-1, 1:-1] = map_smooth
    map_aug1[1:-1, 0:-2] = map_smooth
    map_aug2[1:-1, 2:] = map_smooth
    map_aug3[0:-2, 1:-1] = map_smooth
    map_aug4[2:, 2:] = map_smooth

    peakMap = np.multiply(np.multiply(np.multiply((map_aug > map_aug1),(map_aug > map_aug2)),(map_aug > map_aug3)),(map_aug > map_aug4))

    peakMap = peakMap[1:-1, 1:-1]

    idx_tuple = np.nonzero(peakMap)     # find 1
    Y = idx_tuple[0]
    X = idx_tuple[1]

    score = np.zeros([len(Y),1])
    for i in range(len(Y)):
        score[i] = heatmap[Y[i], X[i]]

    return X, Y, score