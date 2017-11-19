# Author: Shangxuan Wu and Xinshuo
# Email: {wushx6, xinshuo.weng}@gmail.com

# apply trained network in multi-scale way to get results
# parameter:
#   test_image:     any image to test
#   config:         please use get_config.m to obtain these configurations
#
# return:
#   heatmaps:       multichannel heatmap for all points (including background channel): height x width x (num_pts + 1)
#   pts_locations:  3 x N mat, each column represents (x, y, score) for each point.

import caffe
import numpy as np
import cv2, pdb
import util
import toolbox.util
import padding
import peak

def applyModel(test_image, config, savename):

    # test image shape is 1280 h 960 w 3 c
    im_height = test_image.shape[0]
    im_width = test_image.shape[1]

    # load model and testing network
    net = config.net

    ## load pre-defined setting
    minimum_width = config.minimum_width
    num_pts = config.num_pts
    num_levels_scale = config.num_levels_scale
    scale_starting = config.scale_starting
    scale_ending = config.scale_ending

    if config.debug_mode:
        assert scale_starting <= scale_ending, 'starting ratio should <= ending ratio'
        assert num_levels_scale >= 1, 'number of level for multi-scale shoule be larger or equal than 1'
    

    ## search thourgh a range of scales, and choose the scale according to sum of peak value of all pts
    if num_levels_scale > 1:
        # TODO get this fixed
        range_ = np.arange(np.log2(scale_starting) , ((np.log2(scale_ending) - np.log2(scale_starting)) / (num_levels_scale - 1)) , np.log2(scale_ending))
        multiplier = np.power(2, range_)
        num_scales = multiplier.size
    elif num_levels_scale == 1:
        multiplier = [(scale_starting + scale_ending) / 2.0]
        num_scales = 1
    else:
        assert False, 'error, number of level in scales should be positive integer.'
    
    #pdb.set_trace()

    ## run test image to obtain heatmap and prediction in multi-scale manner
    

    #score = cell(1, num_scales)
    score = []
    peakValue = np.zeros([num_scales, num_pts])
    #pad = cell(1, num_scales)
    pad = []
    #ori_size = cell(1, num_scales)
    ori_size = []


    for scale_index in range(num_scales):       
        # resize and pad the test image
        scale_factor = multiplier[scale_index]      # resize scale
        if (not config.evaluation_mode) and config.debug_mode:
            print('processing multi-scale prediction, current scale factor is %.3f\n', scale_factor)
        
        image_resized = toolbox.util.resize_portion(test_image, scale_factor)

        ori_size.append(image_resized.shape)
        minimum_resolution = [minimum_width, max(ori_size[scale_index][1], minimum_width)]      # keep the image resolution at least boxsize x boxsize
        
        [imageToTest, tmp_pad] = padding.padHeight(image_resized, config.pad_value, minimum_resolution)
        pad.append(tmp_pad)

        ## resize image to smaller scale to fit caffe restriction
        while imageToTest.shape[0] * imageToTest.shape[1] > config.INT_MAX_CAFFE:
            imageToTest = cv2.resize(imageToTest, 0.95)
        
        # if config.subset == 'mouth':
        #     cv2.imwrite('aaa.png', imageToTest)
        #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~saved~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        # input imageToTest.shape = 1280 h 960 w 3 c
        imageToTest = toolbox.util.preprocess_image_caffe(imageToTest, config.mean_value, False)
        # output imageToTest.shape = 

        ## get heatmap output and re-scale it

        # should be 3 c x 1280 h x 960 w

        net.blobs['data'].reshape(1, imageToTest.shape[0], imageToTest.shape[1], imageToTest.shape[2])
        #pdb.set_trace()
        #net.reshape()
        now_score = applyDNN(imageToTest, net, config)

        score.append(now_score)  # score has already been resized to the size of input image in caffe
        #score[scale_index] = result
        # 

        #score[scale_index] = resizeIntoScaledImg(score[scale_index], pad[scale_index])     # re-scale the score heatmap to the size of original image
        #score[scale_index] = cv2.resize(score[scale_index], (im_width,im_height))            # 712 x 674 x 57

    ## generate heatmap
    if config.heatmap_merge == 'avg':
        heatmaps = np.zeros(score[0].shape)
        for scale_index in range(len(multiplier)):
            heatmaps = heatmaps + score[scale_index]       # add all scores from multi-scale images
        heatmaps = heatmaps / len(multiplier)
    
    
    # warm up
    if (not config.evaluation_mode) and config.debug_mode:
        print('\n\nget output landmark locations from heatmaps.......................\n\n')
    

    ## clustering warmup, apply for multi-person, doesn't apply here for only single person
    '''if config.pts_merge_peak == 'clustering':
        score_pts = score{1}(:, :, 1)                                      # first pts in first scale
        threshold = max(score_pts(:)) * config.heatmap_smooth_percentage
        [X, Y, s] = find_peaks(score_pts, threshold, false, false)  
        peak_data = [X' Y']
        [clust_center, ~, ~] = MeanShiftCluster(peak_data, config.cluster_bandwidth, 0)     # center in each cluster
        num_people = size(clust_center, 2)
        assert num_people == 1, 'current version only support single person testing.'
        if num_people != 1:
            fprintf('number of people should be 1 vs %d.\n', num_people)
            visualize_heatmap(test_image, score{1}(:, :, 1), false)
            hold on
            plot(clust_center(1, :), clust_center(2, :), 'rx', 'MarkerSize', 10)
    '''    
    

    ## get pts_location for all pts and scales, only one person right now
    pts_locations = np.zeros([3, num_pts])       # the third row is the confidence
    for pts_index in range(num_pts):
        if (not config.evaluation_mode) and config.debug_mode:
            print('processing point %d/%d\n' % (pts_index, num_pts))

        pts_data = np.zeros([num_scales, 4])
        for scale_index in range(num_scales):     # different scale
            score_pts = score[scale_index][:, :, pts_index]        # use only one pts to find peaks

            threshold = np.max(score_pts) * config.heatmap_smooth_percentage 

            # choose the best single points from the heatmap
            if config.pts_merge_peak =='max':
                [X, Y, s] = peak.find_peaks(score_pts, threshold)        # hope 1 x 1
                if len(s) > 1:
                    best_peak_index = np.argmax(s)
                    X = [X[best_peak_index]]
                    Y = [Y[best_peak_index]]
                    s = [s[best_peak_index]]
                

                if len(s) == 1:
                    pts_data[scale_index, :] = [X[0], Y[0], s[0], multiplier[scale_index]]
                else:                
                    pts_data[scale_index, :] = [0, 0, 0, multiplier[scale_index]]          # handle the edge case where the point is not detected

            elif config.pts_merge_peak == 'weighted':
                # for debug
                # if pts_index == 20
                    # config.vis = true
                # end
                # TODO: add weighted_centroid function
                centroids = weighted_centroid(score_pts, threshold, false, config.vis)
                x = centroids[0, :]
                y = centroids[1, :]
                score_tmp = diag(score_pts(uint16(y), uint16(x)))
                [max_score, max_center_index] = max(score_tmp)         # to compare different peak candidate, we use nearest score but could be improved by using interp
                pts_data[scale_index, :] = [x[max_center_index], y[max_center_index], max_score, multiplier[scale_index]]

            else:
                assert false, 'no method for peak finding.\n'        

        # analysis peaks
        if config.pts_merge_scale == 'max':
            best_scale = np.argmax(pts_data[:, 2])                         # take the best scale index
            best_scalefactor = pts_data[best_scale, 3]
            pts_locations[0:3, pts_index] = pts_data[best_scale, 0:3] # this means transpose

    return heatmaps, pts_locations 

# do forward pass to get scores, specific to only this project
# the output_stage demotes where we want to extract the output, should be >=1
def applyDNN(images, net, config):
    output_stage = config.output_stage  
    if config.debug_mode:
        assert (output_stage > 0 and output_stage <= 6 and util.isInteger(output_stage)), 'output_stage number should be [1, 6]'
    
    #input_data = {single(images)}
    #s_vec = net.forward(input_data)    # scores are now Width x Height x Channels x Num

    # images should be in the size of [1, 21, 160, 120]
    net.blobs['data'].data[0,:,:,:] = images

    s_vec = net.forward()    # scores are now Width x Height x Channels x Num
    


    if output_stage == 1:
        if config.deconv:
            scores = net.blobs['deconv5_6_CPM'].data
        else:
            scores = net.blobs['conv5_5_CPM'].data
        
    elif output_stage >= 2:    
        if config.deconv:
            scores = net.blobs[('Mdeconv%d_stage' % (config.model.stage_depth + 1)) + str(output_stage)].data
        else:
            scores = net.blobs[('Mconv%d_stage' % config.model.stage_depth) + str(output_stage)].data
    

    scores = np.transpose(scores[0,:,:,:], [1,2,0])
    # h w c 160 120 21

    # upsample it by 8 times
    scores_ = toolbox.util.resize_portion(scores, config.downsample)           # use bicubic upsampling

    return scores_


'''
# Some problem with this function
def resizeIntoScaledImg(score, pad):
    pdb.set_trace()
    # score is something like [960 1280 27]
    # pad is something like [1 4]
    
    # transferring to numpy things
    score = np.array(score)
    pad = np.array(pad)

    np_ = score.shape[2]-1
    score = np.transpose(score, [1,0,2])
    if(pad[0] < 0):
        padup = np.concatenate(np.zeros(-pad[1], score.shape[1], np_), np.ones(-pad[0], score.shape[1], 1), axis=2)
        score = [padup,score] # pad up
    else:
        score = score[pad[0]:,:,:] # crop up
    
    
    if(pad[1] < 0):
        padleft = np.concatenate(np.zeros(score.shape[0], -pad[1], np_), np.ones(score.shape[0], -pad[1], 1), axis=2)
        score = [padleft,score] # pad left
    else:
        score = score[:,:-1-pad[1],:] # crop left
    
    
    if(pad[2] < 0):
        paddown = np.concatenate(np.zeros(-pad[2]   , score.shape[1], np_), np.ones(-pad[2], score.shape[1], 1), axis=2)
        score = [score,paddown] # pad down
    else:
        score = score[:-1-pad[2], :, :] # crop down
    
    
    if(pad[3] < 0):
        padright = np.concatenate(np.zeros(score.shape[0], -pad[3], np_), np.ones(score.shape[0], -pad[3], 1), axis=2)
        score = [score,padright] # pad right
    else:
        score = score[:,-1-pad[3]:, :] # crop right
    
    score = np.transpose(score, [1,0,2])
    return score'''

if __name__ == "__main__":
    pass