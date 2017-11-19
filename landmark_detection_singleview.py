# Author: Shangxuan Wu and Xinshuo Weng
# Email: {wushx6, xinshuo.weng}@gmail.com

# this function run model on multiple dataset and get the prediction in json format file for further evaluation
# note that this function is very specific to our evalution_all.py and data storage format

# resize_factor,            scale to 5120 x 3840
# vis_resize_factor,        relative to 5120 x 3840

import time, pdb, os, cv2
import numpy as np
import toolbox.bbox
import toolbox.util
import toolbox.save_matrix2d
import config_helper
import saving
import utils.MUGSY_helper
import cropping, apply_model

def process(data_dir, save_dir, subset, prefix, resize_factor, gpu_id, rotation_correct, gopro, start_frame, _frame, vis_resize_factor, camera_id):
    # camera_id = '330030'

    # data_dir = '/media/xinshuo/disk2/datasets/MUGSY/multiview/20170222_002576793_dots--neutral_tongue_rom/original'
    # save_dir = '/media/xinshuo/disk2/datasets/MUGSY/multiview/20170222_002576793_dots--neutral_tongue_rom'
    # subset = 'mouth'
    # prefix = 'left'
    # resize_factor = 4
    # gpu_id = 0
    # rotation_correct = 0
    # gopro = 0
    # start_frame = 0
    # _frame = 299
    # filter_folder = {'001_351_20160620', '002_11_20160620', '003_14_20160620', '004_17_20160620', '005_19_20160620', '006_21_20160620', '007_26_20160620', '008_28_20160620', '009_30_20160620'}
    # filter_name = {'E4'}

    assert toolbox.util.is_path_exists(data_dir), 'the input data directory does not exist at %s' % data_dir
    toolbox.util.mkdir_if_missing(save_dir)

    ## read configuration
    if subset == 'face_old' or subset == 'face' or subset == 'ears' or subset == 'nose':
        size_set = 'resized_4'
    elif subset == 'mouth':
        size_set = 'cropped_all'
    else:
        size_set = 'cropped'
    
    datasets = 'MUGSY'

    deploy = True
    config = config_helper.get_config('test', datasets, subset, size_set, 'real', deploy, gpu_id)

    # from compact subset naming to detailed
    if subset == 'lower_eyelid' or subset == 'upper_eyelid' or subset =='ears':
        subset = ('%s_%s'% ( prefix, subset))
    

    if 'cropped' in size_set:
        chopping_dir = toolbox.util.fullfile(save_dir, 'chopping/predictions', 'cam%s' % camera_id)
    
    
    if(subset == 'face_old'):
        output_folder = toolbox.util.fullfile(save_dir, 'chopping')
    else:
        output_folder = toolbox.util.fullfile(save_dir, subset)
    

    # mode setting
    if(rotation_correct == 1):
        config.rotation_correct = True
    elif(rotation_correct == 0):
        config.rotation_correct = False
    else:
        assert False, 'rotation correction is wrong.'
    
        
    if gopro == 1:
        config.gopro = True
    elif gopro == 0:
        config.gopro = False
    else:
        assert False, 'gopro is wrong.'
    

    config.deploy = deploy
    config.vis_intermediate = True  # save the intermediate visualization (heatmap attached on cropped images)
    config.vis_back_only = True     # save the only intermediate visualization with background channel
    config.resume_mode = False
    config.vis = False
    config.time = True
    config.debug_mode = False
    # config.save_full = False          # save the visualization on full resolution image
    config.save_heatmap = False         # save the pure heatmaps
    config.save_cropped = True          # save the cropped images
    config.vis_final = True         # save the prediction on full resolution image
    config.force = False                # redo the finished ones
    config.save_local = True            # save single prediction file
    config.label = False                # attach label on top of visualization
    config.enlarge_crop = 1.0

    if config.vis or config.vis_intermediate or config.debug_mode or config.vis_final:
        vis_folder = toolbox.util.fullfile(output_folder, 'visualization', 'cam%s'% camera_id)
        toolbox.util.mkdir_if_missing(vis_folder)
    

    if not vis_resize_factor:
        config.vis_resize_factor = 1
    else:
        config.vis_resize_factor = vis_resize_factor
    

    # save the configuration and model information for reference
    config_dir = toolbox.util.fullfile(save_dir, 'configurations')
    toolbox.util.mkdir_if_missing(config_dir)
    config_savepath = toolbox.util.fullfile(config_dir, '%s_configurations.txt' % subset)
    
    saving.save_struct(config, config_savepath, config.debug_mode)
    model_savepath = toolbox.util.fullfile(config_dir, '%s_models.txt' % subset)
    saving.save_struct(config.model, model_savepath, config.debug_mode)
    #config_savepath = toolbox.util.fullfile(config_dir, '%s_config.mat' % subset)
    #save(config_savepath, 'config')

    # set path for saving visualization images
    if (config.vis or config.vis_intermediate or config.vis_final):
        if config.vis_final:
            vis_final_dir = toolbox.util.fullfile(vis_folder, 'images_with_pts')
            toolbox.util.mkdir_if_missing(vis_final_dir)
        
        
        if config.vis_intermediate:
            vis_intermediate_dir = toolbox.util.fullfile(vis_folder, 'intermediate')
            toolbox.util.mkdir_if_missing(vis_intermediate_dir)
            save_intermediate_folder_array = []
            
            if not config.vis_back_only:        # only visualize background
                for pts_index_save in range(config.num_pts):
                    
                    save_intermediate_folder_array.append(toolbox.util.fullfile(vis_intermediate_dir, 'heatmap_index%10d' % pts_index_save))
                    toolbox.util.mkdir_if_missing(save_intermediate_folder_array[pts_index_save])
                    
            
            save_intermediate_folder_array.append(toolbox.util.fullfile(vis_intermediate_dir, 'heatmap_background'))
            toolbox.util.mkdir_if_missing(save_intermediate_folder_array[-1])
        
    

    # set path for saving heatmaps
    if config.save_heatmap: 
        vis_heatmap_dir = toolbox.util.fullfile(vis_folder, 'heatmap')
        toolbox.util.mkdir_if_missing(vis_heatmap_dir)

        save_heatmap_folder_array = []

        for pts_index_save in range(config.num_pts):
            save_heatmap_folder_array.append(toolbox.util.fullfile(vis_heatmap_dir, 'heatmap_index%10d' % pts_index_save))
            toolbox.util.mkdir_if_missing(save_heatmap_folder_array[pts_index_save])
            
        # 
        save_heatmap_folder_array.append(toolbox.util.fullfile(vis_heatmap_dir, 'heatmap_background'))
        toolbox.util.mkdir_if_missing(save_heatmap_folder_array[config.num_pts])
    

    if config.save_cropped and (size_set == 'cropped' or size_set == 'cropped_all'):
        vis_cropped_dir = toolbox.util.fullfile(vis_folder, 'cropped')
        toolbox.util.mkdir_if_missing(vis_cropped_dir)
    

    print('configuration is:\n')
    print(config.__dict__)
    print('model is:\n')
    print(config.model.__dict__)

    ## load testing image
    print('############################################### loading data ###############################################\n\n')
    print('loading data.....\n')

    imagelist_savepath = toolbox.util.fullfile(save_dir, 'imagelist_test.txt')
    if toolbox.util.is_path_exists(imagelist_savepath):
        [imagelist, num_images] = toolbox.util.load_list_from_file(imagelist_savepath)
    else:           
        [imagelist, num_images] = toolbox.util.load_list_from_folder(data_dir, ['jpeg', 'png', 'jpg'])
    
    print('number of testing images at path %s is %d\n\n', data_dir, num_images)

    if not start_frame:
        start_frame = 0
    else:
        start_frame = max(0, start_frame)
    
    if not _frame:
        _frame = num_images-1
    else:
        _frame = min(_frame, num_images-1)
    

    # convert from 0-indexed to 1-indexed
    start_frame = start_frame + 1
    _frame = _frame + 1
    
    # here is the starting and ending frames
    imagelist = imagelist[start_frame-1:_frame]
    num_images = len(imagelist)
    print('number of requested images is from frame %d to %d\n\n', start_frame, _frame)

    # running for all images
    print('############################################### start running ###############################################\n\n')

    if config.save_local:
        pts_save_dir = toolbox.util.fullfile(output_folder, 'predictions', 'cam%s' % camera_id)
    
    count_failed = 0            # failed due to bad chopping
    count_skipped = 0           # skipped due to lack of input image
    count_prefinished = 0       # skipped due to the pre-finishment
    count_ignored = 0           # ignore due to bad camera 
    count_filtered = 0          # ignore due to filter
    t = time.time()

    for i in range(num_images):
        image_tmp_path = imagelist[i]
        # [detect_check, camera_id] = is_detect(image_tmp_path, subset, config.debug_mode)
        detect_check = True
        if not detect_check:
            fprintf('skipped image from this camera at %s\n', image_tmp_path)
            count_ignored = count_ignored + 1
            continue    
        

        [parent_dir, filename, ext] = toolbox.util.fileparts(image_tmp_path)
        if 'filter_name' in globals():
            detect_check = False
            for filter_index in len(filter_name):
                filter_tmp = filter_name[filter_index]
                if filter_tmp == filename:
                    detect_check = True
                
            
            if not detect_check:
                print('skipped image due to filtering name at %s\n', image_tmp_path)
                count_filtered = count_filtered + 1
                continue
            
        

        [subdir, valid, tmp1, tmp2] = toolbox.util.remove_str_from_str(parent_dir, data_dir, config.debug_mode)

        assert valid, 'the path is not correct at %s' % subdir
        if len(subdir) > 1:
            subdir = subdir[1:]         # remove slash
        print('subdir is %s\n' % subdir)

        if 'filter_folder' in globals():
            detect_check = False
            for filter_index in range(len(filter_folder)):
                filter_tmp = filter_folder[filter_index]
                # filter_tmp
                # subdir
                if findstr(filter_tmp, subdir):
                    detect_check = True
                
            
            if not detect_check:
                print('skipped image due to filtering at %s\n', image_tmp_path)
                count_filtered = count_filtered + 1
                continue

        if config.save_local: 
            save_pts_dir = toolbox.util.fullfile(pts_save_dir, subdir)
            toolbox.util.mkdir_if_missing(save_pts_dir)
            save_path_pts = toolbox.util.fullfile(save_pts_dir, '%s.pts' % filename)
            
            if toolbox.util.file_exist(save_path_pts) and not config.force:
                print('skipped %s\n', save_path_pts)
                count_prefinished = count_prefinished + 1
                continue
            
        

        # ignore the case where image does not exist
        if not os.path.isfile(image_tmp_path):
            print('skipped %s\n' % image_tmp_path)
            count_skipped = count_skipped + 1
            continue                
        

        # count the time
        elapsed = time.time()
        remaining_str = str((elapsed-t) / (i+1) * (num_images - i -1))
        elapsed_str = str(elapsed)
        
        print('Running part: %d/%d, %s, %s, %s, model: %s, path: %s, EP: %s, ETA: %s\n', i, num_images, config.datasets, config.subset, size_set, config.model_name, filename, elapsed_str, remaining_str)

        # load the images
        # image_tmp is 960 x 1280 x 3
        #               h      w    c
        # this is already bgr, 0-255
        image_tmp = cv2.imread(image_tmp_path).astype(float)
        
        # rotate and resize the original image if needed
        if config.rotation_correct:
            rotate_degree = utils.MUGSY_helper.get_rotate_degree(camera_id, True)
            image_tmp = toolbox.util.rotate_bound(image_tmp, rotate_degree)

            # now image_tmp shape is 1280 x 960 x 3
            #                          h     w    c

        image_ori = image_tmp
        if config.gopro:
            if config.vis:
                #figure imshow(image_tmp) pause
                pass
            
        
            crop_init_gopro = [400, 760, 2700, 3600]
            resize_factor = 3840.0 / 2700
            image_tmp = imcrop(image_tmp, crop_init_gopro)

            if(config.vis):
                #figure imshow(image_tmp) pause
                pass
            
        

        if 'resized_' in size_set:
            #height, width = image_tmp.shape[:2]
            #image_tmp = cv2.resize(image_tmp,(resize_factor/4*width, resize_factor/4*height), interpolation = cv2.INTER_CUBIC)

            image_tmp = toolbox.util.resize_portion(image_tmp, resize_factor/4)
            #image_tmp = cv2.resize(image_tmp, resize_factor/4)         # downsample to 1280 x 960
            # size(image_tmp)
            # imshow(image_tmp) pause(0)
        else:
            image_tmp = toolbox.util.resize_portion(image_tmp, resize_factor)               # back to original resolution

            # chopping file
            pts_file_crop = toolbox.util.fullfile(chopping_dir, subdir, '%s.pts' % filename)
            while not toolbox.util.file_exist(pts_file_crop):
                print('results from chopping module does not exist at %s\n', pts_file_crop)
                time.sleep(10)
            
            [chop_data, num_rows] = toolbox.save_matrix2d.parse_matrix_file(pts_file_crop)

            assert num_rows == 3, 'number of rows is not correct'

            valid_index_list = utils.MUGSY_helper.get_part_index_from_chopping(subset, config.debug_mode)
            #valid_index_list = cell(valid_index_list)
            #valid_index_list = cellfun(@(x) x+1, valid_index_list, 'UniformOutput', False)                 # careful +1 from python to matlab
            #valid_index_list = cell2mat(valid_index_list)

            chop_data = chop_data[:, valid_index_list]
            pts_keep_index = np.where(chop_data[2, :] > config.visualization_threshold)
            chop_data = chop_data[0:2, pts_keep_index]
            
            if config.gopro:
                #[python_boxes, ~] = convert_to_numpy_array(chop_data)
                #python_boxes = py.bbox_transform.pts_conversion_bbox(python_boxes, convert_to_numpy_array(crop_init_gopro), config.debug_mode)
                #chop_data = get_numpy_array_from_python(python_boxes)
                python_boxes = chop_data

            
            
            # chop_data = [450, 470, 490, 430 560, 570, 580, 550 1, 1, 1, 1]
            # pts_keep_index = [1,2,3,4]

            # the chopping module is working on 4-downsampled images
            # this is for 1-based matlab code
            # chop_data[0:2, :] = chop_data[0:2, :] * resize_factor

            # the chopping module is working on 4-downsampled images
            # this is for 0-based python code
            chop_data[0:2, :] = (chop_data[0:2, :]+1) * resize_factor - 1

            # only at least 2 points could determine the toolbox.bbox, otherwise we use the center of the image as the center
            if pts_keep_index[0].shape[0] >= 2:
                center_x = np.mean(chop_data[0, :])
                center_y = np.mean(chop_data[1, :])
                [image_tmp, bbox_crop_tmp, crop_rect_ori_tmp] = cropping.crop_center(image_tmp, [center_x, center_y, 2048, 2048], 0.5)
                # lowe_eyelid here should be 2048, 2048, 3
            else:
                print('no enough points detected from chopping\n')
                count_failed = count_failed + 1
                continue
            
        
        # imshow(image_tmp) pause
        
        image_tmp = toolbox.util.resize_portion(image_tmp, config.enlarge_crop*config.resize_factor)

        #image_tmp = cv2.resize(image_tmp, config.enlarge_crop*config.resize_factor)
        # figure imshow(image_tmp) pause

        
        actual_input_size = image_tmp.shape[:2]

        if(config.time):
            image_time = time.time()
            print('time spent on preprocessing images is %f seconds\n', (image_time - elapsed))
        
        
        if(config.vis and config.debug_mode):
            #figure(1) imshow(image_tmp) title('original image.')
            pass
        

        # crop the toolbox.bbox
        if(config.fit_testsize):
            [image_tmp_fitted, crop_ori, crop_rect] = cropping.crop_center(image_tmp, [config.fit_width, config.fit_height], 0.5)

            actual_input_size = image_tmp_fitted.shape[:2]
            # actual_input_size
            test_image = image_tmp_fitted

            # test image shape is 1280 h 960 w 3 c
            [heatmaps, pts_locations_fitted] = apply_model.applyModel(test_image, config, filename)         

            pts_heatmap = pts_locations_fitted

            # convert for crop center and fit size
            #[python_boxes, ~] = convert_to_numpy_array(pts_locations_fitted)
            python_boxes = pts_locations_fitted
            python_boxes = toolbox.bbox.pts_conversion_back_bbox(python_boxes, crop_ori, config.debug_mode)
            pts_locations = python_boxes    

            if(config.vis and config.debug_mode):
                '''figure(4) title('visualizing original image predictions.') visualize_image_with_pts(image_tmp, pts_locations(1:2, :), config.vis, config.debug_mode) pause(0.5)'''
            
        else:
            actual_input_size = size(image_tmp)
            test_image = image_tmp
            # figure imshow(test_image) pause
            [heatmaps, pts_locations] = applyModel(test_image, config, filename)
        

        # compensate resize factor
        pts_locations[0:2,:] = pts_locations[0:2, :] / (config.resize_factor * config.enlarge_crop)

        if(config.time):
            model_time = time.time()
            print('time spent on model is #f seconds\n', model_time - image_time)
        

        # convert the point location due to the chopping
        if(size_set == 'cropped' or size_set =='cropped_all'):
            #[python_boxes, ~] = convert_to_numpy_array(pts_locations)
            python_boxes = pts_locations
            python_boxes = toolbox.bbox.pts_conversion_back_bbox(python_boxes, bbox_crop_tmp, config.debug_mode)
            pts_locations = python_boxes   

            pts_locations[0:2, :] = pts_locations[0:2, :] / resize_factor           # back to input resolution
        else:
            pts_locations[0:2, :] = pts_locations[0:2, :] * 4 / resize_factor           # back to input resolution
        

        # convert for pre-cropping
        '''if config.gopro:
            python_boxes = pts_locations
            python_boxes = toolbox.bbox.pts_conversion_back_bbox(python_boxes, convert_to_numpy_array(crop_init_gopro), config.debug_mode)
            pts_locations = get_numpy_array_from_python(python_boxes)   
        '''

        if config.save_cropped and (size_set == 'cropped' or size_set == 'cropped_all'):
            cropped_savepath = toolbox.util.fullfile(vis_cropped_dir, subdir, (filename + '.png'))
            toolbox.util.mkdir_if_missing(toolbox.util.fileparts(cropped_savepath)[0])
            cv2.imwrite(cropped_savepath, test_image)
        
        if config.time:
            crop_time = time.time()
            print('time spent on saving cropped images is %f seconds\n' % (crop_time - model_time))
        

        if config.save_heatmap:
            for pts_index in range(config.num_pts+1):
                heatmap_tmp = heatmaps[:, :, pts_index]
                heatmap_savepath_tmp = toolbox.util.fullfile(save_heatmap_folder_array[pts_index], subdir, filename +'.png')
                toolbox.util.mkdir_if_missing(toolbox.util.fileparts(heatmap_savepath_tmp))
                cv2.imwrite(heatmap_savepath_tmp, heatmap_tmp)
            
        if config.time:
            heatmap_time = time.time()
            print('time spent on saving heatmap is %f seconds\n' % (heatmap_time - crop_time))
        

        if config.save_local:
            # original image is full resolution
            [nrows, ncols] = toolbox.save_matrix2d.save_matrix2d_to_file(pts_locations, save_path_pts)
            print('save pts file to %s\n', save_path_pts)
        

        if config.time:
            local_time = time.time()
            print('time spent on saving local prediction result file is #f seconds\n', local_time - heatmap_time)

        if config.vis_final:
            # following is the visualize_image_with_pts() function
            vis_final_path = toolbox.util.fullfile(vis_final_dir, subdir, (filename +'.jpg'));
            toolbox.util.mkdir_if_missing(vis_final_path)
            pts_draw_index = np.where(pts_locations[2,:] >= config.visualization_threshold)
            pts_draw = pts_locations[0:2, pts_draw_index[0]]
            labels = [str(i) for i in pts_draw_index[0]]
            image_ori_with_pts = image_ori.copy()
            for i in range(pts_draw.shape[1]):
                cv2.circle(image_ori_with_pts, (int(pts_draw[0,i]), int(pts_draw[1,i])), 2, (0, 0, 255), 2)
            cv2.imwrite(vis_final_path, image_ori_with_pts)



        if config.time:
            intermediate_time = time.time()
            print('time spent on visualizing intermediate heatmap attached on cropped images is %f seconds\n'% (intermediate_time - local_time))

    # add done flag
    done_savepath = toolbox.util.fullfile(output_folder, 'done.txt')
    # this is the only save_to_file option
    toolbox.util.mkdir_if_missing(output_folder)
    
    with open(done_savepath, 'w') as f:
        f.write('start_frame : %d end_frame: %d\n' % (start_frame, _frame))

    print('%d images failed due to bad chopping module\n' % count_failed)
    print('%d images skipped due to lack of input images\n' % count_skipped)
    print('%d images has been finished beforehand\n' % count_prefinished)
    print('%d images are ignored because of bad camera viewpoint\n' % count_ignored)
    print('%d images are ignored because of filtering\n' % count_filtered)
