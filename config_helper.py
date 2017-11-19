# Author: Xinshuo & Shangxuan
# Email: xinshuow@andrew.cmu.edu, wushx6@gmail.com

import toolbox.caffe_config
import toolbox.util
import utils.MUGSY_helper
import utils.caffe_config
import sys, datetime, pdb

class Model:
	pass

class Config:
	model = Model()
	pass

# set configurations
def get_config(MODE, datasets, subset, size_set, data_type, deploy, gpu_id):
	config = Config()
	## init
	if not 'MODE':
		MODE = 'normal'
	
	if not deploy:
		deploy = False
	
	if not gpu_id:
		gpu_id = 0
	
	assert toolbox.util.isstring(MODE), 'mode should be a string'
	if not datasets:
		datasets = 'MUGSY'
	
	if 'MUGSY' in datasets:
		if not subset:
			subset = 'face'
		
		if not size_set:
			size_set = 'resized_4'
		
		if not data_type:
			data_type = 'real'
		
		assert subset == 'face' or subset == 'ears' or subset == 'nose' or subset == 'upper_eyelid' or subset == 'lower_eyelid' or subset == 'outer_lip' or subset == 'inner_lip' or subset == 'upper_teeth' or subset == 'lower_teeth' or subset == 'overall' or  subset == 'face_old' or subset == 'mouth', 'subset is not correct for MUGSY data.'
		assert size_set == 'resized_4' or size_set == 'resized_3' or size_set == 'resized_2' or size_set == 'original' or size_set == 'cropped' or size_set == 'cropped_all', 'size option is not correct for MUGSY data.'
		assert data_type == 'real' or data_type =='synthetic', 'size option is not correct for MUGSY data.'
	
	assert toolbox.util.isstring(datasets), 'datasets should be a string'
	#config = struct()
	
	# set default path
	config.HOME = '/home/xinshuo/oculus/HPDFD'
	config.MODEL_HOME = '/media/xinshuo/disk2/models/HPDFD'
	config.DATA_HOME = '/media/xinshuo/ssd/datasets'
	config.DATA_HOME1 = '/media/xinshuo/disk1/datasets'
	config.DATA_HOME2 = '/media/xinshuo/disk2/datasets'
	config.DATA_300W = toolbox.util.fullfile(config.DATA_HOME1, 'face/300-W')
	config.DATA_LS3D = toolbox.util.fullfile(config.DATA_HOME1, 'face/LS3D-W')

	config.DATA_MUGSY = {}

	# for original dataset
	config.DATA_MUGSY['face_old'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY/face_old')
	config.DATA_MUGSY['face'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY/face')
	config.DATA_MUGSY['ears'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY/ears')
	config.DATA_MUGSY['nose'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY/nose')
	config.DATA_MUGSY['upper_eyelid'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY/upper_eyelid')
	config.DATA_MUGSY['lower_eyelid'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY/lower_eyelid')
	config.DATA_MUGSY['outer_lip'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY/outer_lip')
	config.DATA_MUGSY['inner_lip'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY/inner_lip')
	config.DATA_MUGSY['upper_teeth'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY/upper_teeth')
	config.DATA_MUGSY['lower_teeth'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY/lower_teeth')
	config.DATA_MUGSY['mouth'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY/mouth')
	config.DATA_MUGSY['overall'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY/overall')
	config.DATA_MUGSY['all'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY')

	# dataset with dotted images
	config.DATA_MUGSY['face_old_dots'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_dots/face_old')
	config.DATA_MUGSY['face_dots'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_dots/face')
	config.DATA_MUGSY['ears_dots'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_dots/ears')
	config.DATA_MUGSY['nose_dots'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_dots/nose')
	config.DATA_MUGSY['upper_eyelid_dots'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_dots/upper_eyelid')
	config.DATA_MUGSY['lower_eyelid_dots'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_dots/lower_eyelid')
	config.DATA_MUGSY['outer_lip_dots'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_dots/outer_lip')
	config.DATA_MUGSY['inner_lip_dots'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_dots/inner_lip')
	config.DATA_MUGSY['upper_teeth_dots'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_dots/upper_teeth')
	config.DATA_MUGSY['lower_teeth_dots'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_dots/lower_teeth')
	config.DATA_MUGSY['mouth_dots'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_dots/mouth')
	config.DATA_MUGSY['all_dots'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_dots')
	
	# dataset with all data merged
	config.DATA_MUGSY['face_old_merged'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_merged/face_old')
	config.DATA_MUGSY['face_merged'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_merged/face')
	config.DATA_MUGSY['ears_merged'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_merged/ears')
	config.DATA_MUGSY['nose_merged'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_merged/nose')
	config.DATA_MUGSY['upper_eyelid_merged'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_merged/upper_eyelid')
	config.DATA_MUGSY['lower_eyelid_merged'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_merged/lower_eyelid')
	config.DATA_MUGSY['outer_lip_merged'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_merged/outer_lip')
	config.DATA_MUGSY['inner_lip_merged'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_merged/inner_lip')
	config.DATA_MUGSY['upper_teeth_merged'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_merged/upper_teeth')
	config.DATA_MUGSY['lower_teeth_merged'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_merged/lower_teeth')
	config.DATA_MUGSY['mouth_merged'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_merged/mouth')
	config.DATA_MUGSY['all_merged'] = toolbox.util.fullfile(config.DATA_HOME, 'MUGSY_merged')
	

	
	# mode setting
	config.TRAIN = False				# running for training
	config.TEST = False				# running for testing and evaluation
	config.NORMAL = False				# running for miscellaneous stuff
	config.debug_mode = True			# debug mode
	config.zoomin_mode = True			# visualize the each landmark in zoom-in manner
	# video mode
	config.video_mode = False			# determine to run evaluation on a video set without annotations or on the test dataset
	config.video_name = 'shared_cpm_sigma27_front_dilated_iter145000_batch2'
	config.video_type = 'fixed'
	config.cam_id = 'cam330030'

	# general hyper-parameter
	config.mean_value = 0.5						# mean value used for Caffe
	# config.INT_MAX_CAFFE = 2147483647
	config.INT_MAX_CAFFE = 11184000

	## specific configuration
	
	if MODE == 'normal':
		config.NORMAL = True
		if datasets == '300-W':								# 300-W configuration
			config.num_pts = 68
		elif datasets == 'LS3D-W':
			config.num_pts = 68
		elif datasets == 'MUGSY' or datasets == 'MUGSY_dots' or datasets == 'MUGSY_merged':
			config.num_pts = utils.MUGSY_helper.get_num_pts(subset)
		else:
			assert False, 'Please choose datasets from {''300-W'', ''LS3D-W'', ''MUGSY''}.'
				

	elif MODE == 'train':
		config.TRAIN = True
	elif MODE == 'test':
		config.TEST = True	

		## mode setting
		config.use_gpu = 0					# CPU mode or GPU mode
		config.GPUdeviceNumber = gpu_id	# GPU device number (doesn't matter for CPU mode)
		config.bbox_init = False			# determine if we use a bounding box as initialization during testing and crop it as the input to the network

		config.vis_intermediate = False	# visualize the intermediate results
		config.vis = False					# visualization mode
				
		# evaluation mode
		config.evaluation_mode = True		# demo or evaluation mode
		if deploy:
			config.evaluation_mode = False
			
		config.evaluation_name = 'shared_cpm_sigma27_front_dilated_iter70000_batch2'

		# resume mode
		config.resume_mode = False			# resume the previous job, a config mat file is needed
		config.resume_set = ''				# a config.mat must exists here
		config.fit_testsize = True			# crop the test image to fir the input size of test model

		## hyper-parameter
		# Scaling configuration: starting and ending ratio of person height to image
		# height, and number of scales per octave
		config.num_levels_scale = 1
		config.heatmap_smooth_percentage = 0.4			# percentage of max value in the heat-map for smoothing
		config.visualization_threshold = 0.3			# visualize the detected landmark only when scores are larger than this threshold
		config.compression_rate = 1.5					# compress the lighting of synthetic images
		config.minimum_width = 512						# minimum width of the test image to pad 
		config.heatmap_merge = 'avg'					# use average heat-map from multi-scale
		config.pts_merge_scale = 'max'					# choose best points location from multi-scale
		config.pts_merge_peak = 'max'					# choose best points location from multiple peaks found 
		#	options:
		#				max:			only take local maximum and select the one with highest score
		#				clustering:		not good yet
		#				weighted:		set a threshold to get binary mask, take weighted centroids based on floating value from all blobs inside the mask
		#								no model or share assumed.
		config.pad_value = 128
		config.cluster_bandwidth = 10					# tolerance of distance during clustering peak points found
		config.zoomin_height = 512						# height for cropping in zoom-in visualization
		config.zoomin_width = 512						# width for cropping in zoom-in visualization
		config.frame_interested = [x for x in range(1099)]				# to increase efficiency, we only run evaluation on frames we are interested

		# deploy mode
		if deploy:		
			if subset == 'face_old':
				config.evaluation_name = 'shared_cpm_sigma25_front_dilated_iter115000_batch2'
				datasets = 'MUGSY'
				size_set = 'resized_4'
				config.exp_name = '20170630_15h32m31s'
				config.fit_width = 960
				config.fit_height = 1280
				# config.output_stage = 6
			elif subset == 'face':
				config.evaluation_name = 'shared_cpm_front_dilated_iter145000_batch2'
				datasets = 'MUGSY'
				size_set = 'resized_4'
				config.exp_name = '20170630_11h31m21s'
				config.fit_width = 960
				config.fit_height = 1280
				# config.output_stage = 6
			elif subset == 'lower_eyelid':
				# config.evaluation_name = 'shared_cpm_sigma21_front_dilated_increased_iter140000_batch1'
				config.evaluation_name = 'shared_cpm_sigma21_front_dilated_iter60000_batch1'
				datasets = 'MUGSY_merged'
				size_set = 'cropped_all'
				config.exp_name = '20170810_20h10m51s'
				config.fit_width = 1080
				config.fit_height = 1080
			elif subset == 'upper_eyelid':
				# config.evaluation_name = 'shared_cpm_sigma15_front_dilated_increased_iter125000_batch3'
				config.evaluation_name = 'shared_cpm_sigma15_front_dilated_iter60000_batch1'
				datasets = 'MUGSY_merged'
				size_set = 'cropped_all'
				config.exp_name = '20170810_20h28m12s'
				# config.output_stage = 6
				config.fit_width = 1080
				config.fit_height = 1080
			elif subset == 'nose':
				config.evaluation_name = 'shared_cpm_sigma11_front_dilated_iter25000_batch2'
				datasets = 'MUGSY'
				size_set = 'cropped'
				# config.output_stage = 6
				config.fit_width = 1080
				config.fit_height = 1080
			elif subset == 'ears':
				config.evaluation_name = 'shared_cpm_sigma11_front_dilated_iter45000_batch1'
				datasets = 'MUGSY'
				size_set = 'cropped'
				# config.output_stage = 6
				config.fit_width = 1080
				config.fit_height = 1080
			elif subset == 'mouth':
				config.evaluation_name = 'shared_cpm_sigma27_front_dilated_iter80000_batch2'
				datasets = 'MUGSY_merged'
				size_set = 'cropped_all'
				config.exp_name = '20170809_15h24m35s'
				config.fit_width = 1080
				config.fit_height = 1080	
			else:
				assert False, 'Impossible! Please choose correct subset.'
				
			

		# get info automatically
		model_name = config.evaluation_name.split('_iter')[0] 

		if 'deconv' in config.evaluation_name:
			print('deconv is on.\n')
			config.deconv = True
		else:
			config.deconv = False
			

		print('fetching info for output stage\n')
		config.output_stage = 6
		iter_number_tmp = config.evaluation_name.split('_iter')[1] #iter_number_tmp = iter_number_tmp{2}
		iter_number = int(iter_number_tmp.split('_batch')[0]) #iter_number = str2num(iter_number{1})
		print('fetching info for downsample\n')
		
		##config.downsample = int(utils.MUGSY_helper.fetch_downsample_from_google_sheets(datasets, subset, size_set, model_name, config.debug_mode))
		#if 'resized_' in size_set:
		#	resize_factor = int(size_set.split('_')[1])
		#	config.downsample = config.downsample / resize_factor
			
		config.downsample = 8

		print('fetching info for resize factor\n')
		if subset == 'face_old':
			config.resize_factor = 1
		elif subset == 'mouth':
			config.resize_factor = 0.5
		elif subset == 'upper_eyelid':
			config.resize_factor = 0.5
		elif subset == 'lower_eyelid':
			config.resize_factor = 0.5
		elif subset == 'face':
			config.resize_factor = 1
		else:
			assert False, 'Impossible! Please choose correct subset.'
		#config.resize_factor = 1


		if subset == 'outer_lip' or subset == 'inner_lip':
			config.fit_width = 1360
			config.fit_height = 1360
		elif subset == 'lower_teeth':
			config.fit_width = 1560
			config.fit_height = 1140
		else:
			print('fetching info for fit size\n')
			
			

		## dataset specific configuration
			
		if datasets == '300-W':								# 300-W configuration
			config.num_pts = 68
			config.DATA_PATH = config.DATA_300W
			config.bbox_init = True
			config.scale_starting = 0.3
			config.scale_ending = 1.5
			config.output_stage = 5						# number of stage for the output we want to extract	
			config.exp_name = '20170519_17h16m41s'			# ID for sub-variants of the model
			config.iter_number = 150000
			config.model_name = 'cpm_vanilla'				# model to use
			config.model.description = '300W 68-landmarks detection'
		elif datasets == 'LS3D-W':
			config.num_pts = 68
			config.DATA_PATH = config.DATA_LS3D

			config.model.description = 'LS3D-W 68-landmarks detection'
		elif datasets == 'MUGSY' or datasets == 'MUGSY_merged' or datasets == 'MUGSY_dots':
			config.num_pts = utils.MUGSY_helper.get_num_pts(subset)
			config.scale_starting = 0.7
			config.scale_ending = 1.3

			if datasets == 'MUGSY':
				config.DATA_PATH = toolbox.util.fullfile(config.DATA_MUGSY[subset], size_set)
			elif datasets == 'MUGSY_dots':
				config.DATA_PATH = toolbox.util.fullfile(config.DATA_MUGSY['%s_dots' % subset], size_set)
			else:
				config.DATA_PATH = toolbox.util.fullfile(config.DATA_MUGSY['%s_merged' % subset], size_set)
							
			config.zoomin_mode = True

			print('fetching info for model date\n')
			#config.exp_name = char(py.MUGSY_helper.fetch_model_date_from_google_sheets(datasets, subset, size_set, model_name, config.debug_mode))	# ID for sub-variants of the model
			
			config.iter_number = iter_number	
			config.model_name = model_name				# model to use



			config.model.description = 'MUGSY 20-landmarks detection'
			config.model.stage_depth = 7
		else:
			assert False, 'Please choose datasets from {''300-W'', ''LS3D-W'', ''MUGSY''}.'
				

		# experiment specific configuration
		config.exp_dir = toolbox.util.fullfile(config.HOME, 'experiments', subset, '%s_%s' % (config.model_name, config.exp_name))		
		config.output_dir = toolbox.util.fullfile(config.exp_dir, 'output')
		config.demo_save_dir = toolbox.util.fullfile(config.HOME, 'demo', 'results')
		toolbox.util.mkdir_if_missing(config.demo_save_dir)
		config.model.caffemodel = toolbox.util.fullfile(config.MODEL_HOME, subset, config.model_name, config.exp_name, '%s_iter_%d.caffemodel' % (config.model_name, config.iter_number))
		config.model.deployFile = toolbox.util.fullfile(config.exp_dir, 'hpdfd_test.prototxt')

		# extra info
		config.datasets = datasets
		config.subset = subset
		config.size_set = size_set
		config.data_type = data_type
		config.num_output_channels = config.num_pts + 1	# one more background channel is padded to the last channel
		config.time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')

		## Caffe related
		utils.caffe_config.suppress_caffe_terminal_log()
		caffepath = toolbox.util.fullfile(config.HOME, 'caffe/python')
		sys.path.insert(0,caffepath)
		import caffe	
		sys.path.insert(0,caffepath)
		caffe.set_mode_gpu()
		caffe.set_device(config.GPUdeviceNumber)
		if not config.resume_mode:
			config.net = caffe.Net(config.model.deployFile, config.model.caffemodel, caffe.TEST)
			
		config.datasets = datasets
		config.size_set = size_set
		config.subset = subset

	else:
		assert False, 'Please choose mode from {"normal", "train", "test"}.'
	
	return config

if __name__ == "__main__":
	pass