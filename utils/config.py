# Author: Xinshuo Weng
# Email: xinshuo.weng@gmail.com

# this file sets all configurations dependent on individual systems
import os, sys
import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C
#
# MISC
#

# set the path
__C.HOME = '/home/xinshuo/oculus/HPDFD'					# code path
__C.DATA_HOME = '/media/xinshuo/ssd/datasets'			# data path
__C.DATA_HOME1 = '/media/xinshuo/disk1/datasets'		# data path
__C.DATA_HOME2 = '/media/xinshuo/disk2/datasets'		# data path
__C.DATA_HOME3 = '/mnt/dome/iyu/recordings'				# data path
__C.MODEL_HOME = '/home/xinshuo/models'					# model path
__C.caffe_path = os.path.join(__C.HOME, 'caffe')		# caffe path


__C.DATA_300W = os.path.join(__C.DATA_HOME1, 'face/300-W')	# dataset path
__C.DATA_LS3D = os.path.join(__C.DATA_HOME1, 'face/LS3D-W')

__C.DATA_MUGSY = dict()
__C.DATA_MUGSY['face']		 	= os.path.join(__C.DATA_HOME, 'MUGSY/face')					# dataset path
__C.DATA_MUGSY['face_old']		= os.path.join(__C.DATA_HOME, 'MUGSY/face_old')				# dataset path
__C.DATA_MUGSY['ears'] 			= os.path.join(__C.DATA_HOME, 'MUGSY/ears')					# dataset path
__C.DATA_MUGSY['nose'] 			= os.path.join(__C.DATA_HOME, 'MUGSY/nose')					# dataset path
__C.DATA_MUGSY['inner_lip'] 	= os.path.join(__C.DATA_HOME, 'MUGSY/inner_lip')			# dataset path
__C.DATA_MUGSY['outer_lip'] 	= os.path.join(__C.DATA_HOME, 'MUGSY/outer_lip')			# dataset path
__C.DATA_MUGSY['lower_eyelid'] 	= os.path.join(__C.DATA_HOME, 'MUGSY/lower_eyelid')			# dataset path
__C.DATA_MUGSY['upper_eyelid'] 	= os.path.join(__C.DATA_HOME, 'MUGSY/upper_eyelid')			# dataset path
__C.DATA_MUGSY['lower_teeth'] 	= os.path.join(__C.DATA_HOME, 'MUGSY/lower_teeth')			# dataset path
__C.DATA_MUGSY['upper_teeth'] 	= os.path.join(__C.DATA_HOME, 'MUGSY/upper_teeth')			# dataset path
__C.DATA_MUGSY['mouth'] 		= os.path.join(__C.DATA_HOME, 'MUGSY/mouth')				# dataset path
__C.DATA_MUGSY['overall'] 		= os.path.join(__C.DATA_HOME, 'MUGSY/overall')				# dataset path
__C.DATA_MUGSY['all'] 			= os.path.join(__C.DATA_HOME, 'MUGSY')						# dataset path
__C.DATA_MUGSY['all2'] 			= os.path.join(__C.DATA_HOME2, 'MUGSY')						# dataset path
__C.DATA_MUGSY['mesh'] 			= os.path.join(__C.DATA_HOME, 'MUGSY/meshes')				# dataset path
__C.DATA_MUGSY['calibration'] 	= os.path.join(__C.DATA_HOME, 'MUGSY/calibrations')			# dataset path
__C.DATA_MUGSY['video'] 		= os.path.join(__C.DATA_HOME, 'MUGSY/videos')				# dataset path
__C.DATA_MUGSY['data_src'] 		= __C.DATA_HOME3											# dataset path

__C.DATA_MUGSY['face_dots']		 	= os.path.join(__C.DATA_HOME, 'MUGSY_dots/face')					# dataset path
__C.DATA_MUGSY['inner_lip_dots'] 	= os.path.join(__C.DATA_HOME, 'MUGSY_dots/inner_lip')				# dataset path
__C.DATA_MUGSY['outer_lip_dots'] 	= os.path.join(__C.DATA_HOME, 'MUGSY_dots/outer_lip')				# dataset path
__C.DATA_MUGSY['lower_eyelid_dots'] 	= os.path.join(__C.DATA_HOME, 'MUGSY_dots/lower_eyelid')		# dataset path
__C.DATA_MUGSY['upper_eyelid_dots'] 	= os.path.join(__C.DATA_HOME, 'MUGSY_dots/upper_eyelid')		# dataset path
__C.DATA_MUGSY['lower_teeth_dots'] 	= os.path.join(__C.DATA_HOME, 'MUGSY_dots/lower_teeth')				# dataset path
__C.DATA_MUGSY['upper_teeth_dots'] 	= os.path.join(__C.DATA_HOME, 'MUGSY_dots/upper_teeth')				# dataset path
__C.DATA_MUGSY['mouth_dots'] 		= os.path.join(__C.DATA_HOME, 'MUGSY_dots/mouth')					# dataset path
__C.DATA_MUGSY['all_dots'] 			= os.path.join(__C.DATA_HOME, 'MUGSY_dots')							# dataset path

__C.DATA_MUGSY['face_merged']		 	= os.path.join(__C.DATA_HOME, 'MUGSY_merged/face')					# dataset path
__C.DATA_MUGSY['inner_lip_merged'] 	= os.path.join(__C.DATA_HOME, 'MUGSY_merged/inner_lip')				# dataset path
__C.DATA_MUGSY['outer_lip_merged'] 	= os.path.join(__C.DATA_HOME, 'MUGSY_merged/outer_lip')				# dataset path
__C.DATA_MUGSY['lower_eyelid_merged'] 	= os.path.join(__C.DATA_HOME, 'MUGSY_merged/lower_eyelid')		# dataset path
__C.DATA_MUGSY['upper_eyelid_merged'] 	= os.path.join(__C.DATA_HOME, 'MUGSY_merged/upper_eyelid')		# dataset path
__C.DATA_MUGSY['lower_teeth_merged'] 	= os.path.join(__C.DATA_HOME, 'MUGSY_merged/lower_teeth')				# dataset path
__C.DATA_MUGSY['upper_teeth_merged'] 	= os.path.join(__C.DATA_HOME, 'MUGSY_merged/upper_teeth')				# dataset path
__C.DATA_MUGSY['mouth_merged'] 		= os.path.join(__C.DATA_HOME, 'MUGSY_merged/mouth')					# dataset path
__C.DATA_MUGSY['all_merged'] 			= os.path.join(__C.DATA_HOME, 'MUGSY_merged')							# dataset path


__C.model_pretrained = os.path.join(__C.MODEL_HOME, 'pretrained')

# set the mode
__C.DEBUG = True
__C.VIS = False



#
# Training options
#
__C.TRAIN = edict()
__C.TRAIN.model_save_dir = os.path.join(__C.MODEL_HOME, 'HPDFD')
__C.gpu_id = 1

#
# Testing options
#

__C.TEST = edict()
cfg.test_threshold = 0.2