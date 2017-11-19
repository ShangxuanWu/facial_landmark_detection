# Author: Shangxuan Wu and Xinshuo Weng
# Email: {wushx6, xinshuo.weng}@gmail.com

## Frequently-used configurations, please modify it

# input image paths in a txt list 
input_img_list = './example_image_list.txt'

# output folder for storing images and detection results
output_folder = './output/'

# gpu settings
use_gpu = False
gpu_id = 0

# where the trained models are stored
model_folder = '~/'

# time the detection process
profile = True

# gopro settings
gopro = False

# visualize
vis = False

# MUGSY special settings
is_MUGSY = True

if is_MUGSY:
	pass

## Fixed configurations, don't modify it unless necessary

part_list = ['face_old', 
			'left_lower_eyelid', 
			'face', 
			'right_lower_eyelid', 
			'left_upper_eyelid', 
			'right_upper_eyelid', 
			'mouth']


# there is some problem of ear models and mouth models, I'm not that sure which model it is
deploy_model = {
	'face_old': 'shared_cpm_sigma25_front_dilated_iter115000_batch2',
	'face': 'shared_cpm_front_dilated_iter145000_batch2',
	'lower_eyelid':'shared_cpm_sigma21_front_dilated_iter60000_batch1',
	'upper_eyelid':'shared_cpm_sigma15_front_dilated_iter60000_batch1',
	'nose':'shared_cpm_sigma11_front_dilated_iter25000_batch2',
	'ears':'shared_cpm_sigma11_front_dilated_iter45000_batch1',
	'mouth':'shared_cpm_sigma27_front_dilated_iter80000_batch2',
}

size_set = {
	'face_old': 'resized_4',
	'face': 'resized_4',
	'lower_eyelid':'cropped_all',
	'upper_eyelid':'cropped_all',
	'nose':'cropped',
	'ears':'resized_4',
	'mouth':'cropped_all',
}

size_set = {
	'face_old': 'resized_4',
	'face': 'resized_4',
	'lower_eyelid':'cropped_all',
	'upper_eyelid':'cropped_all',
	'nose':'cropped',
	'ears':'cropped',
	'mouth':'cropped_all',
}