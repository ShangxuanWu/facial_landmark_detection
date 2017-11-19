# Author: Xinshuo Weng
# Email: xinshuow@andrew.cmu.edu

# this file includes general help functions for MUGSY data
import init_paths
init_paths.main()
from check import isstring, isscalar
from file_io import get_sheet_service, update_patchs2sheet, get_data_from_sheet, update_row2sheet

# # number of points for all parts
# num_pts = dict()
# num_pts['face_old']					= 20
# num_pts['face'] 						= 26
# num_pts['left_upper_eyelid'] 			= 30
# num_pts['right_upper_eyelid'] 		= 30
# num_pts['upper_eyelid'] 				= 30
# num_pts['left_lower_eyelid'] 			= 17
# num_pts['right_lower_eyelid'] 		= 17
# num_pts['lower_eyelid'] 				= 17
# num_pts['nose'] 						= 10
# num_pts['outer_lip'] 					= 16
# num_pts['inner_lip'] 					= 16
# num_pts['upper_teeth']				= 18
# num_pts['lower_teeth']				= 18
# num_pts['left_ears'] 					= 19
# num_pts['right_ears'] 				= 19
# num_pts['ears']		 				= 19
# num_pts['mouth']		 				= 68
# # num_pts['iris']						= 3
# # num_pts['pupil']					= 3
# num_pts['overall'] 					= 236

# # index offset of keypoints for all parts
# index_offset = dict()
# index_offset['face_old'] 			= 0
# index_offset['face'] 					= 0
# index_offset['left_upper_eyelid'] 		= 26
# index_offset['right_upper_eyelid'] 		= 56
# index_offset['left_lower_eyelid'] 		= 86
# index_offset['right_lower_eyelid'] 		= 103
# index_offset['nose'] 					= 120
# index_offset['outer_lip'] 				= 130
# index_offset['inner_lip'] 				= 146
# index_offset['upper_teeth']				= 162
# index_offset['lower_teeth']				= 162			# lower teeth already has offset in the raw_annotations
# index_offset['left_ears'] 				= 198
# index_offset['right_ears'] 				= 217
# index_offset['mouth']	 				= 130
# # index_offset['iris']					= 236
# # index_offset['pupil']					= 239
# index_offset['overall']					= 0

# anno_version = 1



# number of points for all parts
num_pts = dict()
num_pts['face_old']					= 20
num_pts['face'] 					= 26
num_pts['left_upper_eyelid'] 		= 24
num_pts['right_upper_eyelid'] 		= 24
num_pts['upper_eyelid'] 			= 24
num_pts['left_lower_eyelid'] 		= 17
num_pts['right_lower_eyelid'] 		= 17
num_pts['lower_eyelid'] 			= 17
num_pts['nose'] 					= 10
num_pts['outer_lip'] 				= 16
num_pts['inner_lip'] 				= 16
num_pts['upper_teeth']				= 18
num_pts['lower_teeth']				= 18
num_pts['left_ears'] 				= 19
num_pts['right_ears'] 				= 19
num_pts['ears']		 				= 19
num_pts['mouth']		 			= 68
# num_pts['iris']					= 3
# num_pts['pupil']					= 3
num_pts['overall'] 					= 224

# index offset of keypoints for all parts
index_offset = dict()
index_offset['face_old'] 				= 0

index_offset['face'] 					= 0
index_offset['left_upper_eyelid'] 		= 26
index_offset['right_upper_eyelid'] 		= 50
index_offset['left_lower_eyelid'] 		= 74
index_offset['right_lower_eyelid'] 		= 91
index_offset['nose'] 					= 108
index_offset['mouth']	 				= 118
index_offset['outer_lip'] 				= 118
index_offset['inner_lip'] 				= 134
index_offset['upper_teeth']				= 150
index_offset['lower_teeth']				= 150			# lower teeth already has offset in the raw_annotations

index_offset['left_ears'] 				= 186
index_offset['right_ears'] 				= 205

# index_offset['iris']					= 224
# index_offset['pupil']					= 227
index_offset['overall']					= 0

anno_version = 2






rotate_degree = {'330001':  90, '330005': -90, '330006': -90, '330007': -90, '330010':  90, '330011': -90, '330012': -90, '330013': -90, '330014': -90, '330015': -90,
 				 '330016': -90, '330017': -90, '330018': -90, '330019':  90, '330020':  90, '330021':  90, '330022':  90, '330023': -90, '330024': -90, '330025': -90, 
 				 '330026': -90, '330027': -90, '330028': -90, '330029': -90, '330030': -90, '330031': -90, '330032': -90, '330033': -90, '330034': -90, '330035':  90, 
 				 '330036': -90, '330037': -90, '330038':  90, '330039': -90, '330040': -90, '330041': -90, '330042': -90, '330043': -90, '330044': -90, '330045': -90}

rotate_degree_v2 = {'330000': -90, '330005':  90, '330006': -90, '330007':  90, '330010': -90, '330011': -90, '330012': -90, '330014':  90, '330015': -90, '330016':  90, 
					'330017':  90, '330018': -90, '330019': -90, '330020':  90, '330022':  90, '330023':  90, '330024': -90, '330025':  90, '330026': -90, '330027': -90, 
					'330028':  90, '330029':  90, '330030':  90, '330031':  90, '330032':  90, '330033':  90, '330036':  90, '330037':  90, '330038': -90, '330040': -90, 
					'330041': -90, '330042': -90, '330043': -90, '330045': -90, '400004': -90, '400007': -90, '400008': -90, '400010':  90, '400012':  90, '400017': -90, 
					'400021':  90, '400024':  90, '400025':  90, '400028':  90, '400036': -90, '400039': -90, '400040':  90, '400041': -90, '410001': -90, '410004':  90, 
					'410016':  90, '410018': -90, '410019':  90, '410029':  90, '410033':  90, '410043': -90, '410044': -90, '410045':  90, '410048': -90, '410049':  90,
					'410050':  90, '410051': -90, '410053':  90, '410057': -90, '410061': -90, '410066':  90, '410067':  90, '410068': -90, '410069':  90, '410070': -90, 
					'410073': -90,}


def get_rotate_dict():
	return rotate_degree

def get_rotate_degree(camera_id, debug=True):
	if debug:
		assert isstring(camera_id), 'the input camera id is not a string for getting rotation degree'
		assert camera_id in get_camera_list(), 'the camera id requested: %s does not exist' % camera_id

	return rotate_degree[camera_id]


def get_rotate_dict_v2():
	return rotate_degree_v2

def get_rotate_degree_v2(camera_id, debug=True):
	if debug:
		assert isstring(camera_id), 'the input camera id is not a string for getting rotation degree'
		assert camera_id in get_camera_list(), 'the camera id requested: %s does not exist' % camera_id

	return rotate_degree_v2[camera_id]


def get_compact_subset_list():
	return ['face', 'ears', 'lower_eyelid', 'upper_eyelid', 'nose', 'outer_lip', 'inner_lip', 'upper_teeth', 'lower_teeth', 'mouth']

def get_detailed_subset_list():
	return ['face', 'left_ears', 'right_ears', 'left_lower_eyelid', 'right_lower_eyelid', 'left_upper_eyelid', 'right_upper_eyelid', 'nose', 'outer_lip', 'inner_lip', 'upper_teeth', 'lower_teeth', 'mouth']

def get_camera_list():
	return ['330001', '330005', '330006', '330007', '330010', '330011', '330012', '330014', '330015', '330016', '330017', '330018', '330019', '330020', '330021', '330022', '330023', 
			'330024', '330025', '330026', '330027', '330028', '330029', '330030', '330031', '330032', '330033', '330034', '330035', '330036', '330037', '330038', '330039', '330040',
			'330041', '330042', '330043', '330044', '330045']

def get_camera_list_v2():
	return rotate_degree_v2.keys()


def subset_detailed_convert2compact(subset, debug=True):
	'''
	convert a subset in detailed version to the corresponding compact version
	'''

	if debug:
		assert subset in get_detailed_subset_list() or subset == 'face_old', 'the input subset is not in the detailed subset list'

	if subset == 'left_lower_eyelid' or subset == 'right_lower_eyelid':
		return 'lower_eyelid'
	elif subset == 'left_upper_eyelid' or subset == 'right_upper_eyelid':
		return 'upper_eyelid'
	elif subset == 'left_ears' or subset == 'right_ears':
		return 'ears'
	else:
		return subset


def get_left_camera_list():
	return ['330035', '330039', '330036', '330024', '330023', '330045',
			'330021', '330040', '330037', '330041', '330043', '330033', '330028', '330030',
			'330025', '330042', '330038', '330020', '330012']

def get_right_camera_list():
	return ['330016', '330011', '330032', '330017', '330005', '330019',
		    '330014', '330034', '330006', '330015', '330018', '330026', '330044', '330027', '330022',
		    '330031', '330010', '330001', '330029', '330014', '330007' ]

def get_filename(recording_id, recording_type, camera_id, frame_number, labeler_id, debug=True):
	'''
	return the full filename given all info
	'''
	if debug:
		assert isstring(recording_id), 'recording id is not a string'
		assert isstring(recording_type), 'recording type is not a string'
		assert isscalar(frame_number), 'frame number is not a scalar'
		assert isstring(labeler_id), 'labeler id is not a string'
		assert camera_id in get_camera_list(), 'camera id %s is not in the camera list' % camera_id

	return '--'.join([recording_id, recording_type, camera_id, '%05d' % (frame_number), labeler_id])


def get_image_id(filename, debug=True):
	'''
	return the real image id and the labeler id, this function assume the name is separated by '--'
	'''
	if debug:
		assert isstring(filename), 'input filename is not a string'

	substrings = filename.split('--')
	image_id = '--'.join(substrings[:-1])

	return image_id

def get_labeler_id(filename, debug=True):
	'''
	return the real image id and the labeler id, this function assume the name is separated by '--'
	'''
	if debug:
		assert isstring(filename), 'input filename is not a string'

	substrings = filename.split('--')
	labeler_id = substrings[-1]

	return labeler_id

def get_frame_number(filename, debug=True):
	'''
	extract the frame number from MUGSY filename
	'''
	if debug:
		assert isstring(filename), 'input filename is not a string'

	substrings = filename.split('--')
	return int(substrings[3])


def get_person_id(filename, debug=True):
	'''
	extract the person ID from MUGSY filename
	'''
	if debug:
		assert isstring(filename), 'input filename is not a string'

	substrings = filename.split('_')
	return substrings[1]	

def get_recording_id(filename, debug=True):
	'''
	extract the recording id, including date and person id and dot flag
	'''
	if debug:
		assert isstring(filename), 'input filename is not a string'

	substrings = filename.split('--')
	return substrings[0]	


def get_recording_type(filename, debug=True):
	'''
	extract the recording type, sentence or neutral tongue or expression
	'''
	if debug:
		assert isstring(filename), 'input filename is not a string'

	substrings = filename.split('--')
	return substrings[1]	


def get_camera_id(filename, debug=True):
	'''
	extract the camera ID from MUGSY filename
	'''
	if debug:
		assert isstring(filename), 'input filename is not a string'

	# print filename
	substrings = filename.split('--')
	return substrings[2]	

def get_image_name(image_id, labeler_id, debug=True):
	'''
	merge the image id and labeler id and returns the imagename
	'''
	return image_id + '--' + labeler_id

def get_crop_bbox_from_subset_and_camera(subset, camera_id, debug=True):
	'''
	get pre-defined cropping bbox from subset and camera id

	return:
			bbox in TLBR format
	'''
	if debug:
		assert subset in get_detailed_subset_list() or subset in get_compact_subset_list() or subset == 'face_old', 'subset is not correct!'
		assert camera_id in get_camera_list(), 'camera id is not correct!'

	if 'lower_eyelid' in subset:
		if camera_id == '330030':
			bbox = [700, 1169, 1659, 1888]

	return bbox

def check_left_right(subset, filename, debug=True):
	if debug:
		assert subset in ['ears', 'lower_eyelid', 'upper_eyelid'], 'subset is not correct!'

	if subset == 'lower_eyelid':
		camera_id = get_camera_id(filename, debug=debug)
		if camera_id == '330014':
			return 'right'
		elif camera_id == '330030':
			return 'left'
		else:
			assert False, 'camera wrong!!'
	elif subset == 'upper_eyelid':
		if camera_id in ['330001', '330010']:
			return 'right'
		elif camera_id == ['330020', '330038']:
			return 'left'
		else:
			assert False, 'camera wrong!!'
	else:
		assert False, 'not supported'


def get_line_index_list(subset, debug=True):
	if debug:
		assert subset in get_detailed_subset_list() or subset in get_compact_subset_list() or subset == 'face_old' or subset == 'overall', 'subset is not correct!'

	if 'lower_eyelid' in subset:
		return [[0, 8, 4, 7, 2, 6, 3, 5, 1], [16, 15, 14, 13, 12, 11, 10, 9]]
	elif 'upper_eyelid' in subset:
		return [[0, 8, 4, 7, 2, 6, 3, 5, 1], [16, 15, 14, 13, 12, 11, 10, 9], [23, 22, 21, 20, 19, 18, 17]]
	elif subset == 'outer_lip':
		return [[0, 8, 4, 9, 2, 10, 5, 11, 1], [0, 12, 6, 13, 3, 14, 7, 15, 1]]
	elif subset == 'inner_lip':
		return [[1, 11, 5, 10, 2, 9, 4, 8, 0], [0, 12, 6, 13, 3, 14, 7, 15, 1]]
	elif subset == 'face':
		return [[0, 1, 4, 3, 5, 2], [8, 9, 16, 12, 15, 10, 13, 11, 14, 8], [17, 18, 24, 21, 25, 19, 23, 20, 22, 17], [6, 7]]
	elif subset == 'lower_teeth':
		return [[17, 8, 7, 16, 17], [7, 6, 15, 16], [6, 5, 14, 15], [5, 0, 9, 14], [0, 1, 10, 9], [1, 2, 11, 10], [2, 3, 12, 11], [3, 4, 13, 12]]
	elif subset == 'upper_teeth':
		return [[8, 17, 16, 7, 8], [16, 15, 6, 7], [15, 14, 5, 6], [14, 9, 0, 5], [9, 10, 1, 0], [10, 11, 2, 1], [11, 12, 3, 2], [12, 13, 4, 3]]
	elif subset == 'mouth':
		return [[0, 8, 4, 9, 2, 10, 5, 11, 1], [0, 12, 6, 13, 3, 14, 7, 15, 1], [17, 27, 21, 26, 18, 25, 20, 24, 16], [16, 28, 22, 29, 19, 30, 23, 31, 17], 
				[40, 49, 48, 39, 40], [48, 47, 38, 39], [47, 46, 37, 38], [46, 41, 32, 37], [41, 42, 33, 32], [42, 43, 34, 33], [43, 44, 35, 34], [44, 45, 36, 35],
				[67, 58, 57, 66, 67], [57, 56, 65, 66], [56, 55, 64, 65], [55, 50, 59, 64], [50, 51, 60, 59], [51, 52, 61, 60], [52, 53, 62, 61], [53, 54, 63, 62]]
	else:
		assert False, '%s is not supported' % subset

def get_camera_from_subset(subset, debug=True):
	'''
	get camera id for a specific subset as many parts are captured only from several fixed camera position

	return:	
			a list of camera id suitable for this specific part
	'''
	if debug:
		assert subset in get_detailed_subset_list() or subset in get_compact_subset_list() or subset == 'face_old', 'subset is not correct!'

	if 'lower_eyelid' in subset:
		return ['330030', '330014']
	elif 'left_upper_eyelid' == subset:
		return ['330020', '330038', '330030', '330014']
	elif 'right_upper_eyelid' == subset:
		return ['330010', '330001', '330030', '330014']
	elif subset == 'face_old':
		return ['330030', '330014', '330010', '330001', '330020', '330038', '330012', '330031']
	elif 'outer_lip' == subset:
		return ['330030']
	elif 'inner_lip' == subset:
		return ['330030']
	elif 'lower_teeth' == subset:
		return ['330030']
	elif 'upper_teeth' == subset:
		return ['330030']
	elif 'mouth' == subset:
		return ['330030']
	elif 'nose' == subset:
		return ['330012', '330031']
	elif subset == 'face':
		return ['330030', '330014', '330012', '330031']
	else:
		assert False, '%s is not supported' % subset

def get_num_pts(subset, debug=True):
	'''
	get number of points for a specific facial part
	'''
	if debug:
		assert subset in get_detailed_subset_list() or subset in get_compact_subset_list() or subset == 'face_old' or subset == 'overall', 'subset is not correct!'

	return num_pts[subset]

def get_index_offset(subset, debug=True):
	'''
	get number of points for a specific facial part
	'''
	if debug:
		assert subset in get_detailed_subset_list() or subset == 'face_old', 'subset is not correct!'

	return index_offset[subset]

def get_anno_version():
	return anno_version

def get_num_pts_all():
	return num_pts['overall']

def get_detailed_subset(filename, subset, debug=True):
	'''
	for ears, lower_eyelid, upper_eyelid, this function returns the left right part based on camera position
	'''
	if subset in ['face', 'nose', 'upper_teeth', 'lower_teeth', 'outer_lip', 'inner_lip']:
		return subset
	else:
		camera_id = get_camera_id(filename, debug=debug)
		if camera_id in get_left_camera_list():
			return 'left_' + subset
		elif camera_id in get_right_camera_list():
			return 'right_' + subset
		else:
			assert False, 'camera ID %s error!' % camera_id

def get_part_index_from_chopping(subset, debug=True):
	'''
	get part index for each individual part from face old dataset

	return 
			a list of index
	'''

	if debug:
		assert subset in get_detailed_subset_list() or subset in get_compact_subset_list() or subset == 'face_old', 'subset is not correct!'

	if subset in ['right_lower_eyelid', 'right_upper_eyelid', 'outer_lip', 'inner_lip', 'upper_teeth', 'lower_teeth', 'left_lower_eyelid', 'left_upper_eyelid', 'mouth']:
		if subset == 'right_lower_eyelid' or subset == 'right_upper_eyelid':
			return [4, 5, 6, 7]
		elif subset == 'left_lower_eyelid' or subset == 'left_upper_eyelid':
			return [8, 9, 10, 11]
		elif subset in ['outer_lip', 'inner_lip', 'upper_teeth', 'lower_teeth', 'mouth']:
			return [12, 13, 14, 17]
	else:
		assert False, '%s part is not supported in face old dataset' % subset


def get_search_range_in_sheet():
	return 1000			# define the range to search the name in google sheet


def get_experiments_sheet_id():
	return '1ViiXL89ek9rLACudOnLbAu6_c4UIyIBtosIhkIncWiE'			# sheet id for trained model

def get_training_params_colums_in_experiments_sheet():
	return ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']

def get_row_index_from_experiments_sheet(dataset, subset, size_set, model_name, debug=True):							# all experiments are unique
	'''
	the returned index is already 1-indexed
	'''
	if debug:
		assert subset in get_compact_subset_list() or subset == 'face_old', 'subset is not correct!'
		assert size_set in ['resized_4', 'cropped', 'cropped_all'], 'size set is not correct'

	column_search_range = range(get_search_range_in_sheet())	
	# find index in rows in experiments sheet
	columns_search_dataset = ['A%d' % (search_value_tmp + 1) for search_value_tmp in column_search_range]
	columns_name_dataset = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_experiments_sheet_id(), search_range=columns_search_dataset, debug=debug)		# 
	columns_search_subset = ['B%d' % (search_value_tmp + 1) for search_value_tmp in column_search_range]
	columns_name_subset = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_experiments_sheet_id(), search_range=columns_search_subset, debug=debug)		# 
	columns_search_sizeset = ['C%d' % (search_value_tmp + 1) for search_value_tmp in column_search_range]
	columns_name_sizeset = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_experiments_sheet_id(), search_range=columns_search_sizeset, debug=debug)		# 
	columns_search_model = ['D%d' % (search_value_tmp + 1) for search_value_tmp in column_search_range]
	columns_name_model = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_experiments_sheet_id(), search_range=columns_search_model, debug=debug)		# 

	row_index_exp = None
	for row_index in column_search_range:
		if columns_name_subset[row_index] == subset and columns_name_sizeset[row_index] == size_set and columns_name_model[row_index] == model_name and columns_name_dataset[row_index] == dataset:
			row_index_exp = row_index+1
			break

	if row_index_exp is None:
		assert False, 'No entry model (%s, %s, %s, %s) found!' % (dataset, subset, size_set, model_name)

	return row_index_exp

def fetch_fitsize_from_google_sheets(dataset, subset, size_set, model_name, debug=True):
	'''
	get input size during training
	'''
	row_index = get_row_index_from_experiments_sheet(dataset, subset, size_set, model_name, debug=debug)
	inputsize = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_experiments_sheet_id(), search_range='F'+str(row_index), debug=debug)[0]
	substrings = inputsize.split(' ')
	width = substrings[0]
	height = substrings[2]
	return [width, height]

def fetch_downsample_from_google_sheets(dataset, subset, size_set, model_name, debug=True):
	'''
	get downsample factor during training
	'''
	row_index = get_row_index_from_experiments_sheet(dataset, subset, size_set, model_name, debug=debug)
	downsample = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_experiments_sheet_id(), search_range='H'+str(row_index), debug=debug)[0]
	substrings = downsample.split('x')
	return substrings[0]

def fetch_model_date_from_google_sheets(dataset, subset, size_set, model_name, debug=True):
	'''
	get model date during training
	'''
	row_index = get_row_index_from_experiments_sheet(dataset, subset, size_set, model_name, debug=debug)
	date = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_experiments_sheet_id(), search_range='P'+str(row_index), debug=debug)[0]
	return date

def fetch_resize_factor_from_google_sheets(dataset, subset, size_set, model_name, debug=True):
	'''
	get model date during training
	'''
	row_index = get_row_index_from_experiments_sheet(dataset, subset, size_set, model_name, debug=debug)
	factor = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_experiments_sheet_id(), search_range='g'+str(row_index), debug=debug)[0]
	return factor

def fetch_output_stage_from_google_sheets(dataset, subset, size_set, model_name, debug=True):
	'''
	get model date during training
	'''
	row_index = get_row_index_from_experiments_sheet(dataset, subset, size_set, model_name, debug=debug)
	out_stage = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_experiments_sheet_id(), search_range='I'+str(row_index), debug=debug)[0]
	out_stage = out_stage.split('S')
	return out_stage[-1]

def get_evaluation_sheet_id():
	return '1cmORxhEOD-E4cYuaKXrJgMeiJ2h5uu4mIoESawaTvFg'  			# sheet id for evaluated model

def get_training_params_colums_in_evaluation_sheet():
	return ['E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

def get_testing_params_colums_in_evaluation_sheet():
	return ['M', 'N', 'O', 'P', 'Q', 'R']

def get_row_index_list_from_evaluation_sheet(dataset, subset, size_set, evaluation_name, debug=True):							# all evaluated models might not be unique
	'''
		the returned index is already 1-indexed
	'''
	if debug:
		assert subset in get_compact_subset_list() or subset == 'face_old', 'subset is not correct!'
		assert size_set in ['resized_4', 'cropped', 'cropped_all'], 'size set is not correct'

	column_search_range = range(get_search_range_in_sheet())	
	# find index in rows in experiments sheet
	columns_search_dataset = ['A%d' % (search_value_tmp + 1) for search_value_tmp in column_search_range]
	columns_name_dataset = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_evaluation_sheet_id(), search_range=columns_search_dataset, debug=debug)		# 
	columns_search_subset = ['B%d' % (search_value_tmp + 1) for search_value_tmp in column_search_range]
	columns_name_subset = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_evaluation_sheet_id(), search_range=columns_search_subset, debug=debug)		# 
	columns_search_sizeset = ['C%d' % (search_value_tmp + 1) for search_value_tmp in column_search_range]
	columns_name_sizeset = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_evaluation_sheet_id(), search_range=columns_search_sizeset, debug=debug)		# 
	columns_search_model = ['D%d' % (search_value_tmp + 1) for search_value_tmp in column_search_range]
	columns_name_model = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_evaluation_sheet_id(), search_range=columns_search_model, debug=debug)		# 
	row_index_list = list()
	for row_index in column_search_range:
		if columns_name_subset[row_index] == subset and columns_name_sizeset[row_index] == size_set and columns_name_model[row_index] == evaluation_name and columns_name_dataset[row_index] == dataset:
			row_index_list.append(row_index + 1)

	if len(row_index_list) == 0:
		assert False, '%s, %s, %s, %s is not on the search range within the google sheet' % (dataset, subset, size_set, evaluation_name)

	return row_index_list

def update_info_evaluation_sheet(dataset, subset, size_set, model_name, evaluation_name, info_list, debug=True):
	exp_index = get_row_index_from_experiments_sheet(dataset, subset, size_set, model_name, debug=debug)
	evaluation_index_list = get_row_index_list_from_evaluation_sheet(dataset, subset, size_set, evaluation_name, debug=debug)

	update_training_info_evaluation_sheet(exp_index, evaluation_index_list, debug=debug)
	update_testing_info_evaluation_sheet(evaluation_index_list, info_list, debug=debug)

def update_training_info_evaluation_sheet(exp_index, evaluation_index_list, debug=True):

	add_index_exp = lambda x: x+str(exp_index)
	columns_list = list(map(add_index_exp, get_training_params_colums_in_experiments_sheet()))
	training_info_list = get_data_from_sheet(service=get_sheet_service(), sheet_id=get_experiments_sheet_id(), search_range=columns_list, debug=debug)		# 

	# paste info to evaluation sheet
	for line_index in evaluation_index_list:
		add_index_evaluation = lambda x: x+str(line_index)
		columns_list = list(map(add_index_evaluation, get_training_params_colums_in_evaluation_sheet()))
		update_row2sheet(service=get_sheet_service(), sheet_id=get_evaluation_sheet_id(), row_starting_position=columns_list[0], data=training_info_list, debug=debug)		# 

def update_testing_info_evaluation_sheet(evaluation_index_list, info_list, debug=True):
	'''
	update testing configuration to the record
	'''
	if debug:
		assert len(info_list) == len(get_testing_params_colums_in_evaluation_sheet()), 'the information list is not correct %d vs %d' % (len(info_list), len(get_testing_params_colums_in_evaluation_sheet()))

	info_list = list(info_list)
	for line_index in evaluation_index_list:
		row_starting_position = get_testing_params_colums_in_evaluation_sheet()[0] + str(line_index)
		update_row2sheet(service=get_sheet_service(), sheet_id=get_evaluation_sheet_id(), row_starting_position=row_starting_position, data=info_list, debug=debug)