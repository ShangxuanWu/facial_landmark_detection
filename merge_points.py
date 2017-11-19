# Author: Xinshuo Weng
# Email: xinshuow@andrew.cmu.edu
import numpy as np
import os, cv2, argparse

from shutil import copyfile
from PIL import Image

import utils.Mugsy_helper
from xinshuo_visualization import visualize_image_with_pts

import toolbox.util

# merge 2d detection results
#	1. merge results from all parts
#	2. transpose from 3 x N to N x 3
#	3. apply transformation

def merge_2d(results_path, start_frame, end_frame, image_dir=None, resize_factor=1.0, rot_correct=True, mugsy_version=1, debug=True, vis=False, save_vis=True):
	if mugsy_version == 0:
		im_height, im_width = 5120, 3840
	elif mugsy_version == 1:
		im_height, im_width = 4096, 2668
	else:
		assert False, 'the mugsy version is wrong'

	camera_list = get_camera_list(mugsy_version)
	force = False
	part_list = utils.Mugsy_helper.get_detailed_subset_list()
	num_pts_all = get_num_pts('overall', debug=debug)
	save_dir = os.path.join(results_path, 'predictions')
	vis_dir = os.path.join(results_path, 'tmp', 'visualization')
	toolbox.util.mkdir_if_missing(save_dir)
	toolbox.util.mkdir_if_missing(vis_dir)

	# process the predictions
	for camera_id in camera_list:
		rotation_degree = utils.Mugsy_helper.get_rotate_degree(camera_id, mugsy_version, debug=debug)
		for frame_index in range(start_frame, end_frame+1):
			pts_savepath = os.path.join(save_dir, 'cam'+camera_id, '%05d.pose' % frame_index)		
			if toolbox.util.is_path_exists(pts_savepath) and not force:
				continue

			print('processing file %s' % pts_savepath)
			pts_array_all = np.zeros((num_pts_all, 3), dtype='float32')
			exist = False
			for subset in part_list:
				pts_file = os.path.join(results_path, subset, 'predictions', 'cam'+camera_id, 'image%05d.pts' % frame_index)
				if not toolbox.util.is_path_exists(pts_file):
					continue

				num_pts = utils.Mugsy_helper.get_num_pts(subset, debug=debug)
				pts_array = toolbox.util.load_2dmatrix_from_file(pts_file, delimiter=' ', dtype=str, debug=debug)			# 3 x num_pts
				pts_array = pts_array[0:3, 0:num_pts].astype('float32')			# ignore the whitespace				
				pts_array[0:2, :] = pts_array[0:2, :] * resize_factor

				if rot_correct:
					pts_array[0:2, :] = toolbox.util.pts_rotate2D(pts_array[0:2, :], -rotation_degree, im_height, im_width, debug=debug)

				offset_tmp = utils.Mugsy_helper.get_index_offset(subset, debug=debug)
				pts_array_all[offset_tmp : offset_tmp+num_pts, :] = pts_array.transpose().copy()							# num_pts x 3
				exist = True

			if exist:
				toolbox.util.mkdir_if_missing(pts_savepath)
				toolbox.util.save_2dmatrix_to_file(pts_array_all, pts_savepath, formatting='%.4f', debug=debug)
				if vis:
					assert image_dir is not None, 'the image path should not be none for visualization'
					image_path = os.path.join(image_dir, 'cam'+camera_id, 'image%05d.png' % frame_index)
					image_vis = Image.open(image_path)
					pts_vis = np.transpose(pts_array_all)

					save_path_tmp = os.path.join(vis_dir, 'cam'+camera_id, 'image%05d.jpg' % frame_index)
					toolbox.util.mkdir_if_missing(save_path_tmp)
					utils.Mugsy_helper.visualize_image_with_pts(image_vis, pts_vis, debug=debug, vis=vis, save_path=save_path_tmp)

def main():
	# 20170928--003235527

	parser = argparse.ArgumentParser()
	parser.add_argument('--results_path', type=str, default='/media/xinshuo/disk2/datasets/Mugsy_v2/rom/20170928--003235527', help='root dir')
	parser.add_argument('--image_dir', type=str, default='/media/xinshuo/disk2/datasets/Mugsy_v2/rom/20170928--003235527/images', help='root dir')
	parser.add_argument('--sf', type=int, default=38312, help='start frame')
	parser.add_argument('--ef', type=int, default=40111, help='end_frame')
	parser.add_argument('--resize_factor', type=float, default=1.0, help='resize factor')
	parser.add_argument('--mugsy_version', type=int, default=1, help='0: old mugsy, 1: new mugsy')
	parser.add_argument('--rot_correct', type=bool, default=True, help='whether rotation correct is needed')
	parser.add_argument('--save_vis', type=bool, default=False, help='save the visualization')
	args = parser.parse_args()

	vis = False
	debug = False
	merge_2d(args.results_path, args.sf, args.ef, image_dir=args.image_dir, resize_factor=args.resize_factor, rot_correct=args.rot_correct, mugsy_version=args.mugsy_version, vis=vis, debug=debug, save_vis=args.save_vis)

if __name__ == '__main__':
	main()