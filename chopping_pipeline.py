# Author: Xinshuo
# Email: xinshuow@andrew.cmu.edu

# this file run chopping pipeline in a single gpu for all parts

import sys, os, pdb
#sys.path.append('/home/xinshuo/oculus/HPDFD/xinshuo_toolbox/python')
#sys.path.append('/home/xinshuo/oculus/HPDFD/xinshuo_toolbox/file_io')
sys.path.append('/home/xinshuo/oculus/HPDFD/datasets/MUGSY')

import init_paths, landmark_detection_singleview
init_paths.main()
import utils.config

#from file_io import mkdir_if_missing, load_txt_file
import toolbox.util
import utils.MUGSY_helper
import merge_points

code_base = '/home/shangxuan/HPDFD_new'

def main():
    if len(sys.argv) < 12:
    	print "images_dir save_dir resize_factor gpu_id start_frame end_frame rot_correct gopro vis_resize_factor camera_id part1 [part2] [part3] ...";
    	sys.exit(-1);

    images_dir = sys.argv[1]
    save_dir = sys.argv[2]
    resize_factor = float(sys.argv[3])
    gpu_id = int(sys.argv[4])
    start_frame = int(sys.argv[5])
    end_frame = int(sys.argv[6])

    rot_correct = int(sys.argv[7])
    gopro = int(sys.argv[8])
    vis_resize_factor = float(sys.argv[9])
    camera_id = sys.argv[10]

    toolbox.util.mkdir_if_missing(save_dir)
    tmp_dir = os.path.join(save_dir, 'tmp')
    toolbox.util.mkdir_if_missing(tmp_dir)

    #part_list = ['face_old', 'left_lower_eyelid', 'face', 'right_lower_eyelid', 'left_upper_eyelid', 'right_upper_eyelid', 'mouth']
    part_list = ['face_old']

    if sys.argv[11] == 'all':
        part_list = ['face_old', 'left_lower_eyelid', 'face', 'right_lower_eyelid', 'left_upper_eyelid', 'right_upper_eyelid', 'mouth']
    elif sys.argv[11] == 'left_ears':
        part_list = ['left_ears']
    elif sys.argv[11] == 'right_ears':
        part_list = ['right_ears']
    else:
        # push all
        for i in range(len(sys.argv) - 11):
            part_list.append(sys.argv[11+i])

    for part_index in range(len(part_list)):
        part_name = part_list[part_index]

        if part_name == 'face_old':
            done_filepath = os.path.join(save_dir, 'chopping', 'done.txt')
        else:
            done_filepath = os.path.join(save_dir, part_name, 'done.txt')
        if toolbox.util.is_path_exists(done_filepath):
            data, _ = toolbox.util.load_txt_file(done_filepath)
            done_frame = data[0].split(' ')
            if int(done_frame[0]) == start_frame and int(done_frame[1]) == end_frame:
                print('%s has been finished before' % (part_name))
                continue

        subset = utils.MUGSY_helper.subset_detailed_convert2compact(part_name, debug=True)

        if subset == 'lower_eyelid' or subset == 'upper_eyelid' or subset == 'ears':
            prefix = part_name.split('_')[0]
        else:
            prefix = ' '

        # use my own python file
        landmark_detection_singleview.process(images_dir, save_dir, subset, prefix, resize_factor, gpu_id, rot_correct, gopro, start_frame, end_frame, vis_resize_factor, camera_id)
    merge_points.merge_2d(save_dir, start_frame, end_frame, image_dir, resize_factor, rot_correct, 0, False, False, False)


if __name__ == '__main__':
    main()