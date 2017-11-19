# Author: Shangxuan Wu and Xinshuo Weng
# Email: {wushx6, xinshuo.weng}@gmail.com

## Run this script by `python main.py`
## Change settings and parameters in config.py


import pdb, cv2, sys, os
import init_paths, global_config

sys.path.append('./xinshuo_toolbox/python')
sys.path.append('./xinshuo_toolbox/file_io')
sys.path.append('./datasets/MUGSY')
sys.path.append('./caffe/python')

import caffe
import init_paths, config, landmark_detection_singleview, util
#init_paths.main()

from file_io import mkdir_if_missing, load_txt_file
from MUGSY_helper import subset_detailed_convert2compact
from check import is_path_exists

code_base = './'

def main():

    if global_config.use_gpu:
        caffe.set_device(global_config.gpu_id)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()


    util.mkdir_if_missing(global_config.output_folder)
    tmp_dir = os.path.join(output_folder, 'tmp')
    util.mkdir_if_missing(tmp_dir)


    for part_index in range(len(config.part_list)):
        pdb.set_trace()
        part_name = part_list[part_index]

        if part_name == 'face_old':
            done_filepath = os.path.join(save_dir, 'chopping', 'done.txt')
        else:
            done_filepath = os.path.join(save_dir, part_name, 'done.txt')
        if is_path_exists(done_filepath):
            data, _ = load_txt_file(done_filepath)
            done_frame = data[0].split(' ')
            if int(done_frame[0]) == start_frame and int(done_frame[1]) == end_frame:
                print('%s has been finished before' % (part_name))
                continue

        subset = subset_detailed_convert2compact(part_name, debug=True)

        if subset == 'lower_eyelid' or subset == 'upper_eyelid' or subset == 'ears':
            prefix = part_name.split('_')[0]
        else:
            prefix = ' '

        m_scrpt_tmp = os.path.join(tmp_dir, '%s.m' % part_name)
        with open(m_scrpt_tmp, 'w') as file:
            str_to_write = write_part_script(images_dir, save_dir, subset, prefix, resize_factor, gpu_id, rot_correct, gopro, start_frame, end_frame, vis_resize_factor, camera_id)
            file.write(str_to_write)

        # use my own python file
        landmark_detection_singleview.process(code_base, images_dir, save_dir, part_name, prefix, resize_factor, gpu_id, rot_correct, gopro, start_frame, end_frame, vis_resize_factor, camera_id)
        caffe.reset_all()        


if __name__ == '__main__':
    main()
