# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this function parse all the stuff in the file as floating number
# and save it to 2d matrix
# nrows is the number of rows parse from the file

import numpy as np

def parse_text_file(file_path):
	data = np.loadtxt(file_path)
    nrows = data.shape(0)
    return data, nrows
