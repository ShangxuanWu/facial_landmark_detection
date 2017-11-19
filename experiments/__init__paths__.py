#!/usr/bin/env python
# -*- coding: utf-8 -*- 

# Author: Xinshuo
# Email: xinshuow@andrew.cmu.edu
"""Set up paths."""

import sys, os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(os.path.abspath(__file__))

# Add caffe to PYTHONPATH
caffe_path = os.path.join(this_dir, '../caffe', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
util_path = os.path.join(this_dir, '../utils')
add_path(util_path)

# Add lib to PYTHONPATH
util_path = os.path.join(this_dir, '../xinshuo_toolbox', 'file_io')
add_path(util_path)

# Add lib to PYTHONPATH
util_path = os.path.join(this_dir, '../xinshuo_toolbox', 'python')
add_path(util_path)

# Add lib to PYTHONPATH
util_path = os.path.join(this_dir, '../xinshuo_toolbox', 'miscellaneous')
add_path(util_path)

# Add lib to PYTHONPATH
util_path = os.path.join(this_dir, '../datasets', 'MUGSY')
add_path(util_path)
