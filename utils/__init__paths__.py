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

# Add lib to PYTHONPATH
python_tool_path = os.path.join(this_dir, '../xinshuo_toolbox', 'python')
add_path(python_tool_path)

# Add lib to PYTHONPATH
file_tool_path = os.path.join(this_dir, '../xinshuo_toolbox', 'file_io')
add_path(file_tool_path)

# Add lib to PYTHONPATH
file_tool_path = os.path.join(this_dir, '../xinshuo_toolbox', 'visualization')
add_path(file_tool_path)

# Add lib to PYTHONPATH
file_tool_path = os.path.join(this_dir, '../xinshuo_toolbox', 'computer_vision', 'bbox_transform')
add_path(file_tool_path)

# Add lib to PYTHONPATH
file_tool_path = os.path.join(this_dir, '../xinshuo_toolbox', 'math')
add_path(file_tool_path)

# Add lib to PYTHONPATH
file_tool_path = os.path.join(this_dir, '../xinshuo_toolbox', 'miscellaneous')
add_path(file_tool_path)

# Add lib to PYTHONPATH
file_tool_path = os.path.join(this_dir, '../xinshuo_toolbox', 'images')
add_path(file_tool_path)


# Add lib to PYTHONPATH
file_tool_path = os.path.join(this_dir, '../datasets', 'MUGSY')
add_path(file_tool_path)



