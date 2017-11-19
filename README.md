# HPDFD
Industrialized Mugsy Facial Landmark Detector!!!!!

## Installation
- clone the github repository recursively including xinshuo_toolbox and its submodules.
~~~shell
git clone --recursive https://github.com/xinshuoweng/HPDFD.git
~~~
- Install our customized Caffe.
~~~shell
cd caffe
make all -j20 && make pycaffe && make matcaffe
cd ..
~~~
- Install required dependency for xinshuo_toolbox (many general functions are heavily used in this project).
~~~shell
cd xinshuo_toolbox
pip install -r requirements.txt
cd ..
~~~
- Install required dependency for this project.
~~~shell
pip install -r requirements.txt
~~~
- Define configuration files (default path for data, code and model) for matlab in utils/get_config.m and for python in utils/config.py. These files are very system or user specific, so we provide *.example for user to define their own configuration files.
- Please start MATLAB from the root folder or run startup.m before using matlab code.

- No need to use MATLAB
- Please build CAFFE in `./caffe` (including pycaffe) and make sure you can do `import caffe` in python first.

## Configure parameters
See `config.py`.


## How to run
Change the parameters and configurations in config.py. Then run 

~~~shell
python main.py
~~~


## Oculus Research Pittsburgh MUGSY-specified settings
Please set `mugsy=True` in `config.py`. This will automatically correct image rotation from MUGSY data.
