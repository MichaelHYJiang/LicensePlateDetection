# Using *Fast* R-CNN for License Plate Detection on Windows with CPU-only Caffe

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  ```

  You can download [Makefile.config](https://dl.dropboxusercontent.com/s/6joa55k64xo2h68/Makefile.config?dl=0) for reference.<br>
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`<br>
3. [optional] MATLAB (required for PASCAL VOC evaluation only)

### Requirements: hardware

Since files in this directory are based on a CPU-only Caffe, GPUs are not needed. Therefore, basically normal personal computers are sufficient for experiments involved.

### Installation (sufficient for the demo)

1. Clone this Fast R-CNN repository
  
2. Build CPU-only Caffe and pycaffe according to steps in `.\caffe-fast-rcnn\Readme.md`

3. Make sure there are necessary python packages on your computer. Otherwise, install them with command `pip install *`, where `*` stands for a package name.

	**Note:** If your Fast R-CNN is cloned or downloaded from the orignal repository or other places, you need to check following locations:
	- In line 25 of the file `.\lib\utils\nms.pyx`, there should be `np.intp_t`, not `np.int_t` 
	- In line 18 and line 23 of the file `.\lib\setup.py`, there should be `extra_compile_args=[]`, without `"-Wno-cpp", "-Wno-unused-function"`
    
4. Open command line window, change directory to `.\lib`, execute command `python setup.py install`

	**Note:** If there is a problem related to `Unable to find vcvarsall.bat`, you can execute the following command corresponding to your IDE:
	- VS2012: `SET VS90COMNTOOLS=%VS110COMNTOOLS%`
	- VS2013: `SET VS90COMNTOOLS=%VS120COMNTOOLS%`
    
5. When the file `setup.py` finishes installing, you need to visit `python_root\lib\site-packages\utils\`, where lies `cython_bbox.pyd` and `cython_nms.pyd`. You need to copy these two files to `.\lib\utils\`

6. Prepare necessary data files according to readme files in `.\data\` and `.\output\` 

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo, execute `demo.bat`.

The demo performs detection on 5 random pictures in `.\demo\lp\` using a VGG_CNN_M_1024 network trained for License Plate detection. The region proposals are pre-computed in order to reduce installation requirements.

The demo uses pre-computed EdgeBoxes proposals computed with [this code](https://github.com/pdollar/edges).


### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for three networks: CaffeNet (model **S**), VGG_CNN_M_1024 (model **M**), and VGG16 (model **L**).

These models are all available in the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but are provided here for your convenience.

### Usage

**Train** a Fast R-CNN detector. Execute `.\train.bat`:

If you see this error

```
EnvironmentError: MATLAB command 'matlab' not found. Please add 'matlab' to your PATH.
```

then you need to make sure the `matlab` binary is in your `$PATH`. MATLAB is currently required for PASCAL VOC evaluation.
