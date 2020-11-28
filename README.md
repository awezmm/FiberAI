# FiberAI


More in-depth instructions to be provided soon.

## Requirements:

Run pip install -r requirements.txt

Install detectron2 from here https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

Install 2020A MATLAB Runtime Driver

## How to Use:

1. Use resizer.py to resize/slice your images to sizes of 1024 x 1024. 

  Run resizer.py with path of image or folder of input images, and output directory.  
  


2. Use main.py to run model inference and find and measure fibers. This also will produce an output.mat file. 

  Run main.py with path of image or folder of input images, and output directory.  



3. Run Fibermainmenu.mlapp to launch MATLAB interface, and select output.mat file.
