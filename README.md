# FiberAI


More in-depth instructions to be provided soon.

Requirements:

Run pip install -r requirements.txt

Install detectron2 from here https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

Install 2020A MATLAB Runtime Driver


1. Use resizer.py to resize/slice your images to sizes of 1024 x 1024. Run resizer.py with: python resizer.py <path of image or folder> <"f" for folder or "i" for image> <path of output directory>
2. Use main.py to run model inference and find and measure fibers. This also will produce an output.mat file. Run main.py with: python <path of image or folder> <path of output directory>
3. Load the FiberAI MATLAB app to run the interface, with the output.mat file.
