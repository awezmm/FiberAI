# FiberAI


## Setup

After cloning/downloading this repository, use terminal to 'cd' into this repository.

### 1. Create a virtual environment:
Packages and dependencies can be installed into a virtual environemnt, so that they don't interfere with other package versions on your computer. Follow these steps to create a new virtual environment:

Create a virtual environemnt inside the downloaded directory. Run the command:
```python3 -m venv env```

Activate the virtual environemnt. Run the command: 
```source env/bin/activate```

### 2. Install required Python packages:
Run the command:
```pip install -r requirements.txt```
This may take a couple minutes to finish running.

### 3. Install Detectron2:
Detectron2 is a framework for implementation of instance segmentation algorithms.

To install on mac, run the command:
```CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'```

More information on installing Detectron2 can be found here: https://detectron2.readthedocs.io/en/latest/tutorials/install.html

### 4. Download model:
Run this command to download the model:
```wget https://github.com/awezmm/FiberAI.models/raw/master/model_final.pth```


## Running FiberAI:

Images have to be of size 1024 x 1024. There is a script provided in this repository (resizer.py) that will divide larger images into equal pieces of 1024 x 1024. For instance, an image of of size 2048 x 2048 will be divided into 4 equal pieces of size 1024 x 1024.

To use resizer.py, first put your original images into a new folder. Next create an empty folder that will hold the output 1024 x 1024 images.
Then, run resizer.py like this: ```python resizer.py fullpath_of_original_folder -d fullpath_of_output_folder```. Make sure to use the full paths of the folders.

Next, run main.py to get fiber measurments. main.py takes 2 arguments: the full path of the input folder and the full path of the output folder. The input folder is the folder that contains 1024 x 1024 images. Create a new empty output folder that will hold output information from main.py
Run main.py like this: ```python main.py -folder fullpath_input_folder -out fullpath_output_folder```.

After you are finished running these scripts, you can run the command: ```deactivate``` to exit from the virtual environment you previously created. To reactivate the virtual environment, run the command: ```source env/bin/activate```.

Note that the current model has been trained on a relatively small dataset, so performance can vary across images. We are in the process of training on a much larger dataset to account for variations in staining patters, brightness levels, fiber lengths, and other fiber patterns that can differ across different labs and different imaging techniques.



