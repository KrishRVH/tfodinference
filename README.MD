# TensorFlow Object Detection
Oklahoma State University | ECEN 4273 | Project 2

Team members:
		
		Krish Ravi
		
		Jacobo Rosillo

Repository link: https://github.com/KrishRVH/tfodinference

---
		
## Installation
Pip install the dependency packages

Put your label_map.pbtxt in ./labels

Put your exported saved models in ./models

Put the videos to conduct inference on in ./testvideos

## Usage 
Run ``` python infer.py ``` in the directory with this ReadMe

Output videos with bounding boxes will be in ./resultvideos 

Change line 16 to change which model to use for inference, and change line 19 to select a different testvideo.

---

## Dependencies
tensorflow 2.10.1

Python 3.9.13

tensorflow-object-detection-api 0.1.1

opencv-python 4.6.0.66

numpy 1.23.4