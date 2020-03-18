# Webapp Demo For Image Translation Project

## Overview

* This is the Image Translation Project Webapp GUI building with flask, the training scripts of the 2 models are in another repo: <https://github.com/williamlwclwc/Image-Translation-Training-Scripts>

## Test images from testing set

* You can try image translation with images from the testing dataset

## Try with your own images

* You can draw your own segmentation maps using the providing simple drawing tools.

## How to run

* Putting testing images under demo/static/datasets/GT_color
* Putting pretrained models (named as generatorHD, generatorSpade) under demo/static/model
* Create an python virtual environment: $ python -m venv venv
* Activate the virtual environment: $ source ./venv/bin/activate
* Install Dependencies: $ pip install -r requirements.txt
* Run debugging server locally: $ python run.py, and the you can open the webapp demo on <http://127.0.0.1:5000/>
* Use Ctrl+C to stop the server and use "$ deactivate" to quit venv