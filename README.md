# Video Recognition Technology: Enhancing Golf Swing Performance for Players of All Levels

## Introduction

Since the COVID-19 pandemic, golf has become one of the fastest-growing sports in America, with 3.3 million Americans playing golf for the first time in 2022. Golf's overall reach is estimated to be 119 million people, with approximately 41.1 million Americans playing golf either on or off-course in 2022. The golf industry's economic activity is valued at $84.1 billion as of 2016, presenting significant economic and business opportunities.

The golf swing is a rigorous, full-body motion that requires fluid coordination from head to toe to achieve optimal results. To perfect their swing, golfers need to devote substantial time to training, meticulous attention to biomechanical detail, high levels of skill, sheer physical ability, and substantial practice. Most amateur golfers turn to learning and practice methods to improve their performance, typically analyzing and adjusting their golf swing via one of two common approaches: golf instructors and technology.

This research focuses on using computer vision and deep learning techniques to streamline amateur golfer swing analysis via golf swing sequencing, extending pre-existing research by McNally, et al. Golf swing sequencing identifies the eight key frames within swing videos: Address, Toe-up, Mid-backswing, Top, Mid-downswing, Impact, Mid-follow-through, Finish. After identifying the key events/images of a golf swing, this research uses MediaPipe's Pose human pose estimation technology to extract key golfer biomechanical features at each event, facilitating more accessible opportunities for improvement. By comparing these features to professional golfers and/or previous swings, golfers can better analyze and adjust their swing to improve their performance.

## Golf Swing Sequencing

This repository contains PyTorch implementions for testing a series of deep learning models for performing golf swing sequencing.

Each model is trained on split 1 **without any data augmentation** on processed videos from the golfdb database. The processed video inputs for each model are stored in the data directory under their respective model name (e.g. the optical flow processed videos are under data/videos_160_optical_flow). The data folder also contains python scripts for converting the raw golfdb videos to their processed form (e.g. data/optical_flow.py).

The python scripts to train and evaluate each model are located within their respective model directory (e.g. the SwingNet model trained and evaluated on optical flow videos is located in optical_flow_model).

### How to run

#### Model Training
Within each model folder, there is a Jupyter Notebook titled run.ipynb. This code is meant to be executed on Google Colab. To reproduce the model results, set the notebook runtime type with the following parameters: Hardware accelerator = GPU, Runtime shape = High-RAM.

#### Golf Swing Sequencing Application

1. Follow this [tutorial](https://code.visualstudio.com/docs/python/python-tutorial) to install VSCode & Python.

2. Clone this repository or copy the swing_sequence_application folder into local file directory and set this as working directory.

3.
```
# Create a virtual environment in the .venv subdirectory
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

4. 
```
pip install requirements.txt
```

5. 
```
shiny run --reload
```

If completed successfully, you should be able to open the application running on your local machine:

https://media.github.iu.edu/user/16772/files/c258ab01-dc25-4a8b-820c-f757bc12205a

### Results

Executing the run notebooks for each model should produce the following out-of-sample Percentage of Correct Events (PCE) scores:

+ Original SwingNet Model: 71.5%
+ **Optical Flow Model: 79.6%**
+ Background Removal Model: 69.0%
+ Human Pose Model: 69.5%

The resulting best model can predict the golf swing sequencing events of new videos (along with their confidence):

<img src="/images/191_optical_flow_swing_sequence.jpg" alt="Alt text" title="Optical Flow Model Predictions on 191.mp4 (out-of-sample in split 2)">

## Dependencies
* [PyTorch](https://pytorch.org/)

## References

The baseline code edited to perform the deep learning modeling experiments in this repo is sourced from wmcnally/golfdb:
```
@InProceedings{McNally_2019_CVPR_Workshops,
author = {McNally, William and Vats, Kanav and Pinto, Tyler and Dulhanty, Chris and McPhee, John and Wong, Alexander},
title = {GolfDB: A Video Database for Golf Swing Sequencing},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```
