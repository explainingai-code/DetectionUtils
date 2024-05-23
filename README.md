# DetectionUtils
This repository provides implementation of utilities for Object Detection

As of today the repo provides code for the following:
* Mean Average Precision Computation(Area Method and 11 points interpolation)
* IOU calculation
* Non-Maximum Suppression

This code is an attempt to provide the implementations of these metrics in the simplest manner (for understanding the computations happening better).

It is NOT the most efficient way to implement them.
For efficient implementations, I would urge you to look at object detection libraries like mmdetection's implementation.

## Mean Average Precision, IOU and NMS Tutorial Video
<a href="https://www.youtube.com/watch?v=duBGmrxNHS8">
   <img alt="Mean Average Precision Tutorial" src="https://github.com/explainingai-code/DetectionUtils/assets/144267687/8fcc2f6a-e2b6-40ce-aa56-24a715f7bf4b"
   width="400">
</a>

## Computing Mean Average Precision
* For computing map one can use the `compute_map` method in the `compute_ap.py` script
* `evaluate_map` method within that script provides comments on how to take predictions and ground truth and set up them in a format that `compute_map` expects.

___  

