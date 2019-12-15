#!/bin/bash
echo "Creating directories"
mkdir data
echo "Fetching YOLO data"
wget https://pjreddie.com/media/files/yolov3.weights
mv yolov3.weights ./yolo/
