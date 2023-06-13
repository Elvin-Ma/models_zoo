#!/bin/bash
python train.py --img 640 --batch 16 --epochs 100 --data mask.yaml --weights yolov5s.pt
