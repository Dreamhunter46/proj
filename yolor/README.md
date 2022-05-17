This repository is an implementation of yolor in addition to mediapipe for human with a certain pose detection using webcam on google colab. 
Largely based on https://github.com/ultralytics/yolov5 & https://github.com/chunkhai96/yolov5 

**All code and models are under active development, and are subject to modification or deletion without notice.** Use at your own risk.


## Requirements

Python 3.7 or later with all `requirements.txt` dependencies installed, including `torch >= 1.5`. To install run:
```bash
$ pip install -qr requirements.txt
```
and then to run:
```bash
%run main.py --source 0 --weights yolor_p6.pt --conf 0.4 --machine colab --show False
```
machine indicates if the notebook is runned on colab and show indicates to show or not the detection on js.

or use the colab notebook: YOLOr & Mediapipe detection.ipynb

