# 3D-Print-Watcher-PyTorch
A module to watch for 3D print failure using PyTorch and Yolov5. The module requires a local Yolov5 to run, since it needs to be run offline on a Pi. Download it and place it in the module directory, renaming the folder to yolov5_module.

#
Example
```
from PrintWatcherNeural import ImageDetection as ImgDec

# create the detector object
detector = ImgDec()
# call detector and print the result
print(imgDec.detect("test.jpg"))
```

The imgDec returns a tensor with the sigmoidal likelyhood of failure.
