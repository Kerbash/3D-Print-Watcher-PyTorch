import numpy as np

from .recognitionmodule import RecognitionModule as recMod
from .failuredetector import FailureDetector as failMod
from pathlib import Path

class ImageDetection:
    """
    Image detection class

    """
    def __init__(self, device = 'cpu'):
        # get the current module path
        root = str(Path(__file__).parent.resolve())
        # get YOLO path
        rm_path = root + "\\ultralytics_yolov5_master"
        fm_path = root + "\\failDecModule.pt"

        self.recMod = recMod(rm_path, root + "\\yolo3Dprint.pt", device)
        self.failMod = failMod(fm_path)

    def detect(self, image):
        """
        Detect whether the image passed is a failed 3d print

        :param image: path to the image file
        :return: -
        """
        # load only one image at a time
        if type(image) is list:
            raise ValueError("Insert only one image at a time please")

        # get the result in a list
        raw_result = self.recMod.detect(image).pandas().xyxy[0].iloc

        # convert it into an array to be fed into the CNN
        # 4 classes (0: extruder, 1: nozzle, 2: print, 3: mistake(warp, spaghetti, etc))
        # 5 items each (enough room for 5 items)
        # 6 elements [class #, confidence value, xmin, ymin, xmax, ymax]
        result_array = np.zeros((4, 5, 6))

        key = {0: 0, 1: 0, 2: 0, 3: 0}

        for item in raw_result:
            # find out what kind of item it is
            cl = item["class"]
            if (cl > 3):
                location = 3
            else:
                location = cl
            # if there is more room
            if key[location] < 5:
                result_array[location][key[location]] = [cl, item["confidence"], item["xmin"], item["ymin"], item["xmax"], item["ymax"]]
                key[location] += 1

        return self.failMod.detect(result_array)