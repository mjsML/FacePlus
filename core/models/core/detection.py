# implementation is inspired by the original deploy script from :
# https://github.com/deepinsight/insightface/blob/master/deploy/helper.py
# and enhanced performance / parallelization inspired by:
# https://github.com/1996scarlet/faster-mobile-retinaface/blob/master/face_detector.py
import numpy as np

from core.utilities import image
from queue import Queue
class baseDetection():
    def __init__(self, *, thd, gpu, margin, nms_thd, verbose):
        self.threshold = thd
        self.nms_threshold = nms_thd
        self.device = gpu
        self.margin = margin

        self._queue = Queue(2)
        self.write_queue = self._queue.put_nowait
        self.read_queue = iter(self._queue.get, b'')


    def margin_clip(self, b):
        margin_x = (b[2] - b[0]) * self.margin
        margin_y = (b[3] - b[1]) * self.margin

        b[0] -= margin_x
        b[1] -= margin_y
        b[2] += margin_x
        b[3] += margin_y

        return np.clip(b, 0, None, out=b)

    def non_maximum_suppression(self,boxes,mode="Union"):

        return image.nonMaximumSuppression(boxes,self.nms_threshold,mode)

    def filter_boxes(boxes, min_size, max_size=-1):
        image.filter_boxes(boxes, min_size, max_size)