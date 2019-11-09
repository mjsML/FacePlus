# implementation is inspired by the original deploy script from :
# https://github.com/deepinsight/insightface/blob/master/deploy/helper.py
# and enhanced performance / parallelization inspired by:
# https://github.com/1996scarlet/faster-mobile-retinaface/blob/master/face_detector.py
#

import math
import cv2
import numpy as np
import sys
from numpy import zeros, concatenate, float32, tile, repeat, arange, exp, maximum, minimum
class image:
    @staticmethod
    def nonMaximumSuppression(boxes,threshold,mode="Union"):
            """
            non max suppression

            Parameters:
            ----------
                box: numpy array n x 5
                    input bbox array
                threshold: float number
                    threshold of overlap
                mode: float number
                    how to compute overlap ratio, 'Union' or 'Min'
            Returns:
            -------
                index array of the selected bbox
            """

            #if empty list return
            if boxes.size == 0:
                return []
            #if list is integer convert to floats

            if boxes.dtype.kind=="i":
                boxes=boxes.astype("float")
            
            x1, y1, x2, y2, scores = boxes.T

            areas = (x2-x1+1) * (y2-y1+1)
            
            idxs=np.argsort(scores)[::-1]

            while idxs.size>0:

                pick, last = idxs[0], idxs[1:]
                yield boxes[pick]

                xx1 = maximum(x1[pick], x1[last])
                yy1 = maximum(y1[pick], y1[last])
                xx2 = minimum(x2[pick], x2[last])
                yy2 = minimum(y2[pick], y2[last])

                width = maximum(0.0, xx2 - xx1 + 1)
                height = maximum(0.0, yy2 - yy1 + 1)

                intersection = width * height
                if mode=="Min":
                    overlap=intersection/np.minimum(areas[pick],areas[idxs[last]])
                else:
                    overlap = intersection / (areas[pick] + areas[last] - intersection )

                idxs = last[overlap < threshold]




    @staticmethod
    def filter_boxes(boxes, min_size, max_size=-1):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        if max_size > 0:
            boxes = np.where(minimum(ws, hs) < max_size)[0]
        if min_size > 0:
            boxes = np.where(maximum(ws, hs) > min_size)[0]
        return boxes


class AnchorConfiguration:
    def __init__(self, *,  stride, scales,
                 base_size=16, ratios=(1., ), dense_anchor=False):
        self.stride = stride
        self.scales = np.array(scales)
        self.scales_shape = self.scales.shape[0]

        self.base_size = base_size
        self.ratios = np.array(ratios)
        self.dense_anchor = dense_anchor

        self.base_anchors = self._generate_anchors()

    def _generate_anchors(self):
        base_anchor = np.array([1, 1, self.base_size, self.base_size]) - 1
        ratio_anchors = self._ratio_enum(base_anchor)

        anchors = np.vstack([self._scale_enum(ratio_anchors[i, :])
                             for i in range(ratio_anchors.shape[0])])

        if self.dense_anchor:
            assert self.stride % 2 == 0
            anchors2 = anchors.copy()
            anchors2[:, :] += int(self.stride/2)
            anchors = np.vstack((anchors, anchors2))

        return anchors

    def _anchorProperties(self, anchor):
        """
        Return width, height, x center, and y center for an anchor (window).
        """

        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    def _makeAnchors(self, ws, hs, x_ctr, y_ctr):
        """
        Given a vector of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr), output a set of anchors (windows).
        """

        ws = ws[:, None]
        hs = hs[:, None]
        anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                             y_ctr - 0.5 * (hs - 1),
                             x_ctr + 0.5 * (ws - 1),
                             y_ctr + 0.5 * (hs - 1)))
        return anchors

    def _ratio_enum(self, anchor):
        """
        Enumerate a set of anchors for each aspect ratio wrt an anchor.
        """

        w, h, x_ctr, y_ctr = self._anchorProperties(anchor)
        size = w * h
        size_ratios = size / self.ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * self.ratios)
        anchors = self._makeAnchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def _scale_enum(self, anchor):
        """
        Enumerate a set of anchors for each scale wrt an anchor.
        """

        w, h, x_ctr, y_ctr = self._anchorProperties(anchor)
        ws = w * self.scales
        hs = h * self.scales
        anchors = self._makeAnchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def __repr__(self):
        return f'Stride: {self.stride}'


anchor_config = [
    AnchorConfiguration(stride=32, scales=(32, 16)),
    AnchorConfiguration(stride=16, scales=(8, 4)),
]


def generate_runtime_anchors(height, width, stride, base_anchors):
    A = base_anchors.shape[0]

    all_anchors = zeros((height*width, A, 4), dtype=float32)

    rw = tile(arange(0, width*stride, stride),
              height).reshape(-1, 1, 1)
    rh = repeat(arange(0, height*stride, stride),
                width).reshape(-1, 1, 1)

    all_anchors += concatenate((rw, rh, rw, rh), axis=2)
    all_anchors += base_anchors

    return all_anchors


def generate_anchors(dense_anchor=False, cfg=anchor_config):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    return sorted(cfg, key=lambda x: x.stride, reverse=True)


def nonlinear_predictions(boxes, box_deltas):
    if boxes.size:
        ctr_x, ctr_y, widths, heights = boxes.T
        widths -= ctr_x
        heights -= ctr_y

        widths += 1.0
        heights += 1.0

        dx, dy, dw, dh, _ = box_deltas.T

        dx *= widths
        dx += ctr_x
        dx += 0.5 * widths

        dy *= heights
        dy += ctr_y
        dy += 0.5 * heights

        exp(dh, out=dh)
        dh *= heights
        dh -= 1.0
        dh *= 0.5

        exp(dw, out=dw)
        dw *= widths
        dw -= 1.0
        dw *= 0.5

        dx -= dw
        dw += dw
        dw += dx

        dy -= dh
        dh += dh
        dh += dy
