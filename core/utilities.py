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
import mxnet as mx
import math
import cv2
from skimage import transform as stf

class image():
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

    @staticmethod
    def transform( data, center, output_size, scale, rotation):
        scale_ratio = float(output_size) / scale
        rot = float(rotation) * np.pi / 180.0
        t1 = stf.SimilarityTransform(scale=scale_ratio)
        cx = center[0] * scale_ratio
        cy = center[1] * scale_ratio
        t2 = stf.SimilarityTransform(translation=(-1 * cx, -1 * cy))
        t3 = stf.SimilarityTransform(rotation=rot)
        t4 = stf.SimilarityTransform(translation=(output_size / 2, output_size / 2))
        t = t1 + t2 + t3 + t4
        trans = t.params[0:2]
        cropped = cv2.warpAffine(data, trans, (output_size, output_size), borderValue=0.0)
        return cropped, trans

    @staticmethod
    def transform_pt( pt, trans):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(trans, new_pt)
        return new_pt[:2]

    @staticmethod
    def gaussian( img, pt, sigma):
        # Draw a 2D gaussian
        assert (sigma >= 0)
        if sigma == 0:
            img[pt[1], pt[0]] = 1.0
        return True

        # Check that any part of the gaussian is in-bounds
        ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
        br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
        if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
                br[0] < 0 or br[1] < 0):
            # If not, just return the image as is

            return False

        # Generate gaussian
        size = 6 * sigma + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])

        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return True

    @staticmethod
    def estimate_trans_bbox( face, input_size, s=2.0):
        w = face[2] - face[0]
        h = face[3] - face[1]
        wc = int((face[2] + face[0]) / 2)
        hc = int((face[3] + face[1]) / 2)
        im_size = max(w, h)
        scale = input_size / (max(w, h) * s)
        M = [
            [scale, 0, input_size / 2 - wc * scale],
            [0, scale, input_size / 2 - hc * scale],
        ]
        M = np.array(M)
        return M

    @staticmethod
    def parse_lst_line(line):
      vec = line.strip().split("\t")
      assert len(vec)>=3
      aligned = int(vec[0])
      image_path = vec[1]
      label = int(vec[2])
      bbox = None
      landmark = None
      #print(vec)
      if len(vec)>3:
        bbox = np.zeros( (4,), dtype=np.int32)
        for i in xrange(3,7):
          bbox[i-3] = int(vec[i])
        landmark = None
        if len(vec)>7:
          _l = []
          for i in xrange(7,17):
            _l.append(float(vec[i]))
          landmark = np.array(_l).reshape( (2,5) ).T
      #print(aligned)
      return image_path, label, bbox, landmark, aligned

    @staticmethod
    def read_image(img_path, **kwargs):
      mode = kwargs.get('mode', 'rgb')
      layout = kwargs.get('layout', 'HWC')
      if mode=='gray':
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
      else:
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
        if mode=='rgb':
          #print('to rgb')
          img = img[...,::-1]
        if layout=='CHW':
          img = np.transpose(img, (2,0,1))
      return img

    @staticmethod
    def preprocess(img, bbox=None, landmark=None, **kwargs):
      if isinstance(img, str):
        img = image.read_image(img, **kwargs)
      M = None
      image_size = []
      str_image_size = kwargs.get('image_size', '')
      if len(str_image_size)>0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size)==1:
          image_size = [image_size[0], image_size[0]]
        assert len(image_size)==2
        assert image_size[0]==112
        assert image_size[0]==112 or image_size[1]==96
      if landmark is not None:
        assert len(image_size)==2
        src = np.array([
          [30.2946, 51.6963],
          [65.5318, 51.5014],
          [48.0252, 71.7366],
          [33.5493, 92.3655],
          [62.7299, 92.2041] ], dtype=np.float32 )
        if image_size[1]==112:
          src[:,0] += 8.0
        dst = landmark.astype(np.float32)

        tform = stf.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]


      if M is None:
        if bbox is None: #use center crop
          det = np.zeros(4, dtype=np.int32)
          det[0] = int(img.shape[1]*0.0625)
          det[1] = int(img.shape[0]*0.0625)
          det[2] = img.shape[1] - det[0]
          det[3] = img.shape[0] - det[1]
        else:
          det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
          ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
      else: #do align using landmark
        assert len(image_size)==2
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
        return warped




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

def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):

  sym, arg_params, aux_params = mx.model.load_checkpoint(model_str, 0)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model