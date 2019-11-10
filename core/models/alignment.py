# implementation is inspired by the original deploy script from :
# https://github.com/deepinsight/insightface/blob/4a4b8d03fec981912fdef5b3232a37a827cbeed6/deploy/mtcnn_detector.py
import cv2
import numpy as np
import mxnet as mx
from core.utilities import image as img

class mxnet_alignment_model:

    def __init__(self, prefix, epoch, ctx_id=0):
        if ctx_id >= 0:
            ctx = mx.gpu(ctx_id)
        else:
            ctx = mx.cpu()

        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['heatmap_output']
        image_size = (128, 128)
        self.image_size = image_size
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model



    def getLandmarks(self,DetectedFaceImage,bbox):

        input_blob = np.zeros((1, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = DetectedFaceImage
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        alabel = self.model.get_outputs()[-1].asnumpy()[0]
        ret = np.zeros((alabel.shape[0], 2), dtype=np.float32)
        M = img.estimate_trans_bbox(bbox, self.image_size[0], s=0.9)
        for i in range(alabel.shape[0]):
            a = cv2.resize(alabel[i], (self.image_size[1], self.image_size[0]))
            ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
            ret[i] = (ind[1], ind[0])
        return ret, M

    def visualizeLandmarks(self,frame,landmark,M,color=(0, 255, 0)):
        IM = cv2.invertAffineTransform(M)
        for i in range(landmark.shape[0]):
            p = landmark[i]
            point = np.ones((3,), dtype=np.float32)
            point[0:2] = p
            point = np.dot(IM, point)
            landmark[i] = point[0:2]

        for i in range(landmark.shape[0]):
            p = landmark[i]
            point = (int(p[0]), int(p[1]))
            cv2.circle(frame, point, 1, color, 2)
        return frame
