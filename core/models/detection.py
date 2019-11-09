from core.utilities import AnchorConfiguration as ac
# implementation is inspired by the original deploy script from :
# https://github.com/deepinsight/insightface/blob/master/deploy/helper.py
# and enhanced performance / parallelization inspired by:
# https://github.com/1996scarlet/faster-mobile-retinaface/blob/master/face_detector.py



from functools import partial
import cv2
import numpy as np
import mxnet as mx
from core.utilities import generate_anchors, generate_runtime_anchors ,nonlinear_predictions
from numpy import concatenate, float32, block, maximum, minimum, prod
from mxnet.ndarray import concat
import time
from core.models.core.detection import baseDetection

class mxnet_detection_model(baseDetection):

    def __init__(self, prefix, epoch, scale=1., gpu=-1, thd=0.6, margin=0,
                nms_thd=0.4, verbose=False):

        super().__init__(thd=thd, gpu=gpu, margin=margin,
                            nms_thd=nms_thd, verbose=verbose)

        self.scale = scale
        self._rescale = partial(cv2.resize, dsize=None, fx=self.scale,
                                fy=self.scale, interpolation=cv2.INTER_NEAREST)

        self._ctx = mx.cpu() if self.device < 0 else mx.gpu(self.device)
        self._anchors = generate_anchors()
        self._runtime_anchors = {}

        self.model = self._load_model(prefix, epoch)
        self.exec_group = self.model._exec_group



    def _load_model(self, prefix, epoch):
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        model = mx.mod.Module(sym, context=self._ctx, label_names=None)
        model.bind(data_shapes=[('data', (1, 3, 1, 1))],
                   for_training=False)
        model.set_params(arg_params, aux_params)
        return model

    def _get_runtime_anchors(self, height, width, stride, base_anchors):
        key = height, width, stride
        if key not in self._runtime_anchors:
            self._runtime_anchors[key] = generate_runtime_anchors(
                height, width, stride, base_anchors).reshape((-1, 4))
        return self._runtime_anchors[key]

    def _retina_detach(self, out):
        '''

        Parameters
        ----------
        out: map object of staggered scores and deltas.
            scores, deltas = next(out), next(out)

            Each scores has shape [N, A*4, H, W].
            Each deltas has shape [N, A*4, H, W].

            N is the batch size.
            A is the shape[0] of base anchors declared in the fpn dict.
            H, W is the heights and widths of the anchors grid,
            based on the stride and input image's height and width.

        Returns
        -------
        Generator of list, each list has [boxes, scores].

        Usage
        -----
        >>> np.block(list(self._retina_solving(out)))
        '''

        buffer, anchors = out[0].asnumpy(), out[1]
        mask = buffer[:, 4] > self.threshold
        deltas = buffer[mask]
        nonlinear_predictions(anchors[mask], deltas)
        deltas[:, :4] /= self.scale
        return deltas

    def _retina_solve(self):
        out, res, anchors = iter(self.exec_group.execs[0].outputs), [], []

        for fpn in self._anchors:
            scores = next(out)[:, -fpn.scales_shape:,
                     :, :].transpose((0, 2, 3, 1))
            deltas = next(out).transpose((0, 2, 3, 1))

            res.append(concat(deltas.reshape((-1, 4)),
                              scores.reshape((-1, 1)), dim=1))

            anchors.append(self._get_runtime_anchors(*deltas.shape[1:3],
                                                     fpn.stride,
                                                     fpn.base_anchors))

        return concat(*res, dim=0), concatenate(anchors)

    def _retina_forward(self, src):
        '''
        Image preprocess and return the forward results.

        Parameters
        ----------
        src: ndarray
            The image batch of shape [H, W, C].

        scales: list of float
            The src scales para.

        Returns
        -------
        net_out: list, len = STEP * N
            If step is 2, each block has [scores, bbox_deltas]
            Else if step is 3, each block has [scores, bbox_deltas, landmarks]

        Usage
        -----
        >>> out = self._retina_forward(frame)
        '''
        timea = time.perf_counter()

        dst = self._rescale(src).transpose((2, 0, 1))[None, ...]

        if dst.shape != self.model._data_shapes[0].shape:
            self.exec_group.reshape([mx.io.DataDesc('data', dst.shape)], None)

        self.exec_group.data_arrays[0][0][1][:] = dst.astype(float32)
        self.exec_group.execs[0].forward(is_train=False)
        self.inftime = time.perf_counter() - timea


        return self._retina_solve()

    def detect(self, image):
        out = self._retina_forward(image)
        detach = self._retina_detach(out)
        return self.non_maximum_suppression(detach)

