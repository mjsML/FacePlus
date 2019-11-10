# implementation is inspired by the original deploy script from :
#https://github.com/deepinsight/insightface/blob/master/deploy/face_model.py

import numpy as np
import mxnet as mx
import cv2
import sklearn
from core.utilities import image as img
from core.utilities import get_model
import core.configuration as cfg
from sklearn import preprocessing
class mxnet_recognition_model():
    def __init__(self, prefix, epoch, imsize,ctx_id=0):
        # TODO move all args to config file
        if ctx_id >= 0:
            ctx = mx.gpu(ctx_id)
        else:
            ctx = mx.cpu()
        self.ID=None

        image_size = imsize[0],imsize[1]
        self.model = None

        self.model = get_model(ctx, image_size, cfg.config.defaultRecognitionNetworkPath, 'fc1')


        self.threshold = cfg.config.defaultRecognitionThreshold
        #TODO move all args to config file
        self.det_minsize = 50
        self.det_threshold = [0.6, 0.7, 0.8]

        self.image_size = image_size


    def get_input(self, face_img,bbox=None):

        nimg = img.preprocess(face_img, bbox, None, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        return aligned

    def get_feature(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = preprocessing.normalize(embedding).flatten()
        return embedding

    def getDistance(self,f1,f2):
        return np.sum(np.square(f1 - f2))

    def storeID(self,f1):
        self.ID=f1

    def verifyID(self,image):
        if len(self.ID) >0:
            f1=self.get_input(image)
            f1=self.get_feature(f1)
            if self.getDistance(f1, self.ID)<=cfg.config.defaultRecognitionDistanceThreshold:
                return True
            else:
                return False
        return False

    def captureID(self,image):
        #try:

            alignedImage=self.get_input(image)
            f1=self.get_feature(alignedImage)
            self.storeID(f1)
            return True
        #except:
            print("Boom!")
            return False



