from drivers.core import genericRGBCamera, genericDepthCamera, genericIRCamera
import cv2
import freenect
import numpy as np


class RGBDCamera(genericRGBCamera.RGBCamera, genericDepthCamera.depthCamera, genericIRCamera.IRCamera):


    def __init__(self, cameraID):
        self._cameraID = cameraID

    def _getActualData(self,depth):
        depth >>= 3
        depth = depth.astype(np.uint8)
        return depth

    def getRGBFrame(self, cameraID=None):
        if cameraID==None:
            cameraID=self._cameraID
        return freenect.sync_get_video(cameraID)[0]

    def getDepthFrame(self, cameraID=None):
        if cameraID==None:
            cameraID=self._cameraID
        return self._getActualData(freenect.sync_get_depth(cameraID)[0])

    def getIRFrame(self, cameraID=None):
        if cameraID==None:
            cameraID=self._cameraID
        frame = self._getActualData(freenect.sync_get_video(cameraID, freenect.VIDEO_IR_10BIT)[0])
        return frame

    def getStackedIRFrame(self, cameraID=None):
        if cameraID==None:
            cameraID=self._cameraID
        return np.stack((self.getIRFrame(cameraID),) * 3, axis=-1)

    def getBGRFrame(self, cameraID=None):
        if cameraID==None:
            cameraID=self._cameraID
        return self.getRGBFrame(cameraID)[:, :, ::-1]
