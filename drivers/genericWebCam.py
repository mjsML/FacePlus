from drivers.core.genericRGBCamera import RGBCamera
import cv2
class RGBcamera(RGBCamera):

    def __init__(self, cameraID):
        self._cameraID = cameraID
        self.cam = cv2.VideoCapture(cameraID)
    def getRGBFrame(self,cameraID=None):

        if cameraID==None:
            cameraID=self._cameraID
        #print(cameraID)


        #print(cameraID)
        _, img = self.cam.read()

        #print(cameraID)
        return img