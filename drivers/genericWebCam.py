from drivers.core.genericRGBCamera import RGBCamera
import cv2
class RGBcamera(RGBCamera):
    def getRGBFrame(self,cameraID):
        cam = cv2.VideoCapture(cameraID)
        _, img = cam.read()
        return img