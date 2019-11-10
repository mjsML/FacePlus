from core.models.detection import mxnet_detection_model
import cv2
import freenect
import numpy as np
import time
import core.configuration as cfg
#from drivers.kinect360 import RGBDCamera
from drivers.genericWebCam import RGBcamera

if __name__ == '__main__':





    
    fd = mxnet_detection_model(cfg.config.defaultDetectionNetworkPath, 0,
                             scale=.4, gpu=-1, margin=0.15)
    #cv2.namedWindow('Detection')
    #cv2.namedWindow('DetectionDepth')
    cv2.namedWindow('RGBWebcam')
    #k=RGBDCamera(0)
    c=RGBcamera(0)
    while 1:
        #dinput = cv2.cvtColor(k.getDepthFrame(),cv2.COLOR_GRAY2RGB)
        #input2 = k.getStackedIRFrame()
        print("Got here")
        input3=c.getRGBFrame()


        #out = fd.detect(input2)
        out2=fd.detect(input3)

        #z = None
        #for i in out:
            #z = i
        z2=None
        for i2 in out2:
            z2=i2
        print("Got here")
        # if z is not None:
        #     x, y, w, h = int(z[0]), int(z[1]), int(z[2]), int(z[3])
        #     print(x,y,w,h)
        #     input2 = cv2.rectangle(input2, (x, y), (w, h), (0, 255, 0), 3)
        #     cv2.putText(input2, 'ID: MJ :' + "Confidence: {0:.2f}".format(z[4] * 100) + "% ~FPS:" + str("{0:.2f}".format(1 / fd.inftime)),
        #                 (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #     dinput=cv2.rectangle(dinput, (x, y), (w, h), (0, 255, 0), 3)
        #     cv2.putText(dinput, 'ID: MJ :' + "Confidence:{0:.2f}".format(z[4] * 100) + "% ~FPS:" + str("{0:.2f}".format(1 / fd.inftime)),
        #                 (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if z2 is not None:
            x1, y1, w1, h1 = int(z2[0]), int(z2[1]), int(z2[2]), int(z2[3])
            input3 = cv2.rectangle(input3, (x1, y1), (w1, h1), (0, 255, 0), 3)
            cv2.putText(input3, 'ID: MJ :' + "Confidence: {0:.2f}".format(z2[4] * 100) + "% ~FPS:" + str(
                "{0:.2f}".format(1 / fd.inftime)),
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.imshow("Detection", input2)
        #cv2.imshow("DetectionDepth", dinput)
        cv2.imshow("RGBWebcam", input3)
        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            break