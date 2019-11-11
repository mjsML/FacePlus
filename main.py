import datetime

from core.models.detection import mxnet_detection_model
from core.models.recognition import mxnet_recognition_model

import cv2
import freenect
import numpy as np
import time
import core.configuration as cfg



if __name__ == '__main__':
    mode="RGB"

    fd = mxnet_detection_model(cfg.config.defaultDetectionNetworkPath, 0,
                               scale=.4, gpu=-1, margin=0.15)
    fr = mxnet_recognition_model(cfg.config.defaultRecognitionNetworkPath, 0, [112, 112], -1)

    if mode=="RGB":
        from drivers.genericWebCam import RGBcamera
        cv2.namedWindow('RGBWebcam')
        c=RGBcamera(0)
        while 1:
            input3 = c.getRGBFrame()
            out2 = fd.detect(input3)
            z2 = None
            for i2 in out2:
                z2 = i2

            if z2 is not None:
                x1, y1, w1, h1 = int(z2[0]), int(z2[1]), int(z2[2]), int(z2[3])


                input3 = cv2.rectangle(input3, (x1, y1), (w1, h1), (0, 255, 0), 3)
                ta = 0
                tb = 0
                if (y1 > 0):
                    # sim=cv2.resize(input3[y1: h1, x1: w1],(112,112))
                    faceCrop = input3[y1: h1, x1: w1]
                    #cv2.imshow("face", faceCrop)

                    # sim=np.swapaxes(sim, 2,0)
                    ta = datetime.datetime.now()

                    # out3,M = fa.getLandmarks(sim,z2[:4])
                    tb = datetime.datetime.now()
                    # input3=fa.visualizeLandmarks(input3,out3,M)

                    time = (tb - ta).total_seconds()
                else:
                    faceCrop=[]
                cv2.putText(input3, "Confidence: {0:.2f}".format(z2[4] * 100) + "% ~FPS:" + str(
                    "{0:.2f}".format(1 / fd.inftime)),
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            cv2.imshow("RGBWebcam", input3)
            key = cv2.waitKey(5) & 0xFF
            if key == ord("c"):
                ta = datetime.datetime.now()
                if fr.captureID(faceCrop):
                    tb = datetime.datetime.now()
                    time = (tb - ta).total_seconds()
                    print("captured faceID successfully, it took: " + str(time) + " seconds on CPU!")

                else:
                    print("Failed to capture faceID.")
            elif key == ord("v"):
                ta = datetime.datetime.now()
                if fr.verifyID(faceCrop):
                    tb = datetime.datetime.now()
                    time = (tb - ta).total_seconds()
                    print("verified faceID successfully, it took: " + str(time) + " seconds on CPU!")
                else:
                    print("Failed to verify faceID")
            elif key == 27:
                cv2.destroyAllWindows()
                break

    elif mode=="RGBD":
        from drivers.kinect360 import RGBDCamera
        k = RGBDCamera(0)
        cv2.namedWindow('Detection')
        cv2.namedWindow('DetectionDepth')

        while 1:
            dinput = cv2.cvtColor(k.getDepthFrame(), cv2.COLOR_GRAY2RGB)
            input2 = k.getStackedIRFrame()
            out = fd.detect(input2)
            z = None
            for i in out:
                z = i
            if z is not None:
                x, y, w, h = int(z[0]), int(z[1]), int(z[2]), int(z[3])
                faceCrop = dinput[y: h, x: w]
                #print(x, y, w, h)
                input2 = cv2.rectangle(input2, (x, y), (w, h), (0, 255, 0), 3)
                cv2.putText(input2, "Confidence: {0:.2f}".format(z[4] * 100) + "% ~FPS:" + str(
                    "{0:.2f}".format(1 / fd.inftime)),
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                dinput = cv2.rectangle(dinput, (x, y), (w, h), (0, 255, 0), 3)
                cv2.putText(dinput, "Confidence:{0:.2f}".format(z[4] * 100) + "% ~FPS:" + str(
                    "{0:.2f}".format(1 / fd.inftime)),
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                faceCrop=[]

            key = cv2.waitKey(5) & 0xFF
            if key == ord("c"):
                ta = datetime.datetime.now()
                if fr.captureID(faceCrop):
                    tb = datetime.datetime.now()
                    time = (tb - ta).total_seconds()
                    print("captured faceID successfully, it took: " + str(time) + " seconds on CPU!")

                else:
                    print("Failed to capture faceID.")
            elif key == ord("v"):
                ta = datetime.datetime.now()
                if fr.verifyID(faceCrop):
                    tb = datetime.datetime.now()
                    time = (tb - ta).total_seconds()
                    print("verified faceID successfully, it took: " + str(time) + " seconds on CPU!")
                else:
                    print("Failed to verify faceID")
            elif key == 27:
                cv2.destroyAllWindows()
                break

            cv2.imshow("Detection", input2)
            cv2.imshow("DetectionDepth", dinput)














