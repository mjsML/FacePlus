from core.models.detection import mxnet_detection_model
import cv2
import freenect
import numpy as np
import time
if __name__ == '__main__':
    import sys
    from numpy import prod

    FRAME_SHAPE = 480, 640, 3
    BUFFER_SIZE = prod(FRAME_SHAPE)

    # read = sys.stdin.buffer.read
    # write = sys.stdout.buffer.write
    # camera = iter(partial(read, BUFFER_SIZE), b'')
    def getdepth(depth):
        """Converts depth into a 'nicer' format for display

        This is abstracted to allow for experimentation with normalization

        Args:
            depth: A numpy array with 2 bytes per pixel

        Returns:
            A numpy array that has been processed with unspecified datatype
        """
        # np.clip(depth, 0, 2 ** 10 - 1, depth)
        depth >>= 3
        depth = depth.astype(np.uint8)
        return depth


    def get_depth(ind=0):
        return getdepth(freenect.sync_get_depth(ind)[0])


    def get_video_IR():
        array, _ = freenect.sync_get_video(0, freenect.VIDEO_IR_10BIT)
        array = getdepth(array)
        return array
    # input = cv2.imread("/Users/mj/libfreenect/wrappers/python/image1.png")
    path="/Users/mj/Documents/Code/FacePlus/weights/detection"
    fd = mxnet_detection_model(path+"/16and32", 0,
                             scale=.4, gpu=-1, margin=0.15)
    cv2.namedWindow('Detection')
    cv2.namedWindow('DetectionDepth')
    while 1:
        input = get_video_IR()
        input2 = np.stack((input,) * 3, axis=-1)
        dinput = get_depth()
        out = fd.detect(input2)
        #out2 = dinput
        # print(out)
        z = None
        for i in out:
            z = i
        #print(z)
        if z is not None:
            x, y, w, h = int(z[0]), int(z[1]), int(z[2]), int(z[3])
            input2 = cv2.rectangle(input, (x, y), (w, h), (0, 255, 0), 3)
            cv2.putText(input2, 'ID: MJ :' + "Confidence: {0:.2f}".format(z[4] * 100) + "% ~FPS:" + str("{0:.2f}".format(1 / fd.inftime)),
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            dinput=cv2.rectangle(dinput, (x, y), (w, h), (0, 255, 0), 3)
            cv2.putText(dinput, 'ID: MJ :' + "Confidence:{0:.2f}".format(z[4] * 100) + "% ~FPS:" + str("{0:.2f}".format(1 / fd.inftime)),
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.imshow("Detection", input2)
        cv2.imshow("DetectionDepth", dinput)
        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            break