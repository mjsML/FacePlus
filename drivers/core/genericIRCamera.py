from abc import ABC, abstractmethod
class IRCamera(ABC):
    def getIRFrame(self,cameraID):
        pass