import cv2

class VideoStream:
    def __init__(self, source=0):
        self.stream = cv2.VideoCapture(source)

    def read_frame(self):
        ret, frame = self.stream.read()
        return frame if ret else None

    def stop(self):
        self.stream.release()
