from threading import Thread
from queue import Queue
import time


class VideoGet:
    """
    Class qui permet de lire une vidéo continuellement avec un thread.
    """

    def __init__(self, stream, queueSize=128):
        self.stream = stream
        self.queue = Queue(maxsize=queueSize)
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        time.sleep(2.0)
        while not self.stopped:
            time.sleep(0.001)
            if not self.queue.full():
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    self.stop()
                else:
                    self.queue.put(frame)

    def read(self):
        return self.queue.get()
    
    def more(self):
        return self.queue.qsize() > 0
    
    def queueSize(self):
        return self.queue.qsize()

    def stop(self):
        self.stopped = True