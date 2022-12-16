import cv2
import numpy as np


class EmbrunDetection():
    """
    Class qui permet de détecter si de l'embrun empêche les mesures.
    """

    def detection(self, frame):
        """
        Fonction qui renvoie l'écart type de l'histogramme des pixels,
        plus l'écart type est bas, plus il y a des l'embrun.
  
        Parameters:
            frame (OutputArray): Image à analyser.
          
        Returns:
            int: écart type de l'histogramme des pixels.
        """

        tabValues = []
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])

        for i in range(0, len(hist)):
            for j in range(0, int(hist[i][0])):
                tabValues.append(i)

        stdDeviation = (int(np.std(tabValues)))

        return stdDeviation
