import cv2  # OpenCV library
import numpy as np
from analyzers.analyseContour import AnalyseContour
from analyzers.contour import Contour


class AnalyseDerive(AnalyseContour):
    """
    Class qui mesure la taille de la dérive qui sort de l'eau.
    """

    def __init__(self, x1, y1, x2, y2):
        super().__init__(x1, y1, x2, y2)
        self.qualityLimit = 24
    
    def compute(self, frame):

        cropFrame = frame[self.y1:self.y2, self.x1:self.x2]

        # Conversion en noir et blanc et floutage
        gray_img_ecume = cv2.cvtColor(cropFrame, cv2.COLOR_BGR2GRAY)
        gray_img_ecume = cv2.GaussianBlur(gray_img_ecume, (7, 7), 0)

        # Conversion de l'image grisé en image binaire, 2 couleurs de pixel, noir et blanc, plus de gris (binarization opencv)
        ret, binary_img_ecume = cv2.threshold(gray_img_ecume, 80, 255, cv2.THRESH_BINARY)

        # Détection des contours
        # La variable de hiérarchie contient des informations sur la relation entre chaque contour. (si un contour est dans un contour)
        contours_list_ecume, hierarchy = cv2.findContours(binary_img_ecume, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #décalage de la zone d'analyse pour détecter l'embrun en amont
        cropFrameEmbrun = frame[self.y1:self.y2, int(self.x1 + (self.x2 - self.x1)/2):int(self.x2 + (self.x2 - self.x1)/2)]
        qualityIndex = self.embrunDetection.detection(cropFrameEmbrun)

        if contours_list_ecume:

            # Récupère le contour le plus grand
            cEcume = max(contours_list_ecume, key=cv2.contourArea)

            right = tuple(cEcume[cEcume[:, :, 0].argmax()][0]) # valeur x maximale parmis tous les x
            rightPoints = np.where(cEcume[:, :, 0] == right[0]) # cherche toutes les valeurs égales à la valeur la plus à droite parmis les x

            # Si on a bien 2 coordonnées
            if len(rightPoints[0]) == 2 :

                # premier crochet -> tableau contenant les indices des coordonnées les plus à droite
                # deuxième crochet -> choisir entre les indices (normalement 2 indices en tout)
                # troisième crochet -> tout le temps 0
                # quatrième crochet -> choisis entre x et y
                x1 = (cEcume[rightPoints[0][0]][0])[0] + self.x1
                y1 = (cEcume[rightPoints[0][0]][0])[1] + self.y1
                x2 = (cEcume[rightPoints[0][1]][0])[0] + self.x1
                y2 = 268
                y2Test = (cEcume[rightPoints[0][1]][0])[1] + self.y1

                # Si y2Test trop grand -> mauvaise mesure
                if y2Test < 275 :

                    # Décalage des coordonnées du contour pour correspondre sur l'image original (frame)
                    cEcume = cEcume + (self.x1, self.y1)


                    return Contour(y1 - y2, cEcume, (x1, y2), (x2, y1), qualityIndex)

        return Contour(None, None, None, None, qualityIndex)