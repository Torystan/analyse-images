import cv2  # OpenCV library
import numpy as np
import math
from analyzers.analyseContour import AnalyseContour
from analyzers.contour import Contour


class AnalyseMousse(AnalyseContour):
    """
    Class qui mesure la taille de la dérive qui sort de l'eau.
    """

    def __init__(self, x1, y1, x2, y2, qualityLimit):
        super().__init__(x1, y1, x2, y2, qualityLimit)
    
    def compute(self, frame):

        cropFrame = frame[self.y1:self.y2, self.x1:self.x2]

        # Conversion en noir et blanc et floutage
        gray_img_mousse = cv2.cvtColor(cropFrame, cv2.COLOR_BGR2GRAY)
        gray_img_mousse = cv2.GaussianBlur(gray_img_mousse, (7, 7), 0)

        # Calcul de la médiane des pixels d'une portion de l'image près du coin en haut à droite
        median_pix = 0.8 * np.median(gray_img_mousse[0:math.ceil((self.y2-self.y1)/6), math.floor((self.x2-self.x1)*0.8):self.x2-self.x1])

        # Conversion de l'image grisé en image binaire, 2 couleurs de pixel, noir et blanc, plus de gris (binarization opencv)
        ret, binary_img_mousse = cv2.threshold(gray_img_mousse, median_pix, 255, cv2.THRESH_BINARY)

        # Détection des contours
        # La variable de hiérarchie contient des informations sur la relation entre chaque contour. (si un contour est dans un contour)
        contours_list_mousse, hierarchy = cv2.findContours(binary_img_mousse, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #décalage de la zone d'analyse pour détecter l'embrun en amont
        cropFrameEmbrun = frame[self.y1:self.y2, int(self.x1 + (self.x2 - self.x1)/2):int(self.x2 + (self.x2 - self.x1)/2)]
        qualityIndex = self.embrunDetection.detection(cropFrameEmbrun)

        if contours_list_mousse:

            # Trouver le contour le plus proche du coin en haut à droite (pour éviter les contours parasites)
            contourMousse = None
            for c in contours_list_mousse:
                # Si un contour est à moins de 15 pixel du point (coin en haut à droite de la zone d'analyse)
                
                if abs(cv2.pointPolygonTest(c, (self.x2 - self.x1, 1), True)) < 15 and cv2.contourArea(c) > 100:
                    contourMousse = c

            if contourMousse is None:
                return Contour(None, None, None, None, qualityIndex)

            right = tuple(contourMousse[contourMousse[:, :, 0].argmax()][0]) # valeur x maximale parmis tous les x
            rightPoints = np.where(contourMousse[:, :, 0] == right[0]) # cherche toutes les valeurs égales à la valeur la plus à droite parmis les x

            # Si on a bien 2 coordonnées
            if len(rightPoints[0]) >= 2 :

                # premier crochet -> tableau contenant les indices des coordonnées les plus à droite
                # deuxième crochet -> choisir entre les indices (normalement 2 indices en tout)
                # troisième crochet -> tout le temps 0
                # quatrième crochet -> choisis entre x et y
                x1 = (contourMousse[rightPoints[0][0]][0])[0] + self.x1
                y1 = (contourMousse[rightPoints[0][0]][0])[1] + self.y1
                x2 = (contourMousse[rightPoints[0][1]][0])[0] + self.x1
                y2 = self.y1

                # Récupère la coordonnée y du point le plus haut, mais pas besoin si le coin en haut à droite de la zone est bien placé sur la coque
                #y2 = (contourMousse[rightPoints[0][1]][0])[1] + self.y1

                # Décalage des coordonnées du contour pour correspondre sur l'image original (frame)
                contourMousse = contourMousse + (self.x1, self.y1)

                return Contour(y1 - y2, contourMousse, (x1, y2), (x2, y1), qualityIndex)

        return Contour(None, None, None, None, qualityIndex)