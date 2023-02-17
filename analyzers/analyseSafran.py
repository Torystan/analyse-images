import cv2  # OpenCV library
from analyzers.analyseContour import AnalyseContour
from analyzers.contour import Contour
import cv2
import numpy as np


class AnalyseSafran(AnalyseContour):
    """
    Class qui mesure la taille du safran qui sort de l'eau.

    Attributs:
        xRefPoint (int): coordonnée x du point de référence pour démarer la mesure, correspond au point le plus haut du safran.
        yRefPoint (int): coordonnée y du point de référence pour démarer la mesure, correspond au point le plus haut du safran.
    """

    def __init__(self, x1, y1, x2, y2, x1RefPoint, y1RefPoint, x2RefPoint, y2RefPoint, qualityLimit):
        super().__init__(x1, y1, x2, y2, qualityLimit)
        self.x1RefPoint = x1RefPoint - x1
        self.y1RefPoint = y1RefPoint - y1
        self.x2RefPoint = x2RefPoint - x1
        self.y2RefPoint = y2RefPoint - y1

    def compute(self, frame):
        """
        Méthode qui mesure la taille du safran qui sort de l'eau.
        """

        m = (self.y2RefPoint - self.y1RefPoint) / (self.x2RefPoint - self.x1RefPoint)
        p = self.y1RefPoint - m * self.x1RefPoint

        cropFrame = frame[self.y1:self.y2, self.x1:self.x2]

        qualityIndex = self.embrunDetection.detection(cropFrame)

        # Conversion en noir et blanc et floutage
        gray_img_safran = cv2.cvtColor(cropFrame, cv2.COLOR_BGR2GRAY)
        gray_img_safran = cv2.GaussianBlur(gray_img_safran, (3, 7), 0)
        

        # dessins de toutes les bordures
        median_pix = np.median(gray_img_safran)
        lower = int(max(0, 0.6*median_pix))
        upper = int(min(255, 1.25*median_pix))
        edged = cv2.Canny(gray_img_safran, lower, upper)

        # Détection des contours
        # La variable de hiérarchie contient des informations sur la relation entre chaque contour. (si un contour est dans un contour)
        contours_list_safran, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours_list_safran:
            contourSafran = None

            for c in contours_list_safran:
                # Si un contour est à moins de 15 pixel du point (point sur le bord avant du safran)
                if abs(cv2.pointPolygonTest(c, (self.x1RefPoint + 1, self.y1RefPoint + 3), True)) < 15:
                    contourSafran = c
                    tOffSetSafran = (self.x1, self.y1)

                    points = []

                    # Regarde si les points correspondent plus ou moins à l'équation du safran
                    # Equation qui représente la droite au niveau du safran
                    for point in c:
                        x = point[0][0]
                        y = point[0][1]
                        # Résultat de l'équation
                        resultEquation = m * x - y + p

                        if resultEquation > -15 and resultEquation < 15:
                            points.append((x, y))

                    if len(points) >= 2:
                        # firstPointSafran = min(points, key=lambda x:x[1]) # Plus petite valeur en y
                        firstPointSafran = (self.x1RefPoint, self.y1RefPoint)
                        # Plus grande valeur en y
                        secondPointSafran = max(points, key=lambda x: x[1])

                        # Ajout du décalage pour correspondre sur l'image original
                        firstPointSafranOffSet = tuple(map(lambda x, y: x + y, firstPointSafran, tOffSetSafran))
                        secondPointSafranOffSet = tuple(map(lambda x, y: x + y, secondPointSafran, tOffSetSafran))
                        hauteurSafran = secondPointSafran[1] - firstPointSafran[1]

                        # Décalage des coordonnées du contour pour correspondre sur l'image original (frame)
                        contourSafranOffset = contourSafran + (self.x1, self.y1)

                        return Contour(hauteurSafran, contourSafranOffset, firstPointSafranOffSet, secondPointSafranOffSet, qualityIndex)

        return Contour(None, None, None, None, qualityIndex)
