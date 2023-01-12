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

    def __init__(self, x1, y1, x2, y2, xRefPoint, yRefPoint):
        super().__init__(x1, y1, x2, y2)
        self.qualityLimit = 21
        self.xRefPoint = xRefPoint
        self.yRefPoint = yRefPoint

    def compute(self, frame):
        """
        Méthode qui mesure la taille du safran qui sort de l'eau.
        """

        cropFrame = frame[self.y1:self.y2, self.x1:self.x2]

        qualityIndex = self.embrunDetection.detection(cropFrame)

        # Conversion en noir et blanc et floutage
        gray_img_safran = cv2.cvtColor(cropFrame, cv2.COLOR_BGR2GRAY)
        gray_img_safran = cv2.GaussianBlur(gray_img_safran, (3, 7), 0)
        

        # dessins de toutes les bordures
        lower = int(max(0, 1.5*qualityIndex))
        upper = int(min(255, 4*qualityIndex))
        edged = cv2.Canny(gray_img_safran, lower, upper)

        # Détection des contours
        # La variable de hiérarchie contient des informations sur la relation entre chaque contour. (si un contour est dans un contour)
        contours_list_safran, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours_list_safran:
            contourSafran = None

            for c in contours_list_safran:
                # Si un contour est à moins de 4 pixel du point (68, 17) (point sur le bord gauche du safran)
                if abs(cv2.pointPolygonTest(c, (self.xRefPoint + 1, self.yRefPoint + 10), True)) < 4:
                    contourSafran = c
                    tOffSetSafran = (self.x1, self.y1)

                    points = []

                    # Regarde si les points correspondent plus ou moins à l'équation 38x -7y = 2424
                    # Equation qui représente la droite au niveau du safran
                    for p in c:
                        x = p[0][0]
                        y = p[0][1]
                        # Résultat de l'équation
                        resultEquation = (38 * x - 7 * y)

                        if resultEquation > 2424 - 80 and resultEquation < 2424 + 80:
                            points.append((x, y))

                    if len(points) >= 2:
                        # firstPointSafran = min(points, key=lambda x:x[1]) # Plus petite valeur en y
                        firstPointSafran = (self.xRefPoint, self.yRefPoint)
                        # Plus grande valeur en y
                        secondPointSafran = max(points, key=lambda x: x[1])

                        # Ajout du décalage pour correspondre sur l'image original
                        firstPointSafranOffSet = tuple(map(lambda x, y: x + y, firstPointSafran, tOffSetSafran))
                        secondPointSafranOffSet = tuple(map(lambda x, y: x + y, secondPointSafran, tOffSetSafran))
                        hauteurSafran = secondPointSafran[1] - firstPointSafran[1]

                        # Décalage des coordonnées du contour pour correspondre sur l'image original (frame)
                        contourSafranOffset = contourSafran + (self.x1, self.y1)

                        '''
                        gray_img_safran2 = cv2.cvtColor(gray_img_safran, cv2.COLOR_GRAY2BGR)
                        
                        cv2.drawContours(gray_img_safran2, [contourSafran], 0, (255, 0, 255), 2)
                        cv2.line(gray_img_safran2, firstPointSafran, secondPointSafran, (0, 255, 0), 2)

                        imagesConcat = np.concatenate((cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR), gray_img_safran2), axis=1)
                        cv2.imshow('frameSafran', imagesConcat)
                        cv2.waitKey(1)
                        '''

                        return Contour(hauteurSafran, contourSafranOffset, firstPointSafranOffSet, secondPointSafranOffSet, qualityIndex)

        '''
        cv2.imshow('frameSafran', gray_img_safran)
        cv2.waitKey(1)
        '''
        return Contour(None, None, None, None, qualityIndex)
