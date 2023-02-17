import cv2  # OpenCV library
import numpy as np
from analyzers.analyseContour import AnalyseContour
from analyzers.contour import Contour


class AnalyseMousse(AnalyseContour):
    """
    Class qui mesure la taille de la mousse en dessous du bras arrière du bateau.
    """

    def __init__(self, x1, y1, x2, y2, qualityLimit):
        super().__init__(x1, y1, x2, y2, qualityLimit)
    
    def compute(self, frame):

        cropFrame = frame[self.y1:self.y2, self.x1:self.x2]
        
        # Conversion en noir et blanc et floutage
        gray_img_MousseBrasArriere = cv2.cvtColor(cropFrame, cv2.COLOR_BGR2GRAY)
        gray_img_MousseBrasArriere = cv2.GaussianBlur(gray_img_MousseBrasArriere, (9, 9), 0)

        # Conversion de l'image grisé en image binaire, 2 couleurs de pixel, noir et blanc (binarisation opencv)
        median_pix = 0.80 * np.median(gray_img_MousseBrasArriere[0:round((self.y2-self.y1)/5), 0:(self.x2-self.x1) - round((self.x2-self.x1)/10)])
        ret, binary_img_MousseBrasArriere = cv2.threshold(gray_img_MousseBrasArriere, median_pix, 255, cv2.THRESH_BINARY)

        # Détection des contours
        # La variable de hiérarchie contient des informations sur la relation entre chaque contour. (si un contour est dans un contour)
        contours_list_MousseBrasArriere, hierarchy = cv2.findContours(binary_img_MousseBrasArriere, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #décalage de la zone d'analyse pour détecter l'embrun en amont
        cropFrameEmbrun = frame[self.y1:self.y2, int(self.x1 + (self.x2 - self.x1)/2):int(self.x2 + (self.x2 - self.x1)/2)]
        qualityIndex = self.embrunDetection.detection(cropFrameEmbrun)

        if contours_list_MousseBrasArriere:

            # Trouver le contour le plus proche du coin en haut à droite (pour éviter les contours parasites)
            cMousseBrasArriere = None
            for c in contours_list_MousseBrasArriere:
                # Si un contour est à moins de 10 pixel du point (coin en haut à droite de la zone d'analyse)
                if abs(cv2.pointPolygonTest(c, (self.x2 - self.x1, 1), True)) < 10 and cv2.contourArea(c) > 100:
                    cMousseBrasArriere = c

            if cMousseBrasArriere is None:
                return Contour(None, None, None, None, qualityIndex)

            right = tuple(cMousseBrasArriere[cMousseBrasArriere[:, :, 0].argmax()][0]) # valeur x maximal parmis tous les x
            rightPoints = np.where(cMousseBrasArriere[:, :, 0] == right[0]) # cherche toutes les valeurs égales à la valeur la plus à droite parmis les x

            # Si on a bien 2 coordonnées au moins
            if(len(rightPoints[0]) >= 2):

                # premier crochet -> tableau contenant les indices des coordonnées les plus à droite
                # deuxième crochet -> choisir entre les indices (en général 2 indices en tout)
                # troisième crochet -> tout le temps 0
                # quatrième crochet -> x = 0 et y = 1
                x1 = (cMousseBrasArriere[rightPoints[0][0]][0])[0]
                y1 = (cMousseBrasArriere[rightPoints[0][0]][0])[1]
                x2 = (cMousseBrasArriere[rightPoints[0][1]][0])[0]
                y2 = (cMousseBrasArriere[rightPoints[0][1]][0])[1]

                # Si les deux coordonnées ne sont pas dans le bon ordre
                if y2 > y1:
                    temp = y2
                    y2 = y1
                    y1 = temp

                y2 = 0

                # Décalage des coordonnées du contour pour correspondre sur l'image original (frame)
                cMousseBrasArriere = cMousseBrasArriere + (self.x1, self.y1)

                return Contour(y1 - y2, cMousseBrasArriere, (x2 + self.x1, y2 + self.y1), (x1 + self.x1, y1 + self.y1), qualityIndex)

        return Contour(None, None, None, None, qualityIndex)
