import cv2  # OpenCV library
import numpy as np


def computeDerive(aFrame):
    
    result = {}
    frame = aFrame

    # Sélection de la zone à analyser
    crop_image_ecume = frame[250:417, 1400:1514] # [y1:y2, x1:x2]

    # Conversion en noir et blanc et floutage
    gray_img_ecume = cv2.cvtColor(crop_image_ecume, cv2.COLOR_BGR2GRAY)
    gray_img_ecume = cv2.GaussianBlur(gray_img_ecume, (9, 9), 0)

    # Conversion de l'image grisé en image binaire, 2 couleurs de pixel, noir et blanc, plus de gris (binarization opencv)
    ret, binary_img_ecume = cv2.threshold(gray_img_ecume, 90, 255, cv2.THRESH_BINARY)

    # Détection des contours
    # La variable de hiérarchie contient des informations sur la relation entre chaque contour. (si un contour est dans un contour)
    contours_list_ecume, hierarchy = cv2.findContours(binary_img_ecume, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours_list_ecume:

        # Récupère le contour le plus grand
        cEcume = max(contours_list_ecume, key=cv2.contourArea)

        right = tuple(cEcume[cEcume[:, :, 0].argmax()][0]) # valeur x maximale parmis tous les x
        rightPoints = np.where(cEcume[:, :, 0] == right[0]) # cherche toutes les valeurs égales à la valeur la plus à droite parmis les x

        # Si on a bien 2 coordonnées
        if(len(rightPoints[0]) == 2):

            # premier crochet -> tableau contenant les indices des coordonnées les plus à droite
            # deuxième crochet -> choisir entre les indices (normalement 2 indices en tout)
            # troisième crochet -> tout le temps 0
            # quatrième crochet -> choisis entre x et y
            x1 = (cEcume[rightPoints[0][0]][0])[0] + 1400
            y1 = (cEcume[rightPoints[0][0]][0])[1] + 250
            x2 = (cEcume[rightPoints[0][1]][0])[0] + 1400
            y2 = 268
            y2Test = (cEcume[rightPoints[0][1]][0])[1] + 250

            # Si y1 trop grand -> caméra occulté par l'embrun
            if(y2Test < 275):

                result["height"] = y1 - y2
                result["contour"] = cEcume
                result["pos1"] = (x1, y2)
                result["pos2"] = (x2, y1)
                return result

    return None