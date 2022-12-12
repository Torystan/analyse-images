import cv2  # OpenCV library
import numpy as np


def computeMousse(aFrame):

    result = {}

    frame = aFrame

    # Sélection de la zone à analyser
    crop_image_MousseBrasArriere = frame[385:650, 550:650]
    
    # Conversion en noir et blanc et floutage
    gray_img_MousseBrasArriere = cv2.cvtColor(crop_image_MousseBrasArriere, cv2.COLOR_BGR2GRAY)
    gray_img_MousseBrasArriere = cv2.GaussianBlur(gray_img_MousseBrasArriere, (11, 11), 0)

    # Conversion de l'image grisé en image binaire, 2 couleurs de pixel, noir et blanc, plus de gris (binarization opencv)
    ret, binary_img_MousseBrasArriere = cv2.threshold(gray_img_MousseBrasArriere, 90, 255, cv2.THRESH_BINARY)

    # Détection des contours
    # La variable de hiérarchie contient des informations sur la relation entre chaque contour. (si un contour est dans un contour)
    contours_list_MousseBrasArriere, hierarchy = cv2.findContours(binary_img_MousseBrasArriere, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours_list_MousseBrasArriere:

        # Récupère le contour le plus grand
        cMousseBrasArriere = max(contours_list_MousseBrasArriere, key=cv2.contourArea)

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
            x2 = (cMousseBrasArriere[rightPoints[0][len(rightPoints[0])-1]][0])[0]
            y2 = (cMousseBrasArriere[rightPoints[0][len(rightPoints[0])-1]][0])[1]

            # TODO Si y1 trop grand -> caméra occulté par l'embrun
            if(True):

                # Si les deux coordonnées ne sont pas dans le bon ordre
                if y2 > y1:
                    temp = y2
                    y2 = y1
                    y1 = temp

                result["height"] = y1 - y2
                result["contour"] = cMousseBrasArriere
                result["pos1"] = (x2 + 550, y2 + 385)
                result["pos2"] = (x1 + 550, y1 + 385)
                return result

    return None
