import cv2  # OpenCV library


def computeSafran(aFrame):

    result = {}

    frame = aFrame
    crop_image_safran = frame[430:579, 344:481]
    
    # Conversion en noir et blanc et floutage
    gray_img_safran = cv2.cvtColor(crop_image_safran, cv2.COLOR_BGR2GRAY)
    gray_img_safran = cv2.GaussianBlur(gray_img_safran, (9, 9), 0)

    # dessins de toutes les bordures
    edged = cv2.Canny(gray_img_safran, 60, 100)
    
    # Détection des contours
    # La variable de hiérarchie contient des informations sur la relation entre chaque contour. (si un contour est dans un contour)
    contours_list_safran, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours_list_safran:
        contourSafran = None

        for c in contours_list_safran:
            # Si un contour est à moins de 4 pixel du point (68, 17) (point sur le bord gauche du safran)
            if abs(cv2.pointPolygonTest(c, (68, 17), True)) < 4 :
                contourSafran = c
                tOffSetSafran = (344, 430)

                points = []

                # Regarde si les points correspondent plus ou moins à l'équation 38x -7y = 2424
                # Equation qui représente la droite au niveau du safran
                for p in c:
                    x = p[0][0]
                    y = p[0][1]
                    resultEquation = (38 * x - 7 * y) # Résultat de l'équation

                    if resultEquation > 2424 - 80 and resultEquation < 2424 + 80:
                        points.append((x,y))

                if len(points) >= 2:
                    firstPointSafran = min(points, key=lambda x:x[1]) # Plus petite valeur en y
                    secondPointSafran = max(points, key=lambda x:x[1]) # Plus grande valeur en y

                    # Ajout du décalage pour correspondre sur l'image original
                    firstPointSafran = tuple(map(lambda x, y: x + y, firstPointSafran, tOffSetSafran))
                    secondPointSafran = tuple(map(lambda x, y: x + y, secondPointSafran, tOffSetSafran))

                    hauteurSafran = secondPointSafran[1] - firstPointSafran[1]
                else:
                    contourSafran = None

                #TODO embrun
                if contourSafran is not None and True:
                    
                    result.update({"height": hauteurSafran})
                    result["contour"] = contourSafran
                    result["pos1"] = firstPointSafran
                    result["pos2"] = secondPointSafran
                    return result

    return None