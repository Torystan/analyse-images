import os
import cv2  # OpenCV library
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from analyzers.analyseDerive import computeDerive
from analyzers.analyseSafran import computeSafran
from analyzers.analyseMousse import computeMousse

cap = cv2.VideoCapture(os.path.dirname(__file__) + "/video/ccc2.mp4")

valuesDerive = []
numFrameDerive = []

valuesSafran = []
numFrameSafran = []

valuesMousseBrasArriere = []
numFrameMousseBrasArriere = []

valuesDataReliability = []
numFrame = []
dataReliability = 0

nbFrame = 0
movingAverageSize = 50

#Enregistrement
'''
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
videoObject = cv2.VideoWriter('detection_hauteur2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, size)
'''

while True:

    # Récupère une image de la vidéo
    ret, frame = cap.read()
    if frame is None:
        print("-----  Fin de la vidéo  -----")
        break

    ###### Détection Embrun #####
    crop_image_embrun_right = frame[1:1079, 1450:1919]
    crop_image_embrun = frame

    embrun = False
    embrunRight = False

    sum = 0
    sumPixels = 0
    sumRight = 0
    sumPixelsRight = 0

    hist = cv2.calcHist([crop_image_embrun],[0],None,[256],[0,256])
    histRight = cv2.calcHist([crop_image_embrun_right],[0],None,[256],[0,256])

    for i in range(1, len(hist)):
        sum = sum + int(hist[i][0]) * i
        sumPixels += int(hist[i][0])
        sumRight = sumRight + int(histRight[i][0]) * i
        sumPixelsRight += int(histRight[i][0])

    if int(sumRight/sumPixelsRight) > 100 :
        embrunRight = True

    if int(sum/sumPixels) > 125 :
        embrun = True

    ###### Analyse de l'image #####

    resultDerive = computeDerive(frame)
    resultSafran = computeSafran(frame)
    resultMousse = computeMousse(frame)
    
    if resultDerive is not None and not embrun:

        # Enregistre la hauteur
        valuesDerive.append(resultDerive["height"])
        numFrameDerive.append(nbFrame)

        # Dessin des mesures
        cv2.drawContours(frame, [resultDerive["contour"]], 0, (255, 0, 255), 2, offset=(1400, 249))
        cv2.line(frame,resultDerive["pos1"], resultDerive["pos2"],(0,255,0), 2) # Tracé de la hauteur de l'écume juste derière le foil
    else:
        dataReliability += 1


    if resultSafran is not None and not embrunRight:

        valuesSafran.append(resultSafran["height"])
        numFrameSafran.append(nbFrame)

        cv2.drawContours(frame, [resultSafran["contour"]], 0, (255, 0, 255), 2, offset=(344, 430))
        cv2.line(frame, resultSafran["pos1"], resultSafran["pos2"], (0,255,0), 2) # Tracé de la hauteur de l'écume juste derière le foil
    else:
        dataReliability += 1


    if resultMousse is not None and not embrunRight:

        # Enregistre la hauteur
        valuesMousseBrasArriere.append(resultMousse["height"])
        numFrameMousseBrasArriere.append(nbFrame)

        # Dessin des mesures
        cv2.drawContours(frame, [resultMousse["contour"]], 0, (255, 0, 255), 2, offset=(550, 385))
        cv2.line(frame, resultMousse["pos1"], resultMousse["pos2"], (0,255,0), 3) # Tracé de la hauteur de la mousse
    else:
        dataReliability += 1


    ###### Analyse de l'image #####

    valuesDataReliability.append(dataReliability)
    dataReliability = 0
    numFrame.append(nbFrame)
    nbFrame += 1

    #Enregistrement
    #videoObject.write(frame)

    # Afficher l'image avec les dessins
    cv2.imshow('frame', frame)


    ###### Fin de la vidéo #####

    # Attend 1 millisecondes et regarde si l'utilisateur appuie sur 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

#Enregistrement
#videoObject.release()

cv2.destroyAllWindows()


###### Affichage des résultats #####

# Courbe
valuesFoamMovingAverage = uniform_filter1d(valuesDerive, size=movingAverageSize)
valuesMousseBrasArriereMovingAverage = uniform_filter1d(valuesMousseBrasArriere, size=movingAverageSize)
valuesSafranMovingAverage = uniform_filter1d(valuesSafran, size=movingAverageSize)

plt.plot(numFrameDerive, valuesFoamMovingAverage, label = "Dérive")
plt.plot(numFrameMousseBrasArriere, valuesMousseBrasArriereMovingAverage, label = "Mousse bras arrière")
plt.plot(numFrameSafran, valuesSafranMovingAverage, label = "Safran")
plt.plot(numFrame, valuesDataReliability, label = "fiabilité")
plt.title('Hauteur en pixels en fonction du temps')
plt.legend()
plt.show()