import os
import cv2  # OpenCV library
import time
import threading

from analyzers.dataRecovery import DataRecovery



class AnalyseVideo(threading.Thread):
    """
    Class qui permet d'analyser une vidéo et de récupérer les données.

    Attributs:
        video (VideoCapture): Objet permettant de charger une vidéo, puis de la lire avec la méthode read().
        listeAnalyse (dict of str: AnalyseContour): Dictionnaires des objets d'analyse des éléments de l'image
        name (str): Nom de l'analyse

    return:
        dataRecovery (DataRecovery): Objet permettant de récupérer les données.
    """

    def __init__(self, video, listeAnalyse, name):
        """
        Constructeur de la class AnalyseVideo()
        """
        threading.Thread.__init__(self)

        self.name = name
        self.cap = video
        self.record = True
        self.videoObject = None
        self.nbFrame = 1

        self.analyses = listeAnalyse

        self.dataRecovery = DataRecovery()

        # Enregistrement
        if self.record:
            frame_width = int(self.cap.get(3))
            frame_height = int(self.cap.get(4))

            print(str(self.cap.get(cv2.CAP_PROP_FPS)) + self.name)
            size = (frame_width, frame_height)
            self.videoObject = cv2.VideoWriter(os.path.dirname(__file__) + "/../video/record/record" + self.name + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), self.cap.get(cv2.CAP_PROP_FPS), size)

    def run(self):
        """
        Fonction d'éxécution du programme qui éxétute toutes les analyses des zones,
        affiche l'image avec les tracés et renvois les données à la fin de la vidéo.
        """

        while True:

            ###### Lecture de la vidéo ######

            # Récupère une image de la vidéo
            ret, frame = self.cap.read()
            if frame is None:
                print("-----  Fin de la vidéo  -----")
                break

            ###### Récupération des données ######

            for analyseKey, analyseObject in self.analyses.items():
                result = analyseObject.compute(frame)
                date = time.time_ns()
                self.dataRecovery.addData(analyseKey, self.nbFrame, date, result)

            ###### Dessin des données #####

            i = 0

            for aMeasureKey, aMeasureValue in self.dataRecovery.data.items():
                if aMeasureValue["height"][-1] != None:  # Pas de dessin des mesures vides
                    
                    # ligne de mesure couleur verte si correct, rouge si occulté par l'embrun
                    color = (0, 255, 0)
                    if aMeasureValue["quality"][-1] < self.analyses[aMeasureKey].qualityLimit:
                        color = (0, 0, 255)

                    cv2.drawContours(frame, [aMeasureValue["contour"][-1]], 0, (255, 0, 255), 1)
                    cv2.line(frame, aMeasureValue["firstPosMeasure"][-1], aMeasureValue["secondPosMeasure"][-1], color, 2)

                cv2.putText(frame, "Erreur " + aMeasureKey + ": " + str(aMeasureValue["quality"][-1]), (10, 110 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                i += 1

            cv2.putText(frame, str(self.nbFrame), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Erreur : ", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            #cv2.putText(frame, str(self.dataRecovery.data["safran"]["quality"][-1]), (145, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            
            ###### Fin de l'analyse de l'image ######

            # Enregistrement
            if self.record:
                self.videoObject.write(frame)

            # Afficher l'image avec les dessins
            cv2.imshow(self.name, frame)

            # waitKey(x) -> Attendre x milliseconde, et regarde si l'utilisateur appuie sur 'q' pour quitter
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.nbFrame += 1

        ###### Fin de l'analyse de la vidéo #####

        self.cap.release()

        # Enregistrement
        if self.record:
            self.videoObject.release()