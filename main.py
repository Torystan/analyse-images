import os
import cv2  # OpenCV library
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from analyzers.analyseDerive import AnalyseDerive
from analyzers.analyseSafran import AnalyseSafran
from analyzers.analyseMousse import AnalyseMousse
from analyzers.dataRecovery import DataRecovery


class Main():
    """
    Class principale éxécuté au début du programme

    Attributs:
        cap (VideoCapture): Objet permettant de charger une vidéo, puis de la lire avec la méthode read().
        record (boolean): Indique si on veut enregistrer la vidéo.
        videoObject (VideoObject): Objet qui contient l'enregistrement dans le cas ou record est True.
        analyse (dict of str: AnalyseContour): Dictionnaires des objets d'analyse des éléments de l'image
        dataRecovery (DataRecovery): Objet permettant de récupérer les données.
    """

    def __init__(self):
        """
        Constructeur de la class Main()
        """

        self.cap = cv2.VideoCapture(os.path.dirname(__file__) + "/video/ccc2.mp4")
        self.record = False
        self.videoObject = None
        self.nbFrame = 1

        self.analyses = {}
        self.analyses["derive"] = AnalyseDerive(1400, 250, 1514, 417)
        self.analyses["safran"] = AnalyseSafran(344, 430, 481, 579, 67, 7)
        self.analyses["mousse"] = AnalyseMousse(550, 385, 650, 650)

        self.dataRecovery = DataRecovery()

        # Enregistrement
        if self.record:
            frame_width = int(self.cap.get(3))
            frame_height = int(self.cap.get(4))

            size = (frame_width, frame_height)
            self.videoObject = cv2.VideoWriter(os.path.dirname(__file__) + "/video/record.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    def execute(self):
        """
        Fonction d'éxécution du programme principal de la lecture de la vidéo à l'affichage des données.
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
                self.dataRecovery.addData(analyseKey, self.nbFrame, result)

            ###### Dessin des données #####

            for aMeasureKey, aMeasureValue in self.dataRecovery.data.items():
                if aMeasureValue["height"][-1] != None:  # Pas de dessin des mesures vides
                    
                    # ligne de mesure couleur verte si correct, rouge si occulté par l'embrun
                    color = (0, 255, 0)
                    if aMeasureValue["quality"][-1] < self.analyses[aMeasureKey].qualityLimit:
                        color = (0, 0, 255)

                    cv2.drawContours(frame, [aMeasureValue["contour"][-1]], 0, (255, 0, 255), 1)
                    cv2.line(frame, aMeasureValue["firstPosMeasure"][-1], aMeasureValue["secondPosMeasure"][-1], color, 2)

            cv2.putText(frame, str(self.nbFrame), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Erreur : ", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, str(self.dataRecovery.data["safran"]["quality"][-1]), (145, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            ###### Fin de l'analyse de l'image ######

            # Enregistrement
            if self.record:
                self.videoObject.write(frame)

            # Afficher l'image avec les dessins
            cv2.imshow('frame', frame)

            # waitKey(x) -> Attendre x milliseconde, et regarde si l'utilisateur appuie sur 'q' pour quitter
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.nbFrame += 1

        ###### Fin de l'analyse de la vidéo #####

        self.cap.release()

        # Enregistrement
        if self.record:
            self.videoObject.release()

        cv2.destroyAllWindows()

        ###### Traitement des données #####

        self.dataRecovery.convertToDataframe()

        fig, axs = plt.subplots(2)
        fig.suptitle('Hauteur en pixels en fonction du temps avc courbe de qualité')

        for aMeasureKey in self.dataRecovery.data:
            axs[1].plot(self.dataRecovery.data[aMeasureKey]["numFrame"].values.tolist(), self.dataRecovery.data[aMeasureKey]["quality"], label=aMeasureKey)

        for aMeasureKey in self.dataRecovery.data:
            # On garde uniquement les données de bonne qualité
            self.dataRecovery.data[aMeasureKey] = self.dataRecovery.data[aMeasureKey].dropna()
            self.dataRecovery.data[aMeasureKey] = self.dataRecovery.data[aMeasureKey].loc[self.dataRecovery.data[aMeasureKey]["quality"] > self.analyses[aMeasureKey].qualityLimit]

            # Moyenne glissante
            self.dataRecovery.data[aMeasureKey]["height"] = uniform_filter1d(self.dataRecovery.data[aMeasureKey]["height"].values.tolist(), size=1)

        ###### Affichage des résultats ######

        # Courbes
        for aMeasureKey in self.dataRecovery.data:
            axs[0].plot(self.dataRecovery.data[aMeasureKey]["numFrame"].values.tolist(), self.dataRecovery.data[aMeasureKey]["height"], label=aMeasureKey)

        plt.legend()
        plt.show()


if __name__ == "__main__":
    Main().execute()
