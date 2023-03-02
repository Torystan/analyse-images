import os
import cv2  # OpenCV library
from scipy.ndimage import uniform_filter1d
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import statistics

from analyzers.analyseVideo import AnalyseVideo
from analyzers.analyseSafran import AnalyseSafran
from analyzers.analyseMousse import AnalyseMousse


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

        self.cap = cv2.VideoCapture(os.path.dirname(__file__) + "/video/input/vagues/vagueBabord2.mp4")

        # Liste des zones d'analyses
        self.analysesTribord = {}
        self.analysesTribord["derive"] = AnalyseMousse(934, 413, 956, 474, 14)
        self.analysesTribord["safran"] = AnalyseSafran(596, 367, 663, 493, 626, 380, 641, 448, 10)
        self.analysesTribord["mousse"] = AnalyseMousse(694, 388, 714, 496, 21)

        self.analysesBabord = {}
        self.analysesBabord["derive"] = AnalyseMousse(946, 358, 987, 452, 9)
        self.analysesBabord["safran"] = AnalyseSafran(1325, 336, 1408, 470, 1402, 342, 1368, 408, 1)
        self.analysesBabord["mousse"] = AnalyseMousse(1262, 346, 1298, 483, 1)

        # liste des threads
        self.analyseVideoList = []
        self.analyseVideoList.append(AnalyseVideo(self.cap, self.analysesBabord, "videoBabord2"))

        self.listData = {}

    def execute(self):
        """
        Fonction d'éxécution du programme principal.
        """

        # Démarre les threads
        for analyseVideo in self.analyseVideoList:
            analyseVideo.start()

        # Attend la fin des threads
        for analyseVideo in self.analyseVideoList:
            analyseVideo.join()

        # Récupération des données
        for analyseVideo in self.analyseVideoList:
            self.listData[analyseVideo.name] = analyseVideo.dataRecovery

        ###### Traitement des données #####

        fig = make_subplots(rows=2, cols=1)

        for key, dataRecovery in self.listData.items():
            dataRecovery.convertToDataframe()
            copyDataRecovery = dataRecovery.data.copy()

            for aMeasureKey in dataRecovery.data:
                # Suppression des lignes ou la hauteur est nulle
                dataRecovery.data[aMeasureKey] = dataRecovery.data[aMeasureKey].dropna()
                
                # Suppression des données de qualité inférieure à la limite
                for analyseVideo in self.analyseVideoList:
                    if analyseVideo.name == key:
                        dataRecovery.data[aMeasureKey] = dataRecovery.data[aMeasureKey].loc[dataRecovery.data[aMeasureKey]["quality"] > analyseVideo.analyses[aMeasureKey].qualityLimit]

                # Moyenne glissante
                dataRecovery.data[aMeasureKey]["height"] = uniform_filter1d(dataRecovery.data[aMeasureKey]["height"].values.tolist(), size=15)

            ###### Affichage des résultats ######

            for aMeasureKey in dataRecovery.data:

                # Courbes de qualité
                fig.append_trace(go.Scatter(
                x=copyDataRecovery[aMeasureKey]["numFrame"].values.tolist(),
                y=copyDataRecovery[aMeasureKey]["quality"],
                name="qualité_"+aMeasureKey + "_" + key,
                ), row=2, col=1)
            
                # Courbes des hauteurs
                fig.append_trace(go.Scatter(
                    x=dataRecovery.data[aMeasureKey]["numFrame"].values.tolist(),
                    y=dataRecovery.data[aMeasureKey]["height"],
                    name=aMeasureKey + "_" + key,
                ), row=1, col=1)

                copyDataRecovery[aMeasureKey].to_excel(os.path.dirname(__file__) + "/files/outputMesures" + key + aMeasureKey +".xlsx", index = False)

        fig.update_layout(title_text="Courbes des hauteurs et qualité de mesure")
        fig.show()

        list = []
        oldI = 0
        for i in dataRecovery.data[aMeasureKey]["date"].values.tolist():
            list.append(i - oldI)
            oldI = i
        print(str(statistics.median(list)/1000000))

if __name__ == "__main__":
    Main().execute()
