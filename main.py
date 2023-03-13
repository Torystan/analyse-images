import os
import cv2  # OpenCV library
from scipy.ndimage import uniform_filter1d
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from analyzers.analyseVideo import AnalyseVideo
from analyzers.analyseSafran import AnalyseSafran
from analyzers.analyseMousse import AnalyseMousse


class Main():
    """
    Class principale éxécuté au début du programme

    Attributs:
        cap (VideoCapture): Objet permettant de charger une vidéo, puis de la lire avec la méthode read().
        analyse (dict of str: AnalyseContour): Dictionnaires des objets d'analyse des éléments de l'image
        analyseVideoList (list of AnalyseVideo): liste des objets AnalyseVideo.
        listData (list of Data): liste des objets Data.
    """

    def __init__(self):
        """
        Constructeur de la class Main()
        """

        cap = cv2.VideoCapture(os.path.dirname(__file__) + "/video/input/ccc.mkv")

        # Liste des zones d'analyses
        self.analysesTribord = {}
        self.analysesTribord["derive"] = AnalyseMousse(1462, 271, 1514, 410, 27)
        self.analysesTribord["safran"] = AnalyseSafran(355, 433, 451, 572, 411, 435, 423, 506, 21)
        self.analysesTribord["mousse"] = AnalyseMousse(570, 380, 715, 637, 31)

        # liste des threads
        self.analyseVideoList = []
        self.analyseVideoList.append(AnalyseVideo(cap, self.analysesTribord, "videoTribord"))

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
            self.listData[analyseVideo.name] = analyseVideo.data

        ###### Traitement des données #####

        fig = make_subplots(rows=2, cols=1)

        for key, data in self.listData.items():
            data.convertToDataframe()
            copyData = data.data.copy()

            for aMeasureKey in data.data:
                # Suppression des lignes ou la hauteur est nulle
                data.data[aMeasureKey] = data.data[aMeasureKey].dropna()
                
                # Suppression des données de qualité inférieure à la limite
                for analyseVideo in self.analyseVideoList:
                    if analyseVideo.name == key:
                        data.data[aMeasureKey] = data.data[aMeasureKey].loc[data.data[aMeasureKey]["quality"] > analyseVideo.analyses[aMeasureKey].qualityLimit]

                # Moyenne glissante
                data.data[aMeasureKey]["height"] = uniform_filter1d(data.data[aMeasureKey]["height"].values.tolist(), size=50)

            ###### Affichage des résultats ######

            for aMeasureKey in data.data:

                # Courbes de qualité
                fig.append_trace(go.Scatter(
                x=copyData[aMeasureKey]["numFrame"].values.tolist(),
                y=copyData[aMeasureKey]["quality"],
                name="qualité_"+aMeasureKey + "_" + key,
                ), row=2, col=1)
            
                # Courbes des hauteurs
                fig.append_trace(go.Scatter(
                    x=data.data[aMeasureKey]["numFrame"].values.tolist(),
                    y=data.data[aMeasureKey]["height"],
                    name=aMeasureKey + "_" + key,
                ), row=1, col=1)

                copyData[aMeasureKey].to_excel(os.path.dirname(__file__) + "/files/outputMesures" + key + aMeasureKey +".xlsx", index = False)

        fig.update_layout(title_text="Courbes des hauteurs et qualité de mesure")
        fig.show()

if __name__ == "__main__":
    Main().execute()
