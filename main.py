import os
import cv2  # OpenCV library
from scipy.ndimage import uniform_filter1d
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from analyzers.analyseVideo import AnalyseVideo
from analyzers.analyseDerive import AnalyseDerive
from analyzers.analyseSafran import AnalyseSafran
from analyzers.analyseMousse import AnalyseMousse


"""
TODO:   - mesurer le nombre d'image sur une durée d'une minute
        - calibrer les deux caméras avec le mouvement d'un objet.
"""


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

        self.cap = cv2.VideoCapture(os.path.dirname(__file__) + "/video/input/ccc2.mp4")
        self.cap2 = cv2.VideoCapture(os.path.dirname(__file__) + "/video/input/videoBabord.mp4")
        #self.cap = cv2.VideoCapture("rtsp://root:M101_svr@192.168.1.56:554/axis-media/media.amp") # 192.168.1.55 ou 192.168.1.56
        #self.cap2 = cv2.VideoCapture("rtsp://root:M101_svr@192.168.1.56:554/axis-media/media.amp") # 192.168.1.55 ou 192.168.1.56

        self.analyses = {}
        self.analyses["derive"] = AnalyseDerive(1462, 271, 1514, 410, 27)
        self.analyses["safran"] = AnalyseSafran(355, 433, 451, 572, 411, 435, 423, 506, 21)
        self.analyses["mousse"] = AnalyseMousse(570, 380, 715, 637, 31)

        self.analysesTribord = {}
        self.analysesTribord["derive"] = AnalyseDerive(878, 423, 895, 484, 17)
        self.analysesTribord["safran"] = AnalyseSafran(520, 406, 609, 549, 562, 413, 578, 467, 12)
        self.analysesTribord["mousse"] = AnalyseMousse(632, 415, 680, 526, 12)

        self.analysesBabord = {}
        self.analysesBabord["derive"] = AnalyseDerive(850, 568, 888, 654, 13)
        self.analysesBabord["safran"] = AnalyseSafran(1234, 537, 1337, 669, 1315, 550, 1262, 645, 23)
        self.analysesBabord["mousse"] = AnalyseMousse(1170, 555, 1218, 665, 24)

        # liste des threads
        self.analyseVideoList = []
        self.analyseVideoList.append(AnalyseVideo(self.cap, self.analyses, "Tribord"))
        self.analyseVideoList.append(AnalyseVideo(self.cap2, self.analysesBabord, "Babord"))

        self.listData = {}

    def execute(self):
        """
        Fonction d'éxécution du programme principal.
        """

        # démarre les threads
        for analyseVideo in self.analyseVideoList:
            analyseVideo.start()

        # attend la fin des threads
        for analyseVideo in self.analyseVideoList:
            analyseVideo.join()

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
                
                # suppression des données de qualité inférieure à la limite
                for analyseVideo in self.analyseVideoList:
                    if analyseVideo.name == key:
                        dataRecovery.data[aMeasureKey] = dataRecovery.data[aMeasureKey].loc[dataRecovery.data[aMeasureKey]["quality"] > analyseVideo.analyses[aMeasureKey].qualityLimit]

                # Moyenne glissante
                dataRecovery.data[aMeasureKey]["height"] = uniform_filter1d(dataRecovery.data[aMeasureKey]["height"].values.tolist(), size=25)

            ###### Affichage des résultats ######

            for aMeasureKey in dataRecovery.data:
                fig.append_trace(go.Scatter(
                x=copyDataRecovery[aMeasureKey]["numFrame"].values.tolist(),
                y=copyDataRecovery[aMeasureKey]["quality"],
                name="qualité_"+aMeasureKey + "_" + key
                ), row=2, col=1)
            
            # Courbes
            for aMeasureKey in dataRecovery.data:
                fig.append_trace(go.Scatter(
                    x=dataRecovery.data[aMeasureKey]["numFrame"].values.tolist(),
                    y=dataRecovery.data[aMeasureKey]["height"],
                    name=aMeasureKey + "_" + key
                ), row=1, col=1)

                copyDataRecovery[aMeasureKey].to_excel(os.path.dirname(__file__) + "/files/outputMesures" + key + aMeasureKey +".xlsx", index = False)

        fig.update_layout(title_text="Courbes des hauteurs et qualité de mesure")
        fig.show()

if __name__ == "__main__":
    Main().execute()
