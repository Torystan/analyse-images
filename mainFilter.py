import os
import cv2  # OpenCV library
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, lfilter, filtfilt, freqz
import scipy.signal as signal
import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

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
        self.analyseVideoList.append(AnalyseVideo(self.cap, self.analysesBabord, "videoBabord"))

        self.listData = {}

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

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
                dataRecovery.data[aMeasureKey]["height"] = uniform_filter1d(dataRecovery.data[aMeasureKey]["height"].values.tolist(), size=1)

            ###### Affichage des résultats ######

        fig = make_subplots(rows=3, cols=1)

        # Courbe

        # Setting standard filter requirements.
        order = 3
        fs = 30.0
        cutoff = 0.4

        # Creating the data for filteration
        t = dataRecovery.data["mousse"]["numFrame"].values.tolist()

        data = dataRecovery.data["mousse"]["height"]

        # Filtering and plotting
        filterY = self.butter_lowpass_filter(data, cutoff, fs, order)

        fig.append_trace(go.Scatter(
        x=t,
        y=data,
        name="data",
        ), row=1, col=1)

        fig.append_trace(go.Scatter(
        x=t,
        y=filterY,
        name="passe bas",
        ), row=1, col=1)

        # Setting standard filter requirements.
        order = 3
        fs = 30.0
        cutoff = 1

        # Creating the data for filteration
        t = dataRecovery.data["mousse"]["numFrame"].values.tolist()

        data = dataRecovery.data["mousse"]["height"]

        # Filtering and plotting
        filterY = self.butter_highpass_filter(data, cutoff, fs, order)

        fig.append_trace(go.Scatter(
        x=t,
        y=filterY,
        name="passe haut",
        ), row=2, col=1)

        # Setting standard filter requirements.
        order = 3
        fs = 30.0
        cutoff = 0.5

        # Creating the data for filteration
        t = dataRecovery.data["mousse"]["numFrame"].values.tolist()

        data = dataRecovery.data["mousse"]["height"]

        # Filtering and plotting
        y = self.butter_highpass_filter(data, cutoff, fs, order)
        z = self.butter_lowpass_filter(y, 1, fs, order)

        fig.append_trace(go.Scatter(
        x=t,
        y=z,
        name="passe vague",
        ), row=3, col=1)

        fig.update_layout(title_text="Courbes des filtres passe haut et bas")
        fig.show()

if __name__ == "__main__":
    Main().execute()
