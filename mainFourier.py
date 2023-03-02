import os
import cv2  # OpenCV library
from scipy.ndimage import uniform_filter1d
import scipy.signal as signal
import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt

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

        fig, axs = plt.subplots(2)
        fig.suptitle('hauteur')

        # Courbes
        for aMeasureKey in dataRecovery.data:
            axs[0].plot(dataRecovery.data[aMeasureKey]["numFrame"].values.tolist(), dataRecovery.data[aMeasureKey]["height"], label=aMeasureKey)

        plt.figure()

        rate = 1/30

        # Calcul FFT
        X = fft(dataRecovery.data["mousse"]["height"])  # Transformée de fourier
        freq = fftfreq(dataRecovery.data["mousse"]["height"].size, d=rate)  # Fréquences de la transformée de Fourier

        # Calcul du nombre d'échantillon
        N = dataRecovery.data["mousse"]["height"].size

        # On prend la valeur absolue de l'amplitude uniquement pour les fréquences positives et normalisation
        X_abs = np.abs(X[:N//2])*2.0/N
        # On garde uniquement les fréquences positives
        freq_pos = freq[:N//2]

        plt.plot(freq_pos, X_abs, label="Amplitude absolue")
        plt.grid()
        plt.xlabel(r"Fréquence (Hz)")
        plt.ylabel(r"Amplitude $|X(f)|$")
        plt.title("Transformée de Fourier")
        plt.figure()

        f, t, Sxx = signal.spectrogram(dataRecovery.data["mousse"]["height"], 30)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Fréquence (Hz)')
        plt.xlabel('Temps (s)')
        plt.title('Spectrogramme')
        plt.show()

if __name__ == "__main__":
    Main().execute()
