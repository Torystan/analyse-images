import pandas as pd


class DataRecovery():
    """
    Class qui contient toutes données.
      
    Attributs:
        data (dict of str: dict of str: []): Dictionnaire des éléments mesuré qui contiennent eux même un dictionnaire des données de la mesure
    """

    def __init__(self):
        """
        Constructeur de la class DataRecovery.
        """

        self.data = {}

    def addData(self, key, numFrame, contour):
        """
        Fonction qui ajoute des données à l'attribut data.
  
        Paramètres:
            key (str): Clé du dictionnaire des données qui est le nom de l'object mesuré.
            numFrame (int): Numéro de l'image qui correspond à la mesure des données.
            contour (Contour): Object Contour qui contient les données de la mesure.
        """

        if key not in self.data:
            self.data[key] = {"numFrame": [], "height": [], "contour": [], "firstPosMeasure": [], "secondPosMeasure": [], "quality": []}

        self.data[key]["numFrame"].append(numFrame)

        if contour is not None:
            self.data[key]["height"].append(contour.height)
            self.data[key]["contour"].append(contour.contour)
            self.data[key]["firstPosMeasure"].append(contour.firstPosMeasure)
            self.data[key]["secondPosMeasure"].append(contour.secondPosMeasure)
            self.data[key]["quality"].append(contour.qualityIndex)
        else:
            self.data[key]["height"].append(None)
            self.data[key]["contour"].append(None)
            self.data[key]["firstPosMeasure"].append(None)
            self.data[key]["secondPosMeasure"].append(None)
            self.data[key]["quality"].append(None)

    def convertToDataframe(self):
        """
        Fonction qui convertit les dictionnaires de tableaux en DataFrame, écrase les anciennes valeurs.
        """

        for aMeasureKey in self.data:
            self.data[aMeasureKey] = pd.DataFrame.from_dict(self.data[aMeasureKey], orient='index').transpose()
