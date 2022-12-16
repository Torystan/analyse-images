from analyzers.embrunDetection import EmbrunDetection

class AnalyseContour():
    """
    Class abstraite qui décrit les class d'analyse pour mesurer une hauteur.
      
    Attributs:
        x1 (int): Première coordonnée x de la zone à analyser sur l'image.
        y1 (int): Première coordonnée y de la zone à analyser sur l'image.
        x2 (int): Deuxième coordonnée x de la zone à analyser sur l'image.
        y2 (int): Deuxième coordonnée y de la zone à analyser sur l'image.
        embrunDetection (EmbrunDetection): Objet EmbrunDetection permettant de détecter l'embrun dans la zone à analyser
    """

    def __init__(self, x1, y1, x2, y2):
        """
        Contructeur de la class AnalyseContour.
            
        Paramètres:
            x1 (int): Première coordonnée x de la zone à analyser sur l'image.
            y1 (int): Première coordonnée y de la zone à analyser sur l'image.
            x2 (int): Deuxième coordonnée x de la zone à analyser sur l'image.
            y2 (int): Deuxième coordonnée y de la zone à analyser sur l'image.
        """

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.embrunDetection = EmbrunDetection()

    def compute(self, frame):
        """
        Fonction qui analyse l'image pour déterminer une hauteur.

        Paramètres:
            frame (OutputArray): Image à analyser.
            
        Returns:
            Contour: Objet Contour contenant toutes les informations de la mesure.
            None: Si l'analyse n'est pas possible.
        """
        pass
