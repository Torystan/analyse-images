class Contour():
    """
    Class qui contient toutes les propriétés de la mesure d'un contour.
      
    Attributs:
        height (int): Hauteur mesuré.
        contour ([[int]]): Liste des points du contour.
        firstPosMeasure (Tuple(int, int)): Coordonnée (x, y) du premier point de la mesure de la hauteur.
        pos2 (Tuple(int, int)): Coordonnée (x, y) du deuxième point de la mesure de la hauteur.
        qualityIndex (int): Index qui correspond à l'écart type de l'histogramme des pixels.
    """

    def __init__(self, height, contour, firstPosMeasure, secondPosMeasure, qualityIndex):
        """
        Constructeur pour la class Contour.
  
        Paramètres:
            height (int): Hauteur mesuré.
            contour ([[int]]): Liste des points du contour.
            firstPosMeasure (Tuple(int, int)): Coordonnée (x, y) du premier point de la mesure de la hauteur.
            pos2 (Tuple(int, int)): Coordonnée (x, y) du deuxième point de la mesure de la hauteur.
            qualityIndex (int): Index qui correspond à l'écart type de l'histogramme des pixels.
        """

        self.height = height
        self.contour = contour
        self.firstPosMeasure = firstPosMeasure
        self.secondPosMeasure = secondPosMeasure
        self.qualityIndex = qualityIndex
