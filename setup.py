# importation des librairies
import cv2
from tkinter.filedialog import askopenfilename

refPt = []

# fonction de récupération des coordonnées des points cliqué sur l'image et affichage de ces coordonnées
def click_event(event, x, y, flags, params):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y})')
        
        # affichage des coordonnées du point cliqué sur l'image
        cv2.putText(img, f'({x},{y})',(x,y),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # affichage d'un cercle sur le point cliqué
        cv2.circle(img, (x,y), 2, (0,255,255), -1)

    ### Affichage d'un rectangle quand on utilise le clique droit
	# si le bouton gauche de la souris est cliqué, enregistre les
	# coordonnées du point de départ de la sélection
    if event == cv2.EVENT_RBUTTONDOWN:
        refPt = [(x, y)]
    # regarde si le bouton gauche de la souris est relaché
    elif event == cv2.EVENT_RBUTTONUP:
        # Enregistre les coordonnées du point de fin de la sélection
        refPt.append((x, y))
        # dessine un rectangle sur l'image
        cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 1)

        # affichage des coordonnées du rectangle
        cv2.circle(img, (refPt[0][0],refPt[0][1]), 2, (0,255,255), -1)
        cv2.circle(img, (refPt[1][0],refPt[1][1]), 2, (0,255,255), -1)
        cv2.putText(img, f'({refPt[0][0]},{refPt[0][1]})',refPt[0],
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f'({refPt[1][0]},{refPt[1][1]})',refPt[1],
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

 
# ouverture de la fenêtre de sélection de l'image
File = askopenfilename(title='Choisir une image.')
#cap = cv2.VideoCapture("rtsp://root:M101_svr@192.168.1.56:554/axis-media/media.amp")


# lecture de l'image
img = cv2.imread(File)
#ret, img = cap.read()

# création d'une copie de l'image
clone = img.copy()

# création de la fenêtre
cv2.namedWindow('Point Coordinates')

# appel de la fonction click_event
cv2.setMouseCallback('Point Coordinates', click_event)

# affichage de l'image
while True:
    cv2.imshow('Point Coordinates',img)
    k = cv2.waitKey(1) & 0xFF

    # réinitialisation de l'image quand on appuie sur la touche r
    if k == ord("r"):
        img = clone.copy()

    # fermeture de la fenêtre quand on appuie sur la touche echap
    elif k == 27:
        break
cv2.destroyAllWindows()