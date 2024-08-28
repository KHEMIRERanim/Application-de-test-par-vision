import cv2

# Définir les variables globales
drawing = False # True si le bouton de la souris est enfoncé
ix, iy = -1, -1 # Coordonnées du premier point du rectangle

# Fonction de rappel de la souris
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, img_with_rect

    # Si le bouton gauche de la souris est enfoncé, enregistrer les coordonnées du premier point du rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # Si la souris est en mouvement et le bouton gauche est enfoncé, dessiner le rectangle temporaire
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_temp = img.copy() # Créer une copie de l'image originale
        cv2.rectangle(img_temp, (ix, iy), (x, y), (255, 0, 0), 2) # Dessiner le rectangle temporaire
        cv2.imshow('image', img_temp)

    # Si le bouton gauche de la souris est relâché, dessiner le rectangle final et afficher les informations
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img_with_rect, (ix, iy), (x, y), (255, 0, 0), 2)
        cv2.imshow('image', img_with_rect)
        width = abs(x - ix)
        height = abs(y - iy)
        print("Coin supérieur gauche:", (ix, iy))
        print("Coin inférieur droit:", (x, y))
        print("Largeur:", width)
        print("Hauteur:", height)

# Charger l'image
img = cv2.imread('GRAVURE CAPOT_2023-05-03 12.48.11,675.jpg')

# Vérifier si l'image est chargée correctement
if img is None:
    print("Impossible de charger l'image.")
    exit()

img_with_rect = img.copy()

# Créer une fenêtre et associer la fonction de rappel de la souris
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

# Boucle principale
while True:
    cv2.imshow('image', img)
    key = cv2.waitKey(1) & 0xFF
    # Quitter la boucle si la touche 'q' est enfoncée
    if key == ord('q'):
        break
    # Enregistrer l'image si la touche 's' est enfoncée
    elif key == ord('s'):
        cv2.imwrite(r'C:\Users\INFOKOM\Desktop\Test_de_vision\image_with_rectangle.jpg', img_with_rect)
        print("Image enregistrée avec succès.")

# Fermer toutes les fenêtres OpenCV
cv2.destroyAllWindows()
