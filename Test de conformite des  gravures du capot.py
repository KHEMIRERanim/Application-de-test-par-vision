import cv2
import numpy as np
import tensorflow as tf

def main():
    # Chemin vers le modèle CNN au format .h5
    chemin_modele_cnn = r'C:\Users\INFOKOM\Desktop\Test_par_vision\gravurecapot_cnn.h5'
    
    # Chemin vers l'image à tester
    chemin_image = r'C:\Users\INFOKOM\Desktop\Test_par_vision\test-gravure_capot_teststand\GRAVURE CAPOT_2023-05-03 12.51.02,955.jpg'

    # Charger le modèle CNN pour la classification de conformité
    modele_cnn = tf.keras.models.load_model(chemin_modele_cnn)

    # Configuration des paramètres pour le prétraitement de l'image
    hauteur_image = 791
    largeur_image = 1036

    # Fonction pour prédire la conformité avec le modèle CNN
    def predire_conformite(image):
        # Redimensionner et normaliser l'image
        image_redimensionnee = cv2.resize(image, (largeur_image, hauteur_image))
        image_redimensionnee = image_redimensionnee / 255.0

        # Ajouter une dimension pour correspondre à l'entrée du modèle
        image_redimensionnee = np.expand_dims(image_redimensionnee, axis=0)

        # Prédiction
        prediction = modele_cnn.predict(image_redimensionnee)[0]
        if prediction[0] > 0.5:
            return "Passed"  # Renvoyer "Passed" si conforme
        else:
            return "Failed"  # Renvoyer "Failed" sinon

    # Charger l'image
    image = cv2.imread(chemin_image)

    # Vérifier si l'image a été chargée correctement
    if image is None:
        print(f"Impossible de charger l'image : {chemin_image}")
        return "Failed"  # Renvoyer "Failed" si l'image ne peut pas être chargée

    # Prédire la conformité avec le modèle CNN
    resultat = predire_conformite(image)
    print(resultat)
    return resultat

if __name__ == "__main__":
    # Exécuter le programme
    main()
