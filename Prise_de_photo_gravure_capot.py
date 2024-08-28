import os
import cv2

def capture_and_save_image():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()

    if not ret:
        return "failed"

    folder_path = r'C:\Users\INFOKOM\Desktop\Test_de_vision\test_prise_de_photo_gravure_capot_teststand'

    img_name = "captured_image.png"
    save_path = os.path.join(folder_path, img_name)

    cv2.imwrite(save_path, frame)
    cam.release()
    cv2.destroyAllWindows()

    return "passed"

# Appel de la fonction pour capturer et enregistrer l'image
resultat = capture_and_save_image()
print(resultat)  # Affichage de la valeur de sortie
