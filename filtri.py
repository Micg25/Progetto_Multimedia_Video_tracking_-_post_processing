import numpy as np
import cv2
#Applicazione effetto mascheratura regione
def mask(immagine_input):
    matrice_maschera = np.zeros(immagine_input.shape[:2], dtype="uint8")
    cv2.rectangle(matrice_maschera, (500, 340), (800, 720), 255, -1)
    immagine_mascherata = cv2.bitwise_and(immagine_input, immagine_input, mask=matrice_maschera)
    return immagine_mascherata
