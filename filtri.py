import numpy as np
import cv2
#Applicazione effetto mascheratura regione
def mask(immagine_input):
    matrice_maschera = np.zeros(immagine_input.shape[:2], dtype="uint8")
    cv2.rectangle(matrice_maschera, (500, 340), (800, 720), 255, -1)
    immagine_mascherata = cv2.bitwise_and(immagine_input, immagine_input, mask=matrice_maschera)
    return immagine_mascherata

#Modifica luminosità e contrasto
def adjust_contrast(immagine_input, alpha=1.0, beta=0):
    immagine_regolata = cv2.convertScaleAbs(immagine_input, alpha=alpha, beta=beta)
    return immagine_regolata

