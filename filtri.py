import numpy as np
import cv2

def jpeg_compression(immagine_input, quality=50):
    # Codifica l'immagine in formato JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', immagine_input, encode_param)
    
    #Decodifica l'immagine compressa
    immagine_compressa = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    return immagine_compressa


def blurring(immagine_input, kernel_size=(5, 5), sigma=0):

    immagine_sfocata = cv2.GaussianBlur(immagine_input, kernel_size, sigma)
    return immagine_sfocata


def unsharp_masking(immagine_input, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    #Crea versione sfocata
    blurred = cv2.GaussianBlur(immagine_input, kernel_size, sigma)
    
    # Calcola la maschera (differenza tra originale e sfocato)
    mask = cv2.subtract(immagine_input, blurred)
    
    #Applica unsharp masking con threshold
    if threshold > 0:
        # Applica sharpening solo dove la differenza supera la soglia
        low_contrast_mask = np.absolute(mask) < threshold
        np.copyto(mask, 0, where=low_contrast_mask)
    
    #Combina: Originale + amount * Maschera
    immagine_sharpened = cv2.addWeighted(
        immagine_input, 1.0, 
        mask, amount, 
        0
    )
    
    # Clip valori nell'intervallo [0, 255]
    immagine_sharpened = np.clip(immagine_sharpened, 0, 255).astype(np.uint8)
    
    return immagine_sharpened
