import numpy as np
import cv2
from collections import deque

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


def moving_average_filter(immagine_input, kernel_size=5):
    
   # Applica un filtro Media Mobile SPAZIALE all'immagine.
   # calcolando la media dei pixel vicini con pesi uguali.
   
    # Assicura che kernel_size sia dispari
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Crea kernel con pesi uniformi (media semplice)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    
    # Applica convoluzione 2D per calcolare la media mobile
    immagine_filtrata = cv2.filter2D(immagine_input, -1, kernel)
    
    return immagine_filtrata


class TemporalMovingAverageFilter:
    
    # Filtro Media Mobile TEMPORALE che mantiene un buffer di frame
    #e calcola la media tra frame consecutivi per ridurre flickering e rumore temporale.
   
    def __init__(self, buffer_size=5):
        """
        Args:
            buffer_size: Numero di frame da mantenere nel buffer per la media (default=5)
        """
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
    
    def apply(self, frame):
        """
        Applica il filtro temporale al frame corrente.
        
        Args:
            frame: Frame corrente da filtrare
            
        Returns:
            Frame filtrato con media mobile temporale
        """
        # Aggiungi frame corrente al buffer
        self.frame_buffer.append(frame.copy())
        
        # Se il buffer non è ancora pieno, ritorna il frame originale
        if len(self.frame_buffer) < self.buffer_size:
            return frame
        
        # Calcola la media di tutti i frame nel buffer
        frame_medio = np.mean(self.frame_buffer, axis=0).astype(np.uint8)
        
        return frame_medio
    
    def reset(self):
        """Resetta il buffer (utile quando si cambia video o si ricomincia)"""
        self.frame_buffer.clear()


def moving_average_spatio_temporal(immagine_input, temporal_filter, spatial_kernel_size=3):
    
    #Applica filtro Media Mobile combinato SPAZIO-TEMPORALE.
    #Prima applica filtro temporale, poi spaziale per massima riduzione rumore.
  
    # 1. Applica filtro TEMPORALE (riduce flickering tra frame)
    frame_temporal_filtered = temporal_filter.apply(immagine_input)
    
    # 2. Applica filtro SPAZIALE (riduce rumore pixel)
    frame_spatio_temporal = moving_average_filter(frame_temporal_filtered, kernel_size=spatial_kernel_size)
    
    return frame_spatio_temporal

