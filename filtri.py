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


class VideoStabilizer:

    #Stabilizzazione video feature-based usando ORB e trasformazioni affini.
    #Riduce vibrazioni della camera e falsi positivi nel tracking.
   
    
    def __init__(self, reference_frame, smoothing_radius=5):
        self.reference_frame = reference_frame.copy()
        self.smoothing_radius = smoothing_radius
        
        # Detector ORB per feature extraction
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        
        # Matcher per feature matching
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Rileva features nel frame di riferimento
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(
            cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY), None
        )
        
        # Buffer per smoothing trasformazioni
        self.transform_buffer = deque(maxlen=smoothing_radius)
        
        # Dimensioni frame
        self.height, self.width = reference_frame.shape[:2]
        
    def stabilize(self, frame):
        #Stabilizza il frame corrente rispetto al frame di riferimento.

        if self.ref_descriptors is None or len(self.ref_descriptors) == 0:
            return frame  # Fallback: ritorna frame originale
        
        # Converti in grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Rileva features nel frame corrente
        curr_keypoints, curr_descriptors = self.orb.detectAndCompute(gray, None)
        
        if curr_descriptors is None or len(curr_descriptors) == 0:
            return frame  # Nessuna feature trovata
        
        # Match features tra riferimento e frame corrente
        matches = self.matcher.knnMatch(self.ref_descriptors, curr_descriptors, k=2)
        
        # Filtra buoni match usando Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:  # Ratio test
                    good_matches.append(m)
        
        # Serve almeno 4 match per calcolare omografia
        if len(good_matches) < 10:
            return frame  # Non abbastanza match, ritorna originale
        
        # Estrai punti corrispondenti
        ref_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good_matches])
        curr_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in good_matches])
        
        # Calcola matrice di trasformazione (affine per maggiore stabilità)
        # Usa RANSAC per gestire outlier
        transform_matrix, mask = cv2.estimateAffinePartial2D(
            curr_pts, ref_pts, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=3.0
        )
        
        if transform_matrix is None:
            return frame  # Trasformazione non calcolabile
        
        # Applica smoothing alla trasformazione per evitare scatti
        self.transform_buffer.append(transform_matrix)
        
        if len(self.transform_buffer) > 0:
            # Media delle trasformazioni nel buffer
            smooth_transform = np.mean(self.transform_buffer, axis=0)
        else:
            smooth_transform = transform_matrix
        
        # Applica trasformazione per stabilizzare il frame
        stabilized_frame = cv2.warpAffine(
            frame, 
            smooth_transform, 
            (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return stabilized_frame
    
    def reset_reference(self, new_reference_frame):

        #Aggiorna il frame di riferimento
       
        self.reference_frame = new_reference_frame.copy()
        gray = cv2.cvtColor(new_reference_frame, cv2.COLOR_BGR2GRAY)
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(gray, None)
        self.transform_buffer.clear()

