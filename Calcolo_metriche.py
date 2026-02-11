import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_video_sequences(sequenza_originale, sequenza_filtrata):
    
    # Verifica che le sequenze abbiano lo stesso numero di frame
    if len(sequenza_originale) != len(sequenza_filtrata):
        print(f"ERRORE: Le due sequenze hanno lunghezze diverse!")
        print(f"  Originale: {len(sequenza_originale)} frame")
        print(f"  Filtrata:  {len(sequenza_filtrata)} frame")
        return None, None, None
    
    if len(sequenza_originale) < 1:
        print("ERRORE: Sequenze vuote!")
        return None, None, None
    
    numero_fotogrammi = len(sequenza_originale)
    
    somma_mse = 0
    somma_psnr = 0
    somma_ssim = 0
    
    lista_psnr = []
    
    # Confronta ogni coppia di frame corrispondenti
    for indice in range(numero_fotogrammi):
        frame_orig = sequenza_originale[indice]
        frame_filt = sequenza_filtrata[indice]
        
        # Gestione differenze di dimensione/canali
        if frame_orig.shape != frame_filt.shape:
            # Se uno è grayscale e l'altro RGB, converti entrambi a grayscale
            if len(frame_orig.shape) == 3 and len(frame_filt.shape) == 2:
                frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
            elif len(frame_orig.shape) == 2 and len(frame_filt.shape) == 3:
                frame_filt = cv2.cvtColor(frame_filt, cv2.COLOR_BGR2GRAY)
            
            # Se le dimensioni diverse, segnala errore
            if frame_orig.shape != frame_filt.shape:
                print(f"ERRORE al frame {indice}: dimensioni incompatibili dopo conversione")
                print(f"  Originale: {frame_orig.shape}")
                print(f"  Filtrato:  {frame_filt.shape}")
                return None, None, None
        
        # Calcolo MSE tra frame originale e filtrato
        errore_quadratico = np.mean((frame_orig.astype(float) - frame_filt.astype(float)) ** 2)
        somma_mse += errore_quadratico
        
        # Calcolo PSNR
        if errore_quadratico == 0:
            rapporto_segnale = 100.0
        else:
            valore_massimo_pixel = 255.0
            rapporto_segnale = 20 * np.log10(valore_massimo_pixel / np.sqrt(errore_quadratico))
        
        lista_psnr.append(rapporto_segnale)
        somma_psnr += rapporto_segnale
        
        # Calcolo SSIM (richiede grayscale)
        if len(frame_orig.shape) == 3:
            # Converti a grayscale per SSIM
            frame_orig_gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
            frame_filt_gray = cv2.cvtColor(frame_filt, cv2.COLOR_BGR2GRAY) if len(frame_filt.shape) == 3 else frame_filt
        else:
            frame_orig_gray = frame_orig
            frame_filt_gray = frame_filt
        
        punteggio_similarita, _ = ssim(frame_orig_gray, frame_filt_gray, full=True)
        somma_ssim += punteggio_similarita
    
    # Calcolo medie finali
    mse_medio = somma_mse / numero_fotogrammi
    psnr_medio = somma_psnr / numero_fotogrammi
    ssim_medio = somma_ssim / numero_fotogrammi
    
    return mse_medio, psnr_medio, ssim_medio
