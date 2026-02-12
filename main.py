import cv2
from Calcolo_metriche import compare_video_sequences
from filtri import jpeg_compression, blurring, unsharp_masking
from tracciamento import Tracciamento


#Caricamento sorgente video
cattura_video = cv2.VideoCapture('video/birds1.mp4')
if not cattura_video.isOpened():
    print("Impossibile trovare il video!")
    exit()

# Acquisizione fotogramma iniziale per determinare le dimensioni ed il ROI
stato_lettura, fotogramma_iniziale = cattura_video.read()
if not stato_lettura:
    print("Acquisizione fotogramma iniziale fallita!")
    exit()

# Estrazione dimensioni video
altezza_fotogramma, larghezza_fotogramma = fotogramma_iniziale.shape[:2]
print(f"Risoluzione video rilevata: {larghezza_fotogramma}x{altezza_fotogramma}")

# Selezione della ROI (Region of Interest)
print("CONFIGURAZIONE ROI")
print("Modalità disponibili:")
print("1. Definizione manuale tramite interfaccia grafica")
print("2. Configurazione automatica (area centrale)")
print("3. Utilizzo fotogramma completo\n")

scelta_modalita = input("Seleziona modalità operativa (1/2/3) [predefinita: 2]: ").strip()

if scelta_modalita == "1":
    # Modalità selezione manuale
    print("\nSeleziona l'area utilizzando il cursore:")
    print("- Traccia un rettangolo")
    print("- Conferma con INVIO")
    print("- Premi C per reimpostare la selezione")
    coordinate_roi = cv2.selectROI("Definizione ROI", fotogramma_iniziale, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Definizione ROI")
    
    if coordinate_roi == (0, 0, 0, 0):
        print("Nessuna area definita, applicazione configurazione automatica")
        scelta_modalita = "2"
    else:
        pos_x_roi, pos_y_roi, larghezza_roi, altezza_roi = coordinate_roi
        print(f"Area configurata: x={pos_x_roi}, y={pos_y_roi}, larghezza={larghezza_roi}, altezza={altezza_roi}")
elif scelta_modalita == "3":
    # Modalità fotogramma completo
    pos_x_roi, pos_y_roi, larghezza_roi, altezza_roi = 0, 0, larghezza_fotogramma, altezza_fotogramma
    print(f"Utilizzo area completa: {larghezza_fotogramma}x{altezza_fotogramma}")
else:
    scelta_modalita = "2"

if scelta_modalita == "2":
    # Configurazione automatica: area centrale, 60% delle dimensioni totali
    percentuale_larghezza_roi = 0.6
    percentuale_altezza_roi = 0.6
    
    larghezza_roi = int(larghezza_fotogramma * percentuale_larghezza_roi)
    altezza_roi = int(altezza_fotogramma * percentuale_altezza_roi)
    pos_x_roi = (larghezza_fotogramma - larghezza_roi) // 2
    pos_y_roi = (altezza_fotogramma - altezza_roi) // 2
    
    print(f"Configurazione automatica applicata: x={pos_x_roi}, y={pos_y_roi}, larghezza={larghezza_roi}, altezza={altezza_roi}")
