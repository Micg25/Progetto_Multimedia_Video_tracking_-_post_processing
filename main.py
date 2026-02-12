import cv2
from Calcolo_metriche import compare_video_sequences
from filtri import jpeg_compression, blurring, unsharp_masking
from tracciamento import Tracciamento


#Caricamento sorgente video
cattura_video = cv2.VideoCapture('video/traffico1.mp4')
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

# Configurazione distanza euclidea per tracciamento
print("\nCONFIGURAZIONE TRACCIAMENTO OGGETTI")
print("La distanza euclidea determina quanto un oggetto può spostarsi tra frame consecutivi")
print("mantenendo lo stesso ID. Valori tipici: 50-500 pixel.")
print("- Valori BASSI (50-150): oggetti lenti, movimenti ridotti")
print("- Valori MEDI (150-350): situazioni standard")
print("- Valori ALTI (350-500): oggetti veloci, grandi spostamenti\n")

distanza_input = input("Inserisci distanza euclidea massima in pixel [predefinita: 350]: ").strip()
if distanza_input:
    try:
        distanza_euclidea_max = int(distanza_input)
        if distanza_euclidea_max <= 0:
            print("Valore non valido, utilizzo valore predefinito: 350")
            distanza_euclidea_max = 350
    except ValueError:
        print("Valore non valido, utilizzo valore predefinito: 350")
        distanza_euclidea_max = 350
else:
    distanza_euclidea_max = 350

print(f"Distanza euclidea configurata: {distanza_euclidea_max} pixel")

#Inizializzazione sistema tracciamento con distanza configurata
sistema_tracciamento = Tracciamento(distanza_max=distanza_euclidea_max)

# Riposizionamento cattura video all'inizio
cattura_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

#Inizializzazione rilevatore sfondo con algoritmo MOG2
rilevatore_entita = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40) 

#Inizializzazione variabili
fotogramma_prec = None
regione_prec = None
sequenza_fotogrammi = []
fotogrammi_jpeg = []  # Compressione JPEG
fotogrammi_blurred = []  # Gaussian blurring
fotogrammi_sharpened = []  # Sharpening

flag_debug_stampato = False
while True:
    # Acquisizione fotogramma successivo
    stato_acquisizione, fotogramma_attuale = cattura_video.read()
    if not stato_acquisizione:
        break

    # Estrazione area di interesse con coordinate definite
    regione_interesse = fotogramma_attuale[pos_y_roi:pos_y_roi+altezza_roi, pos_x_roi:pos_x_roi+larghezza_roi]
    
    # Fotogramma completo per calcolo metriche (non ROI)
    sequenza_fotogrammi.append(fotogramma_attuale.copy())
    fotogramma_prec = fotogramma_attuale.copy()
    regione_prec = regione_interesse.copy()
    
    # Fase 1: Identificazione entità presenti
    maschera_rilevamento = rilevatore_entita.apply(regione_interesse)
    _, maschera_rilevamento = cv2.threshold(maschera_rilevamento, 254, 255, cv2.THRESH_BINARY)
    contorni_rilevati, _ = cv2.findContours(maschera_rilevamento, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rilevamenti_validi = []
    for singolo_contorno in contorni_rilevati:
        # Valutazione area contorno e filtraggio elementi piccoli
        superficie_contorno = cv2.contourArea(singolo_contorno)
        if superficie_contorno > 300:
            coord_x, coord_y, dimensione_w, dimensione_h = cv2.boundingRect(singolo_contorno)
            rilevamenti_validi.append([coord_x, coord_y, dimensione_w, dimensione_h])

    # Fase 2: Tracciamento entità rilevate
    risultati_tracciamento = sistema_tracciamento.update(rilevamenti_validi)
    for singolo_risultato in risultati_tracciamento:
        id_entita, coord_x, coord_y, dimensione_w, dimensione_h = singolo_risultato
        cv2.putText(regione_interesse, str(id_entita), (coord_x, coord_y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(regione_interesse, (coord_x, coord_y), (coord_x + dimensione_w, coord_y + dimensione_h), (0, 255, 0), 3)

    # Applicazione algoritmi di qualità
    # Algoritmo 1: Compressione JPEG (quality=50) - Introduce artefatti
    fotogramma_jpeg = jpeg_compression(fotogramma_attuale.copy(), quality=50)
    fotogrammi_jpeg.append(fotogramma_jpeg)
    
    # Algoritmo 2: Gaussian Blurring - Smoothing generale
    fotogramma_blurred = blurring(fotogramma_attuale.copy())
    fotogrammi_blurred.append(fotogramma_blurred)
    
    # Algoritmo 3: Unsharp Masking - Aumenta nitidezza
    fotogramma_sharpened = unsharp_masking(fotogramma_attuale.copy(), amount=1.5)
    fotogrammi_sharpened.append(fotogramma_sharpened)
    
    # Output diagnostico dimensioni (solo primo ciclo)
    if not flag_debug_stampato:
        print("Dimensioni fotogramma JPEG:", fotogramma_jpeg.shape)
        print("Dimensioni fotogramma Blurred:", fotogramma_blurred.shape)
        print("Dimensioni fotogramma Sharpened:", fotogramma_sharpened.shape)
        flag_debug_stampato = True
    
    # Rendering finestre visualizzazione
    cv2.imshow("ROI + Tracking", regione_interesse)
    cv2.imshow("Original", fotogramma_attuale)
    cv2.imshow("JPEG Compressed (Q=50)", fotogramma_jpeg)
    cv2.imshow("Gaussian Blur", fotogramma_blurred)
    cv2.imshow("Sharpened (Unsharp)", fotogramma_sharpened)

    # Visualizzazione maschera background subtraction (per debug tracking)
    cv2.namedWindow("Background Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Background Mask", 400, 300) 
    maschera_ridimensionata = cv2.resize(maschera_rilevamento, (400, 300)) 
    cv2.imshow("Background Mask", maschera_ridimensionata)

    # Attesa minima per refresh finestre (1ms = massima fluidità)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cattura_video.release()
cv2.destroyAllWindows()

# Stampa numero totale di oggetti individuati
numero_oggetti_totali = sistema_tracciamento.get_numero_oggetti_totali()
print(f"TRACCIAMENTO COMPLETATO")
print(f"Numero totale di oggetti individuati: {numero_oggetti_totali}")


print("ANALISI METRICHE")
print(" (MSE, PSNR, SSIM)")


print("CONFRONTO: ORIGINALE vs FILTRATI")
print("Legenda metriche:")
print("  MSE:  Più BASSO = migliore (errore minore)")
print("  PSNR: Più ALTO = migliore (>30dB ottimo, <20dB scarso)")
print("  SSIM: Più ALTO = migliore (1=identico, 0=diverso)")

# Metriche: ORIGINALE vs JPEG Compression
mse_jpeg, psnr_jpeg, ssim_jpeg = compare_video_sequences(sequenza_fotogrammi, fotogrammi_jpeg)
if mse_jpeg is not None:
    print("\n[1] ORIGINALE vs JPEG COMPRESSION (Quality=50)")
    print("    Effetto: Introduce artefatti di compressione lossy")
    print(f"    MSE:  {mse_jpeg:.4f}  (atteso: ALTO)")
    print(f"    PSNR: {psnr_jpeg:.2f} dB  (atteso: BASSO/MEDIO, 25-35dB)")
    print(f"    SSIM: {ssim_jpeg:.4f}  (atteso: MEDIO/BASSO, 0.70-0.90)")

# Metriche: ORIGINALE vs Gaussian Blurring
mse_blur, psnr_blur, ssim_blur = compare_video_sequences(sequenza_fotogrammi, fotogrammi_blurred)
if mse_blur is not None:
    print("\n[2] ORIGINALE vs GAUSSIAN BLURRING")
    print("    Effetto: Sfocatura gaussiana per smoothing generale")
    print(f"    MSE:  {mse_blur:.4f}  (atteso: MEDIO)")
    print(f"    PSNR: {psnr_blur:.2f} dB  (atteso: MEDIO-ALTO, 28-35dB)")
    print(f"    SSIM: {ssim_blur:.4f}  (atteso: ALTO, 0.85-0.95)")

# Metriche: ORIGINALE vs Unsharp Masking
mse_sharp, psnr_sharp, ssim_sharp = compare_video_sequences(sequenza_fotogrammi, fotogrammi_sharpened)
if mse_sharp is not None:
    print("\n[3] ORIGINALE vs UNSHARP MASKING (Sharpening)")
    print("    Effetto: Aumenta nitidezza, possibili overshoot ai bordi")
    print(f"    MSE:  {mse_sharp:.4f}  (atteso: MEDIO)")
    print(f"    PSNR: {psnr_sharp:.2f} dB  (atteso: MEDIO/ALTO, 30-40dB)")
    print(f"    SSIM: {ssim_sharp:.4f}  (atteso: MEDIO/ALTO, 0.80-0.95)")

# Riepilogo comparativo

print("RIEPILOGO COMPARATIVO")

if all(v is not None for v in [mse_jpeg, mse_blur, mse_sharp]):
    print("\nRisultati finali")
    
    risultati = [
        ("JPEG Compression", mse_jpeg, psnr_jpeg, ssim_jpeg),
        ("Gaussian Blurring", mse_blur, psnr_blur, ssim_blur),
        ("Sharpening", mse_sharp, psnr_sharp, ssim_sharp)
    ]
    
    # Ordina per SSIM (più alto = migliore)
    risultati_ordinati = sorted(risultati, key=lambda x: x[3], reverse=True)
    
    for i, (nome, mse, psnr, ssim) in enumerate(risultati_ordinati, 1):
        print(f"\n{i}. {nome}")
        print(f"   MSE: {mse:.4f} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
    
    
    #print("NOTE:")
    #print("- JPEG Compression: degrada intenzionalmente per ridurre dimensione file")
    #print("- Gaussian Blur: riduce dettagli e rumore ad alta frequenza")
    #print("- Sharpening: migliora nitidezza percepita ma altera segnale originale")
    