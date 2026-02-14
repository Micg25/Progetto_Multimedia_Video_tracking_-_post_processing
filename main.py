import cv2
from Calcolo_metriche import compare_video_sequences
from filtri import jpeg_compression, blurring, unsharp_masking, moving_average_filter, VideoStabilizer
from tracciamento import Tracciamento, filtra_box_sovrapposti


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

#Inizializzazione sistemi tracciamento con distanza configurata
sistema_tracciamento = Tracciamento(distanza_max=distanza_euclidea_max, use_feature_matching=True)  # Per frame originali con Feature Matching
sistema_tracciamento_stabilized_ma = Tracciamento(distanza_max=distanza_euclidea_max, use_feature_matching=True)  # Per frame stabilizzati + Moving Average SPAZIALE + Feature Matching

# Riposizionamento cattura video all'inizio
cattura_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

#Inizializzazione rilevatori sfondo con algoritmo MOG2
rilevatore_entita = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40)  # Per frame originali
rilevatore_entita_stabilized_ma = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40)  # Per frame stabilizzati + Moving Average 

#Inizializzazione variabili
fotogramma_prec = None
regione_prec = None
sequenza_fotogrammi = []
fotogrammi_jpeg = []  # Compressione JPEG
fotogrammi_blurred = []  # Gaussian blurring
fotogrammi_sharpened = []  # Sharpening

flag_debug_stampato = False

# Mostra frame iniziale e attendi pressione SPAZIO per iniziare
print("CONTROLLI VIDEO:")
print("- Premi SPAZIO per avviare il tracciamento")
print("- Durante il tracciamento: premi SPAZIO per pausa/ripresa")
print("- Premi Q per uscire")

stato_lettura_iniziale, frame_iniziale = cattura_video.read()
if stato_lettura_iniziale:
    cv2.imshow("Premi SPAZIO per iniziare", frame_iniziale)
    while True:
        tasto = cv2.waitKey(30) & 0xFF
        if tasto == ord(' '):  # SPAZIO premuto
            break
        elif tasto == ord('q'):  # Q per uscire
            cattura_video.release()
            cv2.destroyAllWindows()
            exit()
    cv2.destroyWindow("Premi SPAZIO per iniziare")
    cattura_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Riavvolgi al primo frame

# Inizializza stabilizzatore video con il primo frame come riferimento
print("\nInizializzazione stabilizzatore video...")
stato_ref, frame_riferimento = cattura_video.read()
if stato_ref:
    video_stabilizer = VideoStabilizer(frame_riferimento, smoothing_radius=5)
    print("✓ Stabilizzatore inizializzato")
    cattura_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Riavvolgi di nuovo
else:
    print("✗ Errore nell'inizializzazione dello stabilizzatore")
    video_stabilizer = None

# Flag per gestire pausa
in_pausa = False

while True:
    # Gestione pausa PRIMA di acquisire nuovo frame
    if in_pausa:
        tasto = cv2.waitKey(30) & 0xFF
        if tasto == ord(' '):  # SPAZIO per riprendere
            in_pausa = False
            print("▶ Ripresa tracciamento\n")
        elif tasto == ord('q'):
            break
        continue  # Rimani in pausa, non acquisire nuovo frame
    
    # Acquisizione fotogramma successivo
    stato_acquisizione, fotogramma_attuale = cattura_video.read()
    if not stato_acquisizione:
        break

    # Estrazione area di interesse con coordinate definite - COPIA PULITA per MOG2
    regione_interesse_pulita = fotogramma_attuale[pos_y_roi:pos_y_roi+altezza_roi, pos_x_roi:pos_x_roi+larghezza_roi].copy()
    
    # Fotogramma completo per calcolo metriche (non ROI)
    sequenza_fotogrammi.append(fotogramma_attuale.copy())
    fotogramma_prec = fotogramma_attuale.copy()
    regione_prec = regione_interesse_pulita.copy()
    
    # TRACCIAMENTO SU FRAME ORIGINALE
    # Fase 1: Identificazione entità presenti su COPIA PULITA
    maschera_rilevamento = rilevatore_entita.apply(regione_interesse_pulita.copy())
    _, maschera_rilevamento = cv2.threshold(maschera_rilevamento, 254, 255, cv2.THRESH_BINARY)
    
    # Operazioni morfologiche per unire frammenti della stessa auto
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    maschera_rilevamento = cv2.morphologyEx(maschera_rilevamento, cv2.MORPH_CLOSE, kernel_morph, iterations=2)
    maschera_rilevamento = cv2.morphologyEx(maschera_rilevamento, cv2.MORPH_OPEN, kernel_morph, iterations=1)
    
    contorni_rilevati, _ = cv2.findContours(maschera_rilevamento, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rilevamenti_validi = []
    for singolo_contorno in contorni_rilevati:
        # Valutazione area contorno e filtraggio elementi piccoli
        superficie_contorno = cv2.contourArea(singolo_contorno)
        if superficie_contorno > 500:  # Area minima aumentata
            coord_x, coord_y, dimensione_w, dimensione_h = cv2.boundingRect(singolo_contorno)
            rilevamenti_validi.append([coord_x, coord_y, dimensione_w, dimensione_h])
    
    # Filtra box sovrapposti usando NMS con IoU threshold
    rilevamenti_validi = filtra_box_sovrapposti(rilevamenti_validi, iou_threshold=0.3)
    
    # Crea copia per visualizzazione con box
    regione_interesse = regione_interesse_pulita.copy()

    # Fase 2: Tracciamento con FEATURE MATCHING su frame originale
    # Passa il frame ROI per l'estrazione delle features
    risultati_tracciamento = sistema_tracciamento.update(
        rilevamenti_validi,
        frame_roi=regione_interesse_pulita  # Frame per feature extraction
    )
    
    # Visualizzazione con box e ID
    for singolo_risultato in risultati_tracciamento:
        id_entita, coord_x, coord_y, dimensione_w, dimensione_h = singolo_risultato
        
        # Box e ID
        cv2.putText(regione_interesse, str(id_entita), (coord_x, coord_y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(regione_interesse, (coord_x, coord_y), (coord_x + dimensione_w, coord_y + dimensione_h), (0, 255, 0), 3)
    
    # TRACCIAMENTO SU FRAME STABILIZZATO + MOVING AVERAGE SPAZIALE
    if video_stabilizer is not None:
        # Fase 1: Stabilizzazione (compensa movimenti camera)
        fotogramma_stabilized = video_stabilizer.stabilize(fotogramma_attuale.copy())
        
        # Fase 2: Applica filtro media mobile SPAZIALE su frame stabilizzato
        fotogramma_stabilized_ma_tracking = moving_average_filter(
            fotogramma_stabilized, 
            kernel_size=5  # Kernel 5x5 per smoothing spaziale
        )
    else:
        # Fallback: solo moving average se stabilizzatore non disponibile
        fotogramma_stabilized_ma_tracking = moving_average_filter(
            fotogramma_attuale.copy(), 
            kernel_size=5
        )
    
    # Estrai ROI pulita per MOG2
    regione_interesse_stabilized_ma_pulita = fotogramma_stabilized_ma_tracking[pos_y_roi:pos_y_roi+altezza_roi, pos_x_roi:pos_x_roi+larghezza_roi].copy()
    
    # Fase 3: Identificazione entità presenti su frame stabilizzato+MA (su copia pulita senza box)
    maschera_rilevamento_stabilized_ma = rilevatore_entita_stabilized_ma.apply(regione_interesse_stabilized_ma_pulita.copy())
    _, maschera_rilevamento_stabilized_ma = cv2.threshold(maschera_rilevamento_stabilized_ma, 254, 255, cv2.THRESH_BINARY)
    
    # Operazioni morfologiche per unire frammenti della stessa auto
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    maschera_rilevamento_stabilized_ma = cv2.morphologyEx(maschera_rilevamento_stabilized_ma, cv2.MORPH_CLOSE, kernel_morph, iterations=2)
    maschera_rilevamento_stabilized_ma = cv2.morphologyEx(maschera_rilevamento_stabilized_ma, cv2.MORPH_OPEN, kernel_morph, iterations=1)
    
    contorni_rilevati_stabilized_ma, _ = cv2.findContours(maschera_rilevamento_stabilized_ma, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rilevamenti_validi_stabilized_ma = []
    for singolo_contorno in contorni_rilevati_stabilized_ma:
        superficie_contorno = cv2.contourArea(singolo_contorno)
        if superficie_contorno > 500:  # Area minima aumentata
            coord_x, coord_y, dimensione_w, dimensione_h = cv2.boundingRect(singolo_contorno)
            rilevamenti_validi_stabilized_ma.append([coord_x, coord_y, dimensione_w, dimensione_h])
    
    # Filtra box sovrapposti usando NMS con IoU threshold
    rilevamenti_validi_stabilized_ma = filtra_box_sovrapposti(rilevamenti_validi_stabilized_ma, iou_threshold=0.3)
    
    # Crea NUOVA copia per visualizzazione con box di tracciamento
    regione_interesse_stabilized_ma = regione_interesse_stabilized_ma_pulita.copy()
    
    # Fase 4: Tracciamento con FEATURE MATCHING su frame stabilizzato+MA
    # Passa il frame ROI per l'estrazione delle features
    risultati_tracciamento_stabilized_ma = sistema_tracciamento_stabilized_ma.update(
        rilevamenti_validi_stabilized_ma, 
        frame_roi=regione_interesse_stabilized_ma_pulita  # Frame per feature extraction
    )
    
    # Visualizzazione con box e ID
    for singolo_risultato in risultati_tracciamento_stabilized_ma:
        id_entita, coord_x, coord_y, dimensione_w, dimensione_h = singolo_risultato
        
        # Box e ID
        cv2.putText(regione_interesse_stabilized_ma, str(id_entita), (coord_x, coord_y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(regione_interesse_stabilized_ma, (coord_x, coord_y), (coord_x + dimensione_w, coord_y + dimensione_h), (0, 255, 0), 3)

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
    cv2.imshow("ROI + Tracking (Original + Features)", regione_interesse)
    cv2.imshow("ROI + Tracking (Stabilized+MA + Features)", regione_interesse_stabilized_ma)
    cv2.imshow("Original", fotogramma_attuale)
    cv2.imshow("JPEG Compressed (Q=50)", fotogramma_jpeg)
    cv2.imshow("Gaussian Blur", fotogramma_blurred)
    cv2.imshow("Sharpened (Unsharp)", fotogramma_sharpened)

    # Visualizzazione maschere background subtraction (per debug tracking)
    cv2.namedWindow("Background Mask (Original + Features)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Background Mask (Original + Features)", 400, 300) 
    maschera_ridimensionata = cv2.resize(maschera_rilevamento, (400, 300)) 
    cv2.imshow("Background Mask (Original + Features)", maschera_ridimensionata)
    
    cv2.namedWindow("Background Mask (Stabilized+MA + Features)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Background Mask (Stabilized+MA + Features)", 400, 300) 
    maschera_stabilized_ma_ridimensionata = cv2.resize(maschera_rilevamento_stabilized_ma, (400, 300)) 
    cv2.imshow("Background Mask (Stabilized+MA + Features)", maschera_stabilized_ma_ridimensionata)

    # Gestione controlli tastiera
    tasto = cv2.waitKey(1) & 0xFF
    
    if tasto == ord('q'):
        break
    elif tasto == ord(' '):  # SPAZIO per mettere in pausa
        in_pausa = True
        print("\n⏸ VIDEO IN PAUSA - Premi SPAZIO per riprendere")
    
cattura_video.release()
cv2.destroyAllWindows()

# Stampa numero totale di oggetti individuati per entrambi i tracciamenti
numero_oggetti_totali_originale = sistema_tracciamento.get_numero_oggetti_totali()
numero_oggetti_totali_stabilized_ma = sistema_tracciamento_stabilized_ma.get_numero_oggetti_totali()

print(f"\nTRACCIAMENTO COMPLETATO")
print(f"Frame ORIGINALE:")
print(f"  • MOG2 Background Subtraction")
print(f"  • Feature Matching (ORB + Brute Force matcher)")
print(f"  • Coordinate Smoothing (alpha=0.3)")
print(f"  Numero totale di oggetti individuati: {numero_oggetti_totali_originale}")

print(f"\nFrame STABILIZZATO + MOVING AVERAGE SPAZIALE:")
if video_stabilizer is not None:
    print(f"  • Video Stabilization (ORB features + Affine Transform)")
print(f"  • Filtro Moving Average Spaziale (kernel 5x5)")
print(f"  • MOG2 Background Subtraction")
print(f"  • Feature Matching (ORB + Brute Force matcher)")
print(f"  • Coordinate Smoothing (alpha=0.3)")
print(f"  Numero totale di oggetti individuati: {numero_oggetti_totali_stabilized_ma}")
    

print("CONFRONTO TRACCIAMENTI:")
print(f"  Originale vs Stabilizzato+MA: {numero_oggetti_totali_originale - numero_oggetti_totali_stabilized_ma:+d} oggetti")
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
    