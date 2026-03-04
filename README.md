# Progetto_Multimedia_Video_tracking_-_post_processing
Progetto per la materia Multimedia della Magistrale di Informatica

## Descrizione
Sistema di video tracking e post-processing che permette di analizzare video, tracciare oggetti in movimento e applicare diversi filtri e stabilizzazioni.

## Prerequisiti
- Python 3.7 o superiore
- OpenCV
- NumPy
- Streamlit (per l'interfaccia web)

## Installazione Dipendenze
```bash
pip install opencv-python numpy streamlit
```

## Come Avviare il Progetto

### Opzione 1: Interfaccia Web con Streamlit (Consigliato)

#### Su Windows:
Eseguire il file batch fornito che avvia automaticamente l'applicazione:
```bash
run_app.bat
```

#### Su qualsiasi piattaforma:
```bash
streamlit run app.py
```

L'applicazione si aprirà automaticamente nel browser predefinito all'indirizzo `http://localhost:8501`.

#### Funzionalità dell'interfaccia Streamlit:
- Caricamento video tramite interfaccia grafica
- Selezione interattiva della ROI (Region of Interest)
- Configurazione parametri di tracking
- Applicazione filtri video (JPEG compression, blur, unsharp masking, moving average)
- Stabilizzazione video
- Calcolo metriche di confronto (MSE, PSNR, SSIM)
- Visualizzazione risultati in tempo reale
- Download video processati

### Opzione 2: Esecuzione Script Standalone

Per eseguire il progetto senza interfaccia grafica, utilizzare il file `main.py`:

```bash
python main.py
```

#### Funzionalità dello script standalone:
- Caricamento video da path predefinito (`video/traffico1.mp4`)
- Selezione ROI tramite:
  1. **Manuale**: interfaccia grafica per disegnare l'area di interesse
  2. **Automatica**: area centrale del video
  3. **Completa**: utilizzo dell'intero fotogramma
- Tracciamento automatico degli oggetti
- Applicazione filtri configurabili nel codice
- Salvataggio video processati
- Calcolo e visualizzazione metriche

#### Note per l'esecuzione da main.py:
- Assicurarsi che il video da analizzare sia presente nella cartella `video/`
- Il path del video può essere modificato direttamente nel file `main.py` (linea 9)
- Durante l'esecuzione, seguire le istruzioni nel terminale per la configurazione della ROI

## Struttura del Progetto
- `app.py` - Applicazione Streamlit con interfaccia web
- `main.py` - Script standalone per esecuzione da terminale
- `tracciamento.py` - Modulo per il tracking degli oggetti
- `filtri.py` - Modulo con implementazione dei filtri video
- `Calcolo_metriche.py` - Modulo per il calcolo delle metriche (MSE, PSNR, SSIM)
- `run_app.bat` - Script batch per avvio rapido su Windows
- `video/` - Cartella per i video di input
- `immagini/` - Cartella per le immagini e output

## Utilizzo
1. Caricare o selezionare il video da analizzare
2. Configurare la ROI (Region of Interest)
3. Impostare i parametri di tracking
4. Applicare eventuali filtri di post-processing
5. Visualizzare e salvare i risultati
