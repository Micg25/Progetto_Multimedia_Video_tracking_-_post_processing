import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

# Import moduli esistenti
from Calcolo_metriche import compare_video_sequences
from filtri import VideoStabilizer, moving_average_filter, jpeg_compression, blurring, unsharp_masking
from tracciamento import Tracciamento, filtra_box_sovrapposti

st.set_page_config(
    page_title="Video Tracking & Post-Processing",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Video Tracking & Post-Processing Comparison")
st.markdown("### Confronto tra Video Originale e Video Filtrato con Tracking")

# Sidebar per configurazione
with st.sidebar:
    st.header("⚙️ Configurazione")
    
    # Opzione per usare video default o caricare nuovo
    use_default = st.checkbox("Usa video di default", value=True, help="Usa video/traffico1cut.mp4")
    
    # Upload video
    uploaded_file = None
    if not use_default:
        uploaded_file = st.file_uploader("Carica video personalizzato", type=['mp4', 'avi', 'mov'])
    
    st.divider()
    
    # Configurazione ROI
    st.subheader("📍 Region of Interest (ROI)")
    roi_mode = st.radio(
        "Modalità ROI",
        ["Automatica (centrale 60%)", "Fotogramma completo", "Manuale (personalizzata)"],
        help="ROI definisce l'area di tracciamento"
    )
    
    # Se modalità manuale, mostra slider per ROI personalizzata
    roi_manual_params = None
    if roi_mode == "Manuale (personalizzata)":
        st.info("💡 Imposta la ROI usando i parametri sotto. Premi 'Rielabora' per applicare.")
        
        # Valori di default ottimizzati per video traffico
        col_roi1, col_roi2 = st.columns(2)
        with col_roi1:
            roi_x = st.number_input("Posizione X", min_value=0, max_value=3000, value=40, step=10,
                                    help="Coordinata X dell'angolo in alto a sinistra")
            roi_width = st.number_input("Larghezza", min_value=50, max_value=3000, value=1210, step=10,
                                        help="Larghezza della ROI in pixel")
        with col_roi2:
            roi_y = st.number_input("Posizione Y", min_value=0, max_value=2000, value=270, step=10,
                                    help="Coordinata Y dell'angolo in alto a sinistra")
            roi_height = st.number_input("Altezza", min_value=50, max_value=2000, value=430, step=10,
                                         help="Altezza della ROI in pixel")
        
        roi_manual_params = (roi_x, roi_y, roi_width, roi_height)
        
        st.caption(f"📏 ROI selezionata: {roi_width}x{roi_height} pixel @ ({roi_x}, {roi_y})")
        st.caption("⚠️ Assicurati che la ROI sia dentro i limiti del video (il sistema correggerà automaticamente se necessario)")
    
    st.divider()
    
    # Configurazione tracking
    st.subheader("🎯 Parametri Tracking")
    distanza_euclidea = st.slider(
        "Distanza euclidea massima (pixel)",
        min_value=25,
        max_value=500,
        value=350,
        step=10,
        help="Distanza max per considerare lo stesso oggetto tra frame consecutivi"
    )
    
    st.divider()
    
    # Configurazione MOG2
    st.subheader("🔍 Background Subtraction (MOG2)")
    st.markdown("""
    **History**: Numero frame per modello background  
    - Alto (300-500): adattamento lento, scene statiche  
    - Basso (50-150): adattamento veloce, scene dinamiche
    
    **Var Threshold**: Soglia varianza per classificazione  
    - Alto (60-100): meno sensibile, meno movimento  
    - Basso (10-40): più sensibile, più movimento
    """)
    mog2_history = st.slider("History", 50, 500, 200, 10)
    mog2_threshold = st.slider("Var Threshold", 10, 100, 40, 5)
    
    st.divider()
    
    # Limite frame per performance
    st.subheader("⚡ Performance")
    max_frames = st.slider("Max frame da elaborare", 50, 1000, 150, 50, 
                          help="Limita i frame per velocizzare l'elaborazione")
    
    st.divider()
    
    # Pulsante rielaborazione
    process_button = st.button("🔄 Rielabora", use_container_width=True)


def process_video(video_path, roi_mode, roi_manual_params, distanza_euclidea, mog2_history, mog2_threshold, max_frames=150):
    """Elabora il video applicando tracking su originale e filtrato"""
    
    # Carica video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None, None, None
    
    # Leggi primo frame per configurazione
    ret, first_frame = cap.read()
    if not ret:
        return None, None, None, None, None, None
    
    h, w = first_frame.shape[:2]
    
    # Configura ROI
    if roi_mode == "Automatica (centrale 60%)":
        roi_w = int(w * 0.6)
        roi_h = int(h * 0.6)
        roi_x = (w - roi_w) // 2
        roi_y = (h - roi_h) // 2
    elif roi_mode == "Manuale (personalizzata)":
        if roi_manual_params is not None:
            roi_x, roi_y, roi_w, roi_h = roi_manual_params
            # Valida e aggiusta i parametri per assicurare che siano dentro i limiti del video
            roi_x = max(0, min(roi_x, w - 50))
            roi_y = max(0, min(roi_y, h - 50))
            roi_w = max(50, min(roi_w, w - roi_x))
            roi_h = max(50, min(roi_h, h - roi_y))
        else:
            # Fallback ad automatica se parametri non forniti
            roi_w = int(w * 0.6)
            roi_h = int(h * 0.6)
            roi_x = (w - roi_w) // 2
            roi_y = (h - roi_h) // 2
    else:  # Fotogramma completo
        roi_x, roi_y, roi_w, roi_h = 0, 0, w, h
    
    # Inizializza sistemi tracking
    tracker_original = Tracciamento(distanza_max=distanza_euclidea, use_feature_matching=True)
    tracker_filtered = Tracciamento(distanza_max=distanza_euclidea, use_feature_matching=True)
    
    # Inizializza background subtractor
    bg_subtractor_original = cv2.createBackgroundSubtractorMOG2(history=mog2_history, varThreshold=mog2_threshold)
    bg_subtractor_filtered = cv2.createBackgroundSubtractorMOG2(history=mog2_history, varThreshold=mog2_threshold)
    
    # Inizializza stabilizzatore
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, ref_frame = cap.read()
    stabilizer = VideoStabilizer(ref_frame, smoothing_radius=5)
    
    # Liste per salvare frame processati
    frames_original_tracked = []
    frames_filtered_tracked = []
    frames_original_full = []
    frames_filtered_full = []
    masks_original = []
    masks_filtered = []
    
    # Liste per video filtrati (per metriche qualità)
    frames_jpeg = []
    frames_blurred = []
    frames_sharpened = []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Progress bar
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Elaborazione frame {frame_count}/{total_frames}")
        
        # Estrai ROI
        roi_clean = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
        
        # === PROCESSING VIDEO ORIGINALE ===
        # Background subtraction
        mask_original = bg_subtractor_original.apply(roi_clean.copy())
        _, mask_original = cv2.threshold(mask_original, 254, 255, cv2.THRESH_BINARY)
        
        # Operazioni morfologiche
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_original = cv2.morphologyEx(mask_original, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_original = cv2.morphologyEx(mask_original, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Trova contorni
        contours_original, _ = cv2.findContours(mask_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections_original = []
        for contour in contours_original:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                detections_original.append([x, y, w_box, h_box])
        
        # Filtra box sovrapposti
        detections_original = filtra_box_sovrapposti(detections_original, iou_threshold=0.3)
        
        # Tracking
        roi_display_original = roi_clean.copy()
        tracked_original = tracker_original.update(detections_original, frame_roi=roi_clean)
        
        # Disegna tracking
        for track in tracked_original:
            obj_id, x, y, w_box, h_box = track
            cv2.rectangle(roi_display_original, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(roi_display_original, str(obj_id), (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # === PROCESSING VIDEO FILTRATO (Stabilizzato + Moving Average) ===
        # Stabilizzazione
        frame_stabilized = stabilizer.stabilize(frame.copy())
        
        # Moving average spaziale
        frame_filtered = moving_average_filter(frame_stabilized, kernel_size=5)
        
        # Estrai ROI filtrata
        roi_filtered_clean = frame_filtered[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
        
        # Background subtraction
        mask_filtered = bg_subtractor_filtered.apply(roi_filtered_clean.copy())
        _, mask_filtered = cv2.threshold(mask_filtered, 254, 255, cv2.THRESH_BINARY)
        
        # Operazioni morfologiche
        mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Trova contorni
        contours_filtered, _ = cv2.findContours(mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections_filtered = []
        for contour in contours_filtered:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                detections_filtered.append([x, y, w_box, h_box])
        
        # Filtra box sovrapposti
        detections_filtered = filtra_box_sovrapposti(detections_filtered, iou_threshold=0.3)
        
        # Tracking
        roi_display_filtered = roi_filtered_clean.copy()
        tracked_filtered = tracker_filtered.update(detections_filtered, frame_roi=roi_filtered_clean)
        
        # Disegna tracking
        for track in tracked_filtered:
            obj_id, x, y, w_box, h_box = track
            cv2.rectangle(roi_display_filtered, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(roi_display_filtered, str(obj_id), (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Salva frame per visualizzazione
        frames_original_tracked.append(cv2.cvtColor(roi_display_original, cv2.COLOR_BGR2RGB))
        frames_filtered_tracked.append(cv2.cvtColor(roi_display_filtered, cv2.COLOR_BGR2RGB))
        frames_original_full.append(frame.copy())
        frames_filtered_full.append(frame_filtered.copy())
        
        # Salva maschere MOG2 (converti in RGB per visualizzazione)
        masks_original.append(cv2.cvtColor(mask_original, cv2.COLOR_GRAY2RGB))
        masks_filtered.append(cv2.cvtColor(mask_filtered, cv2.COLOR_GRAY2RGB))
        
        # Applica filtri per metriche di qualità
        frame_jpeg = jpeg_compression(frame.copy(), quality=50)
        frame_blurred = blurring(frame.copy())
        frame_sharpened = unsharp_masking(frame.copy(), amount=1.5)
        
        frames_jpeg.append(frame_jpeg)
        frames_blurred.append(frame_blurred)
        frames_sharpened.append(frame_sharpened)
        
        # Limita numero frame per memoria
        if frame_count >= max_frames:
            break
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    # Calcola metriche filtri (JPEG, Blur, Sharpening)
    mse_jpeg, psnr_jpeg, ssim_jpeg = compare_video_sequences(frames_original_full, frames_jpeg)
    mse_blur, psnr_blur, ssim_blur = compare_video_sequences(frames_original_full, frames_blurred)
    mse_sharp, psnr_sharp, ssim_sharp = compare_video_sequences(frames_original_full, frames_sharpened)
    
    # Statistiche tracking
    stats = {
        'total_frames': frame_count,
        'original_objects': tracker_original.get_numero_oggetti_totali(),
        'filtered_objects': tracker_filtered.get_numero_oggetti_totali(),
        'mse_jpeg': mse_jpeg,
        'psnr_jpeg': psnr_jpeg,
        'ssim_jpeg': ssim_jpeg,
        'mse_blur': mse_blur,
        'psnr_blur': psnr_blur,
        'ssim_blur': ssim_blur,
        'mse_sharp': mse_sharp,
        'psnr_sharp': psnr_sharp,
        'ssim_sharp': ssim_sharp
    }
    
    return frames_original_tracked, frames_filtered_tracked, masks_original, masks_filtered, stats, (roi_x, roi_y, roi_w, roi_h)


# Main content

# Inizializza session state
if 'frames_original' not in st.session_state:
    st.session_state.frames_original = None
if 'frames_filtered' not in st.session_state:
    st.session_state.frames_filtered = None
if 'masks_original' not in st.session_state:
    st.session_state.masks_original = None
if 'masks_filtered' not in st.session_state:
    st.session_state.masks_filtered = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'roi_info' not in st.session_state:
    st.session_state.roi_info = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'show_masks' not in st.session_state:
    st.session_state.show_masks = False

# Determina il video da usare
video_path = None
video_name = None

if use_default:
    # Usa video di default
    default_video = Path("video/traffico1cut.mp4")
    if default_video.exists():
        video_path = str(default_video)
        video_name = "traffico1cut.mp4 (video di default)"
    else:
        st.warning("⚠️ Video di default non trovato in video/traffico1cut.mp4")
elif uploaded_file is not None:
    # Salva video caricato temporaneamente
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
        video_name = uploaded_file.name

# Elabora video se disponibile e non già processato (o se richiesto rielaborazione)
if video_path is not None:
    # Mostra anteprima ROI se in modalità manuale
    if roi_mode == "Manuale (personalizzata)" and roi_manual_params is not None:
        with st.expander("👁️ Anteprima ROI Selezionata", expanded=False):
            # Carica primo frame per anteprima
            cap_preview = cv2.VideoCapture(video_path)
            ret_preview, frame_preview = cap_preview.read()
            cap_preview.release()
            
            if ret_preview:
                # Disegna rettangolo ROI sul frame
                frame_with_roi = frame_preview.copy()
                roi_x_p, roi_y_p, roi_w_p, roi_h_p = roi_manual_params
                
                # Assicurati che la ROI sia dentro i limiti
                h_p, w_p = frame_preview.shape[:2]
                roi_x_p = max(0, min(roi_x_p, w_p - 50))
                roi_y_p = max(0, min(roi_y_p, h_p - 50))
                roi_w_p = max(50, min(roi_w_p, w_p - roi_x_p))
                roi_h_p = max(50, min(roi_h_p, h_p - roi_y_p))
                
                # Disegna rettangolo verde per la ROI
                cv2.rectangle(frame_with_roi, 
                            (roi_x_p, roi_y_p), 
                            (roi_x_p + roi_w_p, roi_y_p + roi_h_p), 
                            (0, 255, 0), 3)
                
                # Aggiungi testo
                cv2.putText(frame_with_roi, "ROI", 
                          (roi_x_p + 10, roi_y_p + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Converti e mostra
                frame_with_roi_rgb = cv2.cvtColor(frame_with_roi, cv2.COLOR_BGR2RGB)
                st.image(frame_with_roi_rgb, caption=f"Primo frame con ROI evidenziata: {roi_w_p}x{roi_h_p} @ ({roi_x_p}, {roi_y_p})", 
                        use_container_width=True)
                st.info(f"📐 Dimensioni video: {w_p}x{h_p} | ROI validata: {roi_w_p}x{roi_h_p} @ ({roi_x_p}, {roi_y_p})")
    
    if not st.session_state.video_processed or process_button:
        st.info(f"📹 Elaborazione video: {video_name}")
        with st.spinner("🔄 Elaborazione in corso..."):
            frames_original, frames_filtered, masks_original, masks_filtered, stats, roi_info = process_video(
                video_path, 
                roi_mode,
                roi_manual_params,
                distanza_euclidea,
                mog2_history,
                mog2_threshold,
                max_frames
            )
        
        if frames_original is not None:
            # Salva in session state
            st.session_state.frames_original = frames_original
            st.session_state.frames_filtered = frames_filtered
            st.session_state.masks_original = masks_original
            st.session_state.masks_filtered = masks_filtered
            st.session_state.stats = stats
            st.session_state.roi_info = roi_info
            st.session_state.video_processed = True
            st.success("✅ Elaborazione completata!")
            st.rerun()
        else:
            st.error("❌ Errore nell'elaborazione del video")
    
    # Mostra risultati se disponibili
    if st.session_state.video_processed and st.session_state.frames_original is not None:
        frames_original = st.session_state.frames_original
        frames_filtered = st.session_state.frames_filtered
        masks_original = st.session_state.masks_original
        masks_filtered = st.session_state.masks_filtered
        stats = st.session_state.stats
        roi_info = st.session_state.roi_info
        
        # Verifica che stats abbia le chiavi corrette (nuovo formato)
        if 'mse_jpeg' not in stats:
            st.warning("⚠️ Dati elaborati con versione precedente. Premi '🔄 Rielabora' per aggiornare le metriche.")
            st.stop()
        
        # Statistiche
        st.header("📊 Statistiche & Metriche")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Frame Elaborati", stats['total_frames'])
        
        with col2:
            diff = stats['filtered_objects'] - stats['original_objects']
            st.metric(
                "Oggetti Originale", 
                stats['original_objects'],
                help="Numero totale di oggetti tracciati nel video originale"
            )
        
        with col3:
            st.metric(
                "Oggetti Filtrato", 
                stats['filtered_objects'],
                delta=f"{diff:+d}",
                help="Numero totale di oggetti tracciati nel video filtrato"
            )
        
        with col4:
            improvement = ((stats['original_objects'] - stats['filtered_objects']) / stats['original_objects'] * 100) if stats['original_objects'] > 0 else 0
            st.metric(
                "Riduzione Falsi Positivi", 
                f"{improvement:.1f}%",
                help="Percentuale di riduzione oggetti (meno oggetti = meno falsi positivi)"
            )
        
        # Metriche qualità - Confronto Filtri
        st.subheader("📈 Metriche di Qualità Video (Originale vs Filtri)")
        st.markdown("**Confronto:** Video Originale (ground truth) vs Video Processati")
        st.markdown("""**Legenda metriche:**
        - **MSE**: Più BASSO = migliore (errore minore)
        - **PSNR**: Più ALTO = migliore (>30dB ottimo, <20dB scarso)
        - **SSIM**: Più ALTO = migliore (1=identico, 0=diverso)
        """)
        
        # Crea tabella comparativa
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📸 JPEG Compression**")
            st.caption("Quality=50")
            st.metric("MSE", f"{stats['mse_jpeg']:.4f}")
            psnr_color_jpeg = "🟢" if stats['psnr_jpeg'] > 30 else "🟡" if stats['psnr_jpeg'] > 20 else "🔴"
            st.metric(f"PSNR {psnr_color_jpeg}", f"{stats['psnr_jpeg']:.2f} dB")
            ssim_color_jpeg = "🟢" if stats['ssim_jpeg'] > 0.9 else "🟡" if stats['ssim_jpeg'] > 0.7 else "🔴"
            st.metric(f"SSIM {ssim_color_jpeg}", f"{stats['ssim_jpeg']:.4f}")
        
        with col2:
            st.markdown("**🌫️ Gaussian Blur**")
            st.caption("Kernel 5x5")
            st.metric("MSE", f"{stats['mse_blur']:.4f}")
            psnr_color_blur = "🟢" if stats['psnr_blur'] > 30 else "🟡" if stats['psnr_blur'] > 20 else "🔴"
            st.metric(f"PSNR {psnr_color_blur}", f"{stats['psnr_blur']:.2f} dB")
            ssim_color_blur = "🟢" if stats['ssim_blur'] > 0.9 else "🟡" if stats['ssim_blur'] > 0.7 else "🔴"
            st.metric(f"SSIM {ssim_color_blur}", f"{stats['ssim_blur']:.4f}")
        
        with col3:
            st.markdown("**✨ Sharpening**")
            st.caption("Unsharp Masking")
            st.metric("MSE", f"{stats['mse_sharp']:.4f}")
            psnr_color_sharp = "🟢" if stats['psnr_sharp'] > 30 else "🟡" if stats['psnr_sharp'] > 20 else "🔴"
            st.metric(f"PSNR {psnr_color_sharp}", f"{stats['psnr_sharp']:.2f} dB")
            ssim_color_sharp = "🟢" if stats['ssim_sharp'] > 0.9 else "🟡" if stats['ssim_sharp'] > 0.7 else "🔴"
            st.metric(f"SSIM {ssim_color_sharp}", f"{stats['ssim_sharp']:.4f}")
        
        # Riepilogo comparativo
        st.markdown("---")
        st.subheader("🏆 Riepilogo Comparativo (Ordinato per SSIM)")
        
        risultati = [
            ("JPEG Compression (Q=50)", stats['mse_jpeg'], stats['psnr_jpeg'], stats['ssim_jpeg']),
            ("Gaussian Blurring", stats['mse_blur'], stats['psnr_blur'], stats['ssim_blur']),
            ("Sharpening (Unsharp)", stats['mse_sharp'], stats['psnr_sharp'], stats['ssim_sharp'])
        ]
        
        # Ordina per SSIM decrescente
        risultati_ordinati = sorted(risultati, key=lambda x: x[3], reverse=True)
        
        for i, (nome, mse, psnr, ssim) in enumerate(risultati_ordinati, 1):
            st.markdown(f"**{i}. {nome}** - MSE: {mse:.4f} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
        
        st.divider()
        
        # Confronto video
        st.header("🎬 Confronto Video con Tracking")
        
        # Toggle per mostrare maschere MOG2
        col_toggle1, col_toggle2 = st.columns([2, 3])
        with col_toggle1:
            show_masks = st.checkbox("🔍 Mostra Maschere MOG2", value=False, 
                                    help="Visualizza le maschere di background subtraction")
        with col_toggle2:
            if show_masks:
                st.info("Le maschere mostrano il foreground (bianco) rilevato da MOG2")
        
        # Controlli di riproduzione
        col_play1, col_play2, col_play3, col_play4 = st.columns([1, 1, 1, 2])
        
        with col_play1:
            if st.button("▶️ Play" if not st.session_state.is_playing else "⏸️ Pause", use_container_width=True):
                st.session_state.is_playing = not st.session_state.is_playing
                if st.session_state.is_playing:
                    st.rerun()
        
        with col_play2:
            if st.button("⏹️ Stop", use_container_width=True):
                st.session_state.is_playing = False
                st.session_state.current_frame = 0
                st.rerun()
        
        with col_play3:
            if st.button("⏮️ Restart", use_container_width=True):
                st.session_state.current_frame = 0
                st.rerun()
        
        with col_play4:
            fps = st.slider("FPS (velocità riproduzione)", 1, 60, 30, 1, 
                          help="Frame per secondo durante la riproduzione automatica")
        
        # Logica di riproduzione automatica
        if st.session_state.is_playing:
            import time
            
            # Crea placeholder per frame display
            main_container = st.container()
            info_placeholder = st.empty()
            
            while st.session_state.is_playing and st.session_state.current_frame < len(frames_original):
                # Mostra info stato
                with info_placeholder:
                    st.info(f"▶️ Riproduzione in corso... Frame {st.session_state.current_frame + 1}/{len(frames_original)} | FPS: {fps}")
                
                # Mostra frame corrente in container
                with main_container:
                    if show_masks:
                        # Modalità con maschere: 2x2 grid
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("🔵 Video Originale + Tracking")
                            st.image(frames_original[st.session_state.current_frame], use_container_width=True, clamp=True)
                            st.caption("Tracking con MOG2 + Feature Matching")
                            st.subheader("⬜ Maschera MOG2 Originale")
                            st.image(masks_original[st.session_state.current_frame], use_container_width=True, clamp=True)
                            st.caption(f"Background Subtraction (History={mog2_history}, Threshold={mog2_threshold})")
                        
                        with col2:
                            st.subheader("🟢 Video Filtrato + Tracking")
                            st.image(frames_filtered[st.session_state.current_frame], use_container_width=True, clamp=True)
                            st.caption("Stabilizzazione + MA + Tracking")
                            st.subheader("⬜ Maschera MOG2 Filtrata")
                            st.image(masks_filtered[st.session_state.current_frame], use_container_width=True, clamp=True)
                            st.caption(f"Background Subtraction su frame stabilizzato")
                    else:
                        # Modalità normale: solo tracking
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("🔵 Video Originale + Tracking + Features")
                            st.image(frames_original[st.session_state.current_frame], use_container_width=True, clamp=True)
                            st.caption("MOG2 Background Subtraction + Feature Matching (ORB) + Coordinate Smoothing")
                        
                        with col2:
                            st.subheader("🟢 Video Filtrato + Tracking + ORB + Features")
                            st.image(frames_filtered[st.session_state.current_frame], use_container_width=True, clamp=True)
                            st.caption("Stabilizzazione Video (ORB) + Moving Average Spaziale + MOG2 + Feature Matching + Smoothing")
                
                # Avanza frame
                st.session_state.current_frame += 1
                
                # Se arrivato alla fine, ferma o ripeti
                if st.session_state.current_frame >= len(frames_original):
                    st.session_state.current_frame = 0
                    st.session_state.is_playing = False
                    info_placeholder.empty()
                    st.success("✅ Riproduzione completata!")
                    time.sleep(1)
                    st.rerun()
                    break
                
                # Attendi in base agli FPS (con minimo per evitare sovraccarico)
                delay = max(0.033, 1.0 / fps)  # Minimo 30ms
                time.sleep(delay)
                st.rerun()
        
        else:
            # Modalità manuale con slider
            # Slider per navigare frame
            frame_idx = st.slider(
                "Seleziona frame manualmente",
                0,
                len(frames_original) - 1,
                st.session_state.current_frame,
                help="Scorri per vedere i diversi frame processati",
                key="manual_frame_slider"
            )
            
            # Aggiorna current_frame se modificato manualmente
            if frame_idx != st.session_state.current_frame:
                st.session_state.current_frame = frame_idx
            
            # Mostra frame affiancati
            if show_masks:
                # Modalità con maschere: 2x2 grid
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔵 Video Originale + Tracking")
                    st.image(frames_original[frame_idx], use_container_width=True)
                    st.caption("Tracking con MOG2 + Feature Matching")
                    st.subheader("⬜ Maschera MOG2 Originale")
                    st.image(masks_original[frame_idx], use_container_width=True)
                    st.caption(f"Background Subtraction (History={mog2_history}, Threshold={mog2_threshold})")
                
                with col2:
                    st.subheader("🟢 Video Filtrato + Tracking")
                    st.image(frames_filtered[frame_idx], use_container_width=True)
                    st.caption("Stabilizzazione + MA + Tracking")
                    st.subheader("⬜ Maschera MOG2 Filtrata")
                    st.image(masks_filtered[frame_idx], use_container_width=True)
                    st.caption(f"Background Subtraction su frame stabilizzato")
            else:
                # Modalità normale: solo tracking
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔵 Video Originale + Tracking + Features")
                    st.image(frames_original[frame_idx], use_container_width=True)
                    st.caption("MOG2 Background Subtraction + Feature Matching (ORB) + Coordinate Smoothing")
                
                with col2:
                    st.subheader("🟢 Video Filtrato + Tracking + ORB + Features")
                    st.image(frames_filtered[frame_idx], use_container_width=True)
                    st.caption("Stabilizzazione Video (ORB) + Moving Average Spaziale + MOG2 + Feature Matching + Smoothing")
        
        st.divider()
        
        # Guida controlli riproduzione
        with st.expander("🎮 Come usare i Controlli di Riproduzione"):
            st.markdown("""
            **Controlli Disponibili:**
            
            - **▶️ Play / ⏸️ Pause**: Avvia o mette in pausa la riproduzione automatica dei frame
            - **⏹️ Stop**: Ferma la riproduzione e torna al primo frame
            - **⏮️ Restart**: Ricomincia dal primo frame
            - **FPS Slider**: Regola la velocità di riproduzione (1-60 FPS)
            
            **Modalità di Visualizzazione:**
            
            1. **Riproduzione Automatica**: Premi "▶️ Play" per vedere i frame scorrere automaticamente
               come un video. Regola la velocità con lo slider FPS. Premi "⏸️ Pause" per fermare.
            
            2. **Navigazione Manuale**: Quando in pausa, usa lo slider per selezionare manualmente
               qualsiasi frame e analizzarlo in dettaglio.
            
            **Nota**: Durante la riproduzione, la pagina si aggiornerà automaticamente per mostrare
            il frame successivo. Alla fine del video, la riproduzione si fermerà automaticamente.
            """)
        
        # Guida selezione ROI
        with st.expander("📍 Guida Selezione ROI (Region of Interest)"):
            st.markdown("""
            **Modalità ROI Disponibili:**
            
            1. **Automatica (centrale 60%)**: Seleziona automaticamente l'area centrale del video (60% delle dimensioni)
               - ✅ Buona per scene con oggetti concentrati al centro
               - ✅ Riduce calcoli processando solo l'area centrale
            
            2. **Fotogramma completo**: Usa l'intero video senza ritagliare
               - ✅ Traccia oggetti in tutta la scena
               - ⚠️ Maggiore carico computazionale
            
            3. **Manuale (personalizzata)**: Definisci una ROI specifica con parametri personalizzati
               - ✅ Massima flessibilità per scene specifiche
               - 💡 Usa l'anteprima per verificare l'area selezionata
               - 📏 Il sistema corregge automaticamente se i valori superano i limiti del video
               
            **Consigli per ROI Manuale:**
            - Posiziona la ROI dove si concentra il movimento (es. strada principale nel traffico)
            - Evita aree statiche (es. cielo, edifici fissi)
            - Una ROI più piccola = elaborazione più veloce
            - Usa l'anteprima per verificare prima di elaborare
            - Premi "🔄 Rielabora" dopo aver modificato i parametri ROI
            """)
        
        # Info tracking migliorato
        st.info("""
        💡 **Come interpretare i risultati:**
        
        - **Oggetti tracciati**: Un numero minore nel video filtrato indica una possibile riduzione dei falsi positivi
        - **MSE (Mean Squared Error)**: Più basso = migliore (errore minore)
        - **PSNR (Peak Signal-to-Noise Ratio)**: Più alto = migliore | >30dB ottimo, <20dB scarso
        - **SSIM (Structural Similarity Index)**: Più alto = migliore | 1=identico, 0=diverso

        
        **🔍 Maschere MOG2:**
        - Attiva "Mostra Maschere MOG2" per vedere come l'algoritmo distingue foreground (oggetti in movimento) dal background
        - **Bianco**: Pixel classificati come foreground (movimento/oggetti)
        - **Nero**: Pixel classificati come background (scena statica)
        - Una maschera più "pulita" nel video filtrato indica migliore separazione foreground/background

        """)
        
        # Cleanup file temporaneo se caricato
        if not use_default and uploaded_file is not None:
            try:
                os.unlink(video_path)
            except:
                pass

else:
    # Nessun video disponibile
    st.warning("""
    ⚠️ **Nessun video disponibile**
    
    Per procedere:
    1. Assicurati che il file `video/traffico1cut.mp4` esista nella cartella del progetto
    2. Oppure disabilita "Usa video di default" e carica un video personalizzato
    """)


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    🎓 Progetto Multimedia - Video Tracking & Post-Processing<br>
    Università degli Studi - A.A. 2025/2026
</div>
""", unsafe_allow_html=True)
