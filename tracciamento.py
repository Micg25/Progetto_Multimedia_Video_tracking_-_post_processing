import math
import cv2
import numpy as np

class CoordinateSmoothing:
    
    #Applica smoothing esponenziale alle coordinate dei bounding box
    #per ridurre il jitter del tracciamento.
    
    def __init__(self, alpha=0.3):
  
        self.alpha = alpha
        self.smoothed_coords = {}  # {id: (x, y, w, h)}
    
    def smooth(self, id_obj, x, y, w, h):
        
        #Applica smoothing esponenziale alle coordinate.
      
        if id_obj not in self.smoothed_coords:
            # Primo rilevamento, salva coordinate senza smoothing
            self.smoothed_coords[id_obj] = (float(x), float(y), float(w), float(h))
            return x, y, w, h
        else:
            # Applica media mobile esponenziale
            prev_x, prev_y, prev_w, prev_h = self.smoothed_coords[id_obj]
            
            smooth_x = self.alpha * x + (1 - self.alpha) * prev_x
            smooth_y = self.alpha * y + (1 - self.alpha) * prev_y
            smooth_w = self.alpha * w + (1 - self.alpha) * prev_w
            smooth_h = self.alpha * h + (1 - self.alpha) * prev_h
            
            # Salva coordinate smoothed
            self.smoothed_coords[id_obj] = (smooth_x, smooth_y, smooth_w, smooth_h)
            
            return int(smooth_x), int(smooth_y), int(smooth_w), int(smooth_h)
    
    def remove(self, id_obj):
        """Rimuove le coordinate smoothed per un ID obsoleto."""
        if id_obj in self.smoothed_coords:
            del self.smoothed_coords[id_obj]

def filtra_box_sovrapposti(box_list, iou_threshold=0.5):
    #Filtra i bounding box rimuovendo quelli sovrapposti usando Non-Maximum Suppression (NMS).
    #Unisce box con IoU > threshold mantenendo quello più grande.
    

    if len(box_list) <= 1:
        return box_list
    
    # Converti in array numpy
    boxes = np.array(box_list)
    
    # Estrai coordinate
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    # Calcola aree
    areas = boxes[:, 2] * boxes[:, 3]
    
    # Ordina per area decrescente (box più grandi hanno priorità)
    indices = np.argsort(areas)[::-1]
    
    keep = []
    
    while len(indices) > 0:
        # Prendi il box con area maggiore
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calcola IoU con i rimanenti box
        remaining = indices[1:]
        
        # Intersezione
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        
        # Union
        union = areas[current] + areas[remaining] - intersection
        
        # IoU
        iou = intersection / union
        
        # Mantieni solo box con IoU < threshold (non sovrapposti)
        indices = remaining[iou < iou_threshold]
    
    # Ritorna box filtrati
    return [box_list[i] for i in keep]

class Tracciamento:
    def __init__(self, distanza_max=350, use_feature_matching=False, use_smoothing=True, smoothing_alpha=0.3):
        #Contatore id
        self.identificativo_corrente = 0
        #Registro coordinate entità tracciate
        self.coordinate_centrali = {}
        #Distanza massima per considerare stesso oggetto
        self.distanza_max = distanza_max
        
        # Coordinate smoothing per ridurre jitter
        self.use_smoothing = use_smoothing
        if self.use_smoothing:
            self.coordinate_smoother = CoordinateSmoothing(alpha=smoothing_alpha)
        
        # Feature matching per tracciamento più robusto
        self.use_feature_matching = use_feature_matching
        if self.use_feature_matching:
            self.orb_detector = cv2.ORB_create(nfeatures=100)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self.oggetti_features = {}  # {id: {'descriptors': desc, 'keypoints': kp}}

    def update(self, rettangoli_rilevati, frame_roi=None):

        #Aggiorna il tracciamento con i nuovi rilevamenti.
     
        #Lista risultati con box e identificativi
        risultati_tracciamento = []

        #Elaborazione di ogni rettangolo rilevato
        for rettangolo in rettangoli_rilevati:
            coord_x, coord_y, larghezza, altezza = rettangolo
            centro_x = (coord_x + coord_x + larghezza) // 2
            centro_y = (coord_y + coord_y + altezza) // 2

            entita_gia_tracciata = False
            miglior_id = None
            miglior_score = float('inf')
            
            # Se feature matching è abilitato e abbiamo il frame
            if self.use_feature_matching and frame_roi is not None:
                # Estrai features dal box corrente
                box_roi = frame_roi[coord_y:coord_y+altezza, coord_x:coord_x+larghezza]
                if box_roi.size > 0:
                    keypoints_new, descriptors_new = self.orb_detector.detectAndCompute(box_roi, None)
                    
                    if descriptors_new is not None and len(descriptors_new) > 0:
                        # Cerca match con oggetti già tracciati
                        for id_entita, punto_centrale in self.coordinate_centrali.items():
                            # Prima verifica distanza euclidea (filtro iniziale)
                            distanza_euclidea = math.hypot(centro_x - punto_centrale[0], centro_y - punto_centrale[1])
                            
                            if distanza_euclidea < self.distanza_max:
                                # Se abbiamo features per questo ID, calcola similarity
                                if id_entita in self.oggetti_features:
                                    descriptors_old = self.oggetti_features[id_entita]['descriptors']
                                    
                                    if descriptors_old is not None and len(descriptors_old) > 0:
                                        # Match features
                                        try:
                                            matches = self.bf_matcher.match(descriptors_new, descriptors_old)
                                            
                                            if len(matches) > 0:
                                                # Calcola score combinato: distanza + feature similarity
                                                avg_feature_distance = sum([m.distance for m in matches]) / len(matches)
                                                # Score combinato: 70% feature matching, 30% distanza euclidea
                                                combined_score = 0.7 * avg_feature_distance + 0.3 * distanza_euclidea
                                                
                                                if combined_score < miglior_score:
                                                    miglior_score = combined_score
                                                    miglior_id = id_entita
                                                    entita_gia_tracciata = True
                                        except:
                                            # Fallback a distanza euclidea se matching fallisce
                                            if distanza_euclidea < miglior_score:
                                                miglior_score = distanza_euclidea
                                                miglior_id = id_entita
                                                entita_gia_tracciata = True
                                else:
                                    # Nessuna feature salvata, usa solo distanza
                                    if distanza_euclidea < miglior_score:
                                        miglior_score = distanza_euclidea
                                        miglior_id = id_entita
                                        entita_gia_tracciata = True
                        
                        # Se trovato match, aggiorna
                        if entita_gia_tracciata and miglior_id is not None:
                            self.coordinate_centrali[miglior_id] = (centro_x, centro_y)
                            # Aggiorna features
                            self.oggetti_features[miglior_id] = {
                                'descriptors': descriptors_new,
                                'keypoints': keypoints_new
                            }
                            # Applica smoothing coordinate se abilitato
                            if self.use_smoothing:
                                coord_x, coord_y, larghezza, altezza = self.coordinate_smoother.smooth(
                                    miglior_id, coord_x, coord_y, larghezza, altezza
                                )
                            risultati_tracciamento.append([miglior_id, coord_x, coord_y, larghezza, altezza])
                        else:
                            # Nuovo oggetto
                            self.coordinate_centrali[self.identificativo_corrente] = (centro_x, centro_y)
                            self.oggetti_features[self.identificativo_corrente] = {
                                'descriptors': descriptors_new,
                                'keypoints': keypoints_new
                            }
                            # Applica smoothing coordinate se abilitato (primo frame = no smoothing)
                            if self.use_smoothing:
                                coord_x, coord_y, larghezza, altezza = self.coordinate_smoother.smooth(
                                    self.identificativo_corrente, coord_x, coord_y, larghezza, altezza
                                )
                            risultati_tracciamento.append([self.identificativo_corrente, coord_x, coord_y, larghezza, altezza])
                            self.identificativo_corrente += 1
                    else:
                        # Nessuna feature rilevata, fallback a distanza euclidea
                        entita_gia_tracciata = self._fallback_distanza_euclidea(
                            centro_x, centro_y, coord_x, coord_y, larghezza, altezza, risultati_tracciamento
                        )
                else:
                    # Box vuoto, fallback
                    entita_gia_tracciata = self._fallback_distanza_euclidea(
                        centro_x, centro_y, coord_x, coord_y, larghezza, altezza, risultati_tracciamento
                    )
            else:
                # Feature matching disabilitato, usa solo distanza euclidea
                entita_gia_tracciata = self._fallback_distanza_euclidea(
                    centro_x, centro_y, coord_x, coord_y, larghezza, altezza, risultati_tracciamento
                )

        self._rimuovi_entita_obsolete(risultati_tracciamento)

        return risultati_tracciamento
    
    def _fallback_distanza_euclidea(self, centro_x, centro_y, coord_x, coord_y, larghezza, altezza, risultati_tracciamento):
       #Metodo di tracciamento fallback basato solo su distanza euclidea
        for id_entita, punto_centrale in self.coordinate_centrali.items():
            distanza_euclidea = math.hypot(centro_x - punto_centrale[0], centro_y - punto_centrale[1])

            if distanza_euclidea < self.distanza_max:
                self.coordinate_centrali[id_entita] = (centro_x, centro_y)
                # Applica smoothing coordinate se abilitato
                if self.use_smoothing:
                    coord_x, coord_y, larghezza, altezza = self.coordinate_smoother.smooth(
                        id_entita, coord_x, coord_y, larghezza, altezza
                    )
                risultati_tracciamento.append([id_entita, coord_x, coord_y, larghezza, altezza])
                return True
        
        # Nuovo oggetto
        self.coordinate_centrali[self.identificativo_corrente] = (centro_x, centro_y)
        # Applica smoothing coordinate se abilitato (primo frame = no smoothing)
        if self.use_smoothing:
            coord_x, coord_y, larghezza, altezza = self.coordinate_smoother.smooth(
                self.identificativo_corrente, coord_x, coord_y, larghezza, altezza
            )
        risultati_tracciamento.append([self.identificativo_corrente, coord_x, coord_y, larghezza, altezza])
        self.identificativo_corrente += 1
        return False

        self._rimuovi_entita_obsolete(risultati_tracciamento)

        return risultati_tracciamento

    def _rimuovi_entita_obsolete(self, risultati_tracciamento):
        # Estrazione identificativi delle entità correntemente rilevate
        id_attivi = {r[0] for r in risultati_tracciamento}
        # Rimozione delle entità non più presenti
        id_da_rimuovere = [id_ent for id_ent in self.coordinate_centrali.keys() if id_ent not in id_attivi]
        for id_ent in id_da_rimuovere:
            del self.coordinate_centrali[id_ent]
            # Rimuovi anche features se presente
            if self.use_feature_matching and id_ent in self.oggetti_features:
                del self.oggetti_features[id_ent]
            # Rimuovi anche coordinate smoothed se presente
            if self.use_smoothing:
                self.coordinate_smoother.remove(id_ent)

    def get_numero_oggetti_totali(self):
        return self.identificativo_corrente
