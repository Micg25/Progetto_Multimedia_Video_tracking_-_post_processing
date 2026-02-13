import math
import cv2
import numpy as np

def filtra_box_sovrapposti(box_list):
    
    #Filtra i bounding box rimuovendo quelli completamente contenuti in altri box.
    #Mantiene solo i box più grandi quando c'è sovrapposizione completa.

    if len(box_list) <= 1:
        return box_list
    
    # Ordina per area decrescente (box più grandi prima)
    box_list_sorted = sorted(box_list, key=lambda b: b[2] * b[3], reverse=True)
    
    box_filtrati = []
    
    for i, box1 in enumerate(box_list_sorted):
        x1, y1, w1, h1 = box1
        contenuto = False
        
        # Controlla se box1 è contenuto in qualche box già accettato
        for box2 in box_filtrati:
            x2, y2, w2, h2 = box2
            
            # Verifica se box1 è completamente contenuto in box2
            if (x1 >= x2 and y1 >= y2 and 
                x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2):
                contenuto = True
                break
        
        if not contenuto:
            box_filtrati.append(box1)
    
    return box_filtrati

class Tracciamento:
    def __init__(self, distanza_max=350, use_feature_matching=False):
        #Contatore id
        self.identificativo_corrente = 0
        #Registro coordinate entità tracciate
        self.coordinate_centrali = {}
        #Distanza massima per considerare stesso oggetto
        self.distanza_max = distanza_max
        
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
                            risultati_tracciamento.append([miglior_id, coord_x, coord_y, larghezza, altezza])
                        else:
                            # Nuovo oggetto
                            self.coordinate_centrali[self.identificativo_corrente] = (centro_x, centro_y)
                            self.oggetti_features[self.identificativo_corrente] = {
                                'descriptors': descriptors_new,
                                'keypoints': keypoints_new
                            }
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
                risultati_tracciamento.append([id_entita, coord_x, coord_y, larghezza, altezza])
                return True
        
        # Nuovo oggetto
        self.coordinate_centrali[self.identificativo_corrente] = (centro_x, centro_y)
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

    def get_numero_oggetti_totali(self):
        return self.identificativo_corrente
