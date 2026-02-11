import math
class Tracciamento:
    def __init__(self):
        #Contatore id
        self.identificativo_corrente = 0
        #Registro coordinate entità tracciate
        self.coordinate_centrali = {}

    def update(self, rettangoli_rilevati):
        #Lista risultati con box e identificativi
        risultati_tracciamento = []

        #Elaborazione di ogni rettangolo rilevato
        for rettangolo in rettangoli_rilevati:
            coord_x, coord_y, larghezza, altezza = rettangolo
            centro_x = (coord_x + coord_x + larghezza) // 2
            centro_y = (coord_y + coord_y + altezza) // 2

            entita_gia_tracciata = False
            for id_entita, punto_centrale in self.coordinate_centrali.items():
                distanza_euclidea = math.hypot(centro_x - punto_centrale[0], centro_y - punto_centrale[1])

                if distanza_euclidea < 25:
                    self.coordinate_centrali[id_entita] = (centro_x, centro_y)
                    risultati_tracciamento.append([id_entita, coord_x, coord_y, larghezza, altezza])
                    entita_gia_tracciata = True
                    break

            # Assegnazione id
            if not entita_gia_tracciata:
                self.coordinate_centrali[self.identificativo_corrente] = (centro_x, centro_y)
                risultati_tracciamento.append([self.identificativo_corrente, coord_x, coord_y, larghezza, altezza])
                self.identificativo_corrente += 1

        self._rimuovi_entita_obsolete(risultati_tracciamento)

        return risultati_tracciamento


