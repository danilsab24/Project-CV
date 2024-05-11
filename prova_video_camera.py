import cv2
import pygame
import pygame.camera
from pygame.locals import *

# Funzione per acquisire video dalla fotocamera e salvarlo come file
def acquisisci_e_salva_video(nome_file, durata_secondi):
    # Impostazioni per acquisire video dalla fotocamera
    video_capture = cv2.VideoCapture(0)  # Utilizza la fotocamera predefinita

    # Impostazioni per il video in uscita
    codec = cv2.VideoWriter_fourcc(*'XVID')  # Codec per il formato del video
    fps = 30.0  # Frame per secondo
    frame_size = (640, 480)  # Dimensioni del frame
    out = cv2.VideoWriter(nome_file, codec, fps, frame_size)

    # Timer per la durata del video
    timer = cv2.getTickCount()
    durata_frame = 1 / fps

    while True:
        # Acquisizione frame dalla fotocamera
        ret, frame = video_capture.read()

        # Visualizza il frame acquisito
        cv2.imshow('Video', frame)

        # Salva il frame nel video
        out.write(frame)

        # Controlla se è passata la durata desiderata
        if cv2.waitKey(int(durata_frame * 1000)) & 0xFF == ord('q') or \
                ((cv2.getTickCount() - timer) / cv2.getTickFrequency()) >= durata_secondi:
            break

    # Rilascia le risorse
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

# Nome del file video e durata in secondi
nome_file = 'video_acquisito.avi'
durata_secondi = 10

# Chiama la funzione per acquisire e salvare il video
#in questo caso il video non viene modificato e viene salvato
#acquisisci_e_salva_video(nome_file, durata_secondi)

#in questo caso il video viene modifica ma non salvato
#preme 'q' per terminare il video
# Funzione per convertire un frame in bianco e nero
def converti_bn(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Funzione per acquisire video dalla fotocamera e visualizzarlo in tempo reale
def acquisisci_e_visualizza():
    # Impostazioni per acquisire video dalla fotocamera
    video_capture = cv2.VideoCapture(0)  # Utilizza la fotocamera predefinita

    while True:
        # Acquisizione frame dalla fotocamera
        ret, frame = video_capture.read()

        # Converti il frame in bianco e nero
        frame_bn = converti_bn(frame)

        # Visualizza il frame in bianco e nero
        cv2.imshow('Video in Bianco e Nero', frame_bn)

        # Controlla se l'utente preme 'q' per interrompere
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia le risorse
    video_capture.release()
    cv2.destroyAllWindows()

# Chiama la funzione per acquisire e visualizzare il video in tempo reale
acquisisci_e_visualizza()


