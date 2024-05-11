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
acquisisci_e_salva_video(nome_file, durata_secondi)
