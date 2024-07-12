import os
import random

# Percorso della cartella contenente le immagini
save_dir = "C:/Users/danie/OneDrive - uniroma1.it/Desktop/dataset/Nothing"

def elimina_immagini_randomicamente(directory, target_count=300):
    # Lista di tutte le immagini nella cartella
    immagini = [f for f in os.listdir(directory) if f.endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]

    # Numero attuale di immagini
    current_count = len(immagini)

    # Controlla se il numero attuale di immagini è già inferiore o uguale a target_count
    if current_count <= target_count:
        print(f"La cartella contiene già {current_count} immagini, che è minore o uguale a {target_count}.")
        return

    # Numero di immagini da eliminare
    num_da_eliminare = current_count - target_count

    # Seleziona casualmente le immagini da eliminare
    immagini_da_eliminare = random.sample(immagini, num_da_eliminare)

    # Elimina le immagini selezionate
    for img in immagini_da_eliminare:
        img_path = os.path.join(directory, img)
        os.remove(img_path)
        print(f"Immagine eliminata: {img_path}")

    print(f"Numero finale di immagini nella cartella: {target_count}")

# Esegue la funzione
elimina_immagini_randomicamente(save_dir)
