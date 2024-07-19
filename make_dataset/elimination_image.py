import os
import random

save_dir = "C:/Users/danie/OneDrive - uniroma1.it/Desktop/daniele/dataset/nothing"

def elimina_immagini_randomicamente(directory, target_count=300):
  
    immagini = [f for f in os.listdir(directory) if f.endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]

    current_count = len(immagini)

    if current_count <= target_count:
        print(f"La cartella contiene già {current_count} immagini, che è minore o uguale a {target_count}.")
        return

    num_da_eliminare = current_count - target_count

    immagini_da_eliminare = random.sample(immagini, num_da_eliminare)

    for img in immagini_da_eliminare:
        img_path = os.path.join(directory, img)
        os.remove(img_path)
        print(f"Immagine eliminata: {img_path}")

    print(f"Numero finale di immagini nella cartella: {target_count}")

elimina_immagini_randomicamente(save_dir)
