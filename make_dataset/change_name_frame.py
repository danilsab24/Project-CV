import os

# Path alla cartella principale
main_folder = r"C:/Users/danie/OneDrive - uniroma1.it/Desktop/daniele/dataset"

# Elenco delle sottocartelle
subfolders = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Funzione per rinominare i file in una cartella
def rename_files_in_folder(folder_path, prefix):
    # Elenca tutti i file nella cartella
    files = os.listdir(folder_path)
    
    # Filtro per mantenere solo i file (escludi le directory)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    
    # Ordina i file per nome
    files.sort()

    # Rinomina i file
    for i, filename in enumerate(files):
        # Estrai l'estensione del file
        file_extension = os.path.splitext(filename)[1]
        
        # Crea il nuovo nome del file
        new_name = f"{prefix}_{i + 1}{file_extension}"
        
        # Costruisci i percorsi completo del file
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        
        # Rinomina il file
        os.rename(old_file, new_file)
        print(f"Rinominato: {old_file} -> {new_file}")

# Cicla attraverso tutte le sottocartelle
for subfolder in subfolders:
    # Costruisci il percorso completo della sottocartella
    folder_path = os.path.join(main_folder, subfolder)
    
    # Rinomina i file nella sottocartella
    rename_files_in_folder(folder_path, subfolder)

print("Rinomina completata")
