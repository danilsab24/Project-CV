import os

main_folder = r"C:/Users/danie/OneDrive - uniroma1.it/Desktop/daniele/dataset"


subfolders = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Rename subforlder
def rename_files_in_folder(folder_path, prefix):

    files = os.listdir(folder_path)
    
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    
    files.sort()

    # Rename files
    for i, filename in enumerate(files):
        
        file_extension = os.path.splitext(filename)[1]
        
        new_name = f"{prefix}_{i + 1}{file_extension}"
        
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        
        os.rename(old_file, new_file)
        print(f"Rinominato: {old_file} -> {new_file}")

for subfolder in subfolders:
    folder_path = os.path.join(main_folder, subfolder)
    
    rename_files_in_folder(folder_path, subfolder)

print("Rinomina completata")
