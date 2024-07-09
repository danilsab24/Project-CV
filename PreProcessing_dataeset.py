import os
import shutil
from sklearn.model_selection import train_test_split

def copy_files(src, dst, files):
    for file in files:
        src_path = os.path.join(src, file)
        dst_path = os.path.join(dst, file)
        if os.path.isfile(src_path):  # Ensure it is a file
            if not os.path.exists(os.path.dirname(dst_path)):
                os.makedirs(os.path.dirname(dst_path))
            try:
                shutil.copy2(src_path, dst_path)
            except PermissionError as e:
                print(f"PermissionError: {e}. Skipping {src_path}")
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
        else:
            print(f"Skipping non-file: {src_path}")

def move_files(src, dst, files):
    for file in files:
        src_path = os.path.join(src, file)
        dst_path = os.path.join(dst, file)
        if os.path.isfile(src_path):  # Ensure it is a file
            if not os.path.exists(os.path.dirname(dst_path)):
                os.makedirs(os.path.dirname(dst_path))
            try:
                shutil.move(src_path, dst_path)
            except PermissionError as e:
                print(f"PermissionError: {e}. Skipping {src_path}")
            except Exception as e:
                print(f"Error moving {src_path}: {e}")
        else:
            print(f"Skipping non-file: {src_path}")

def organize_test_images(test_dir):
    for root, _, files in os.walk(test_dir):
        for file_name in files:
            if file_name.endswith('.jpeg'):
                label = file_name.split('_')[0]
                label_dir = os.path.join(test_dir, label)
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)
                shutil.move(os.path.join(root, file_name), os.path.join(label_dir, file_name))

def get_all_files(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.relpath(os.path.join(root, file), directory))
    return file_list

def main():
    dataset_path = "F:\\dataset_hand_gesture\\ASL_Alphabet_Dataset"
    train_path = os.path.join(dataset_path, "asl_alphabet_train")
    test_path = os.path.join(dataset_path, "asl_alphabet_test")
    copy_dest = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\copia_dataset"
    train_dest = os.path.join(copy_dest, "asl_alphabet_train")
    test_dest = os.path.join(copy_dest, "asl_alphabet_test")

    # Ensure the destination directories exist
    if not os.path.exists(copy_dest):
        os.makedirs(copy_dest)
    if not os.path.exists(train_dest):
        os.makedirs(train_dest)
    if not os.path.exists(test_dest):
        os.makedirs(test_dest)

    # Copy 100% of the test dataset
    print("Copying test dataset...")
    test_files = get_all_files(test_path)
    copy_files(test_path, test_dest, test_files)

    # Organize test images into their respective folders
    print("Organizing test images...")
    organize_test_images(test_dest)

    # Copy 40% of the training dataset
    print("Copying 40% of training dataset...")
    for subdir in os.listdir(train_path):
        subdir_path = os.path.join(train_path, subdir)
        if os.path.isdir(subdir_path):
            train_files = get_all_files(subdir_path)
            train_subset, _ = train_test_split(train_files, test_size=0.6, random_state=42)
            copy_files(subdir_path, os.path.join(train_dest, subdir), train_subset)

    # Move 10% of the training dataset to the test dataset folders
    print("Moving 10% of training dataset to test dataset folders...")
    for subdir in os.listdir(train_path):
        subdir_path = os.path.join(train_path, subdir)
        if os.path.isdir(subdir_path):
            train_files_remaining = get_all_files(subdir_path)
            train_subset_for_test, _ = train_test_split(train_files_remaining, test_size=0.9, random_state=42)
            for file in train_subset_for_test:
                label_dir = os.path.join(test_dest, subdir)
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)
                shutil.move(os.path.join(subdir_path, file), os.path.join(label_dir, file))

    # Create validation dataset from 10% of the remaining training dataset
    print("Creating validation dataset...")
    validation_dest = os.path.join(copy_dest, "asl_alphabet_validation")
    if not os.path.exists(validation_dest):
        os.makedirs(validation_dest)

    for subdir in os.listdir(train_path):
        subdir_path = os.path.join(train_path, subdir)
        if os.path.isdir(subdir_path):
            train_files_remaining = get_all_files(subdir_path)
            train_subset_for_validation, _ = train_test_split(train_files_remaining, test_size=0.9, random_state=42)
            move_files(subdir_path, os.path.join(validation_dest, subdir), train_subset_for_validation)

if __name__ == "__main__":
    main()
