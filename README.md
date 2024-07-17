# ASL Hand Gesture Detection
This repository contains code for creating a dataset for ASL hand gesture detection, training a deep learning model, and testing the trained model. The datasets used and pre-trained models are available for download at the provided links.

## Abstract
<img src="images/ASL_alphabet.jpg" alt="Hand Gesture Example" width="300"/>
Hand gesture detection is a pivotal task in the realm of human-computer interaction, particularly in recognizing the letters of the American Sign Language (ASL) alphabet from hand movements. By accurately interpreting these gestures, it becomes possible to bridge communication gaps for individuals who rely on sign language. The codes in this repository demonstrate how different models vary in their suitability for addressing this task, providing insights into the strengths and limitations of each approach. By leveraging advanced algorithms and deep learning models, we aim to develop a robust system capable of identifying ASL gestures in real-time, thereby facilitating improved learning and communication for both learners and users of sign language.

## Dataset
In this repo we have useed two dataset, the first can be downloaded from this [link](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) this Dataset has been broadly classified into Training and Testing Data. Training Data has been classified and segregated into 29 classes, of which 26 alphabets A-Z and 3 other classes of SPACE, DELETE, NOTHING.
The second dataset it was created by ourselves it contains 29 folder wich represent the letter of alphabet from A to Z plus 3 folder nothing, space and delete
### How to make dataset
As previously mentioned, this repository provides the code to create a dataset with personal photos. The first step is to create a main folder in your environment, and inside the main folder, you must create 29 subfolders named from A to Z (American alphabet), plus NOTHING, DELETE, and SPACE. Once the folders are created, you can run the first script [```camera_acquisition.py```](make_dataset/camera_acquisition.py), it is important to change the path ``` save_dir ```. The acquisition of frames starts after pressing the space key and waiting for 3 seconds. To stop the acquisition, press 'q'. To ensure the folders have the same number of images, you can use the script  [```elimination_image.py```](make_dataset/elimination_image.py)remember to change the path of the  ```save_dir```. The resulting images will look like this:: 
<div style="display: flex; justify-content: space-between;">
  <img src="images/A_118.png" alt="Hand Gesture Example 1" width="300"/>
  <img src="images/frame_0297.png" alt="Hand Gesture Example 2" width="300"/>
</div>

Afterward, we can apply two types of preprocessing to the images. If we use the script [```detect_only_landmarks.py```](make_dataset/detect_only_landmarks.py), the output images will contain only the landmarks and bounding boxes (it is important to change the paths of ```input_dir``` and ```output_base_dir```). The result will be:
<div style="display: flex; justify-content: space-between;">
  <img src="images/A_2_label.jpg" alt="Hand Gesture Example 1" width="300"/>
  <img src="images/frame_0034_label.jpg" alt="Hand Gesture Example 2" width="300"/>
</div>

If we use the in [```landmarks_and_remove_background.py```](make_dataset/landmarks_and_remove_background.py) (change path ```main_folder``` and ```output_folder```), in this case the result will be: 
<div style="display: flex; justify-content: space-between;">
  <img src="images/A_1.png" alt="Hand Gesture Example 1" width="300"/>
  <img src="images/frame_0273.png" alt="Hand Gesture Example 2" width="300"/>
</div>

