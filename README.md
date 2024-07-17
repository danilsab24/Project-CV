# ASL Hand Gesture Detection
This repository contains code for creating a dataset for ASL hand gesture detection, training a deep learning model, and testing the trained model. The datasets used and pre-trained models are available for download at the provided links.

## Abstract
<img src="images/ASL_alphabet.jpg" alt="Hand Gesture Example" width="300"/>
Hand gesture detection is a pivotal task in the realm of human-computer interaction, particularly in recognizing the letters of the American Sign Language (ASL) alphabet from hand movements. By accurately interpreting these gestures, it becomes possible to bridge communication gaps for individuals who rely on sign language. The codes in this repository demonstrate how different models vary in their suitability for addressing this task, providing insights into the strengths and limitations of each approach. By leveraging advanced algorithms and deep learning models, we aim to develop a robust system capable of identifying ASL gestures in real-time, thereby facilitating improved learning and communication for both learners and users of sign language.

## Dataset
In this repo we have useed two dataset, the first can be downloaded from this [link](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) this Dataset has been broadly classified into Training and Testing Data. Training Data has been classified and segregated into 29 classes, of which 26 alphabets A-Z and 3 other classes of SPACE, DELETE, NOTHING.
The second dataset it was created by ourselves it contains 29 folder wich represent the letter of alphabet from A to Z plus 3 folder nothing, space and delete
### How to make dataset
This repo, as already mentioned provide the code for create dataset with personal photos. The first step is created in our enviroment a main foder and insede the main folder they must be created 29 subfolder named from A to Z (american alphabet) plus NOTHING, DELETE and SPACE. Now that we have created the folder we can run the first code [```camera_acquisition.py```](make_dataset/camera_acquisition.py), is important change the path ``` save_dir ```. the acquisition of frames start after pressing the space key and after 3 second, for stopping the acquisition you have to press q. the result will be images like this: 
<div style="display: flex; justify-content: space-between;">
  <img src="images/A_118.png" alt="Hand Gesture Example 1" width="300"/>
  <img src="images/frame_0297.png" alt="Hand Gesture Example 2" width="300"/>
</div>

After that we can use two typer of preprocessing for images if we use the code in [```detect_only_landmarks.py```](make_dataset/detect_only_landmarks.py), in this case the output images will have only landmarks and bounding box (is important change ```input_dir``` and ```output_base_dir```) the result will be:
<div style="display: flex; justify-content: space-between;">
  <img src="images/A_2_label.jpg" alt="Hand Gesture Example 1" width="300"/>
  <img src="images/frame_0034_label.jpg" alt="Hand Gesture Example 2" width="300"/>
</div>
