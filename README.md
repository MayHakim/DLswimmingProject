# DLswimmingProject
## Pose Estimation
### Installations
Ultralytics version 8.0.105: pip install 'ultralytics==8.0.105'
<br>
PyTorch: https://pytorch.org/get-started/locally/ 
<br>
Note: if setting the environment doesn't work, you could try opening a new project and copy the cloned files to it.
### Config
In config.yaml, set the data path to your path to the data folder in the project.
### Choosing a model
There are several options for a model. The basic model Yolov8n pretrained with Coco Dataset is 'yolov8n-pose.pt', the model that underwent additional training with the original data is 'original_model.pt', and the models trained with the manipulations are 'cartoon_trained_6.pt', 'cartoon_trained_30.pt'. These files are the updated weights of the models after training.
### Train
First, you need to set the validation images and labels for training. In the data folder, go to both folders (images and labels) and rename the folder "val" to "test" (for convenience) and the folder "validation data for training" to "val".
<br>
In order to train the model, remove the comment from the line: model.train(data='config.yaml', epochs=100, imgsz=1280).
### Test
First, make sure that in both images and labels folders under "data", the test folder is named "val".
<br>
The function prediction_pipeline saves the true labels, predictions, an output video of the prediction of the test, oks and pdj calculations in the project's folder. 
<br>
YOLO's evaluation metrics and our OKS and PDJ calculations are all printed.
### Additional data
The folder https://drive.google.com/drive/folders/1yPvzcFFob0k967MIJ25c33kzAGdV10ZF?usp=drive_link has all the images we used for our project.
<br>
The folder "manipulated images" contains all the "val" folders for all the manipulations.
<br>
The folders "manipulations for cartoonozed model - manipulation 6/30" contains the full manipulated dataset to train a new model. The folder "train" contains the train and val images folders for model training, and the "val" folder contains the maniplated test images.
## Cartoonization
Our Google Colab notebook for cartoonizations: https://colab.research.google.com/drive/1kJoYlsOmvqBZ9aTT4GxSGyiI9_SlSJb2
<br>
The first part contains imports and the original manipulation functions. This is where you upload the images you'd like to manipulate from your computer.
<br>
Under the section "Our Manipulations" there are all the manipulations we used in our project. For each manipulation- the first block sets the functions with the manipulation's parameter and the second block runs the manipulation on the uploaded images and saves them both in the notebook as folder_name and as a zip in your computer downloads.
<br>
Enjoy your cartoons!


The full paper appears under "DL_Project.pdf".
