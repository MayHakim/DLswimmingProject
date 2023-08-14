from ultralytics import YOLO
import functions

# default models from Ultralytics
# model = YOLO('yolov8n-pose.pt')  # yolov8n pretrained model

# our trained models
model = YOLO('original_model.pt') # load the model trained with the original dataset
# model = YOLO('cartoon_trained_6.pt') # load the model trained on the cartoon dataset with manipulation 6
# model = YOLO('cartoon_trained_30.pt') # load the model trained on the cartoon dataset with manipulation 30

# train the model
# model.train(data='config.yaml', epochs=100, imgsz=1280)

# test the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
print(metrics)

functions.prediction_pipeline('data/images/val', 'data/labels/val', model, 'original_test', 'original_model')
