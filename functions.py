import os
import csv
import cv2
import pandas as pd
import numpy as np
import math

# called by save_keypoints_to_csv
def predict_keypoints(image, model):
    keypoints = model.predict(image)
    return keypoints

def save_keypoints_to_csv(folder, model, video_name, training_name):
    image_files = sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    keypoints_results = [[] for _ in range(13)]  # List to store results for each keypoint
    BBox_results = []
    for image_file in image_files:
        # Read the image file
        image_path = os.path.join(folder, image_file)
        image = cv2.imread(image_path)

        # Predict the keypoints
        keypoints = predict_keypoints(image, model)
        if keypoints[0].boxes.xywh.numel() == 0:
            BBox_results.append(['', '', '', ''])
            for keypoint_index in range(13):
                keypoints_results[keypoint_index].append(['', ''])
        else:
            BBox_results.append(keypoints[0].boxes.xywh[0, :].tolist())
            for keypoint_index in range(13):
                # Save the result of the specific keypoint
                keypoint_result = keypoints[0].keypoints[0, keypoint_index, :].tolist()
                keypoints_results[keypoint_index].append(keypoint_result)

    # Save the results to a CSV file
    csv_filename = f"predictions_{video_name}_{training_name}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        strings_list = ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
        for i in range(13):
            x_string = 'x_' + str(i)
            y_string = 'y_' + str(i)
            strings_list.append(x_string)
            strings_list.append(y_string)
        writer.writerow(strings_list)  # Add column names

        for i, _ in enumerate(image_files):
            if BBox_results[i] == ['', '', '', '']:
                row = ['', '', '', '']
            else:
                row = [BBox_results[i][0], 720 - BBox_results[i][1], BBox_results[i][2], BBox_results[i][3]]
            for keypoint_index in range(13):
                if BBox_results[i] == ['', '', '', '']:
                    row.extend(['', ''])
                    continue
                keypoint = keypoints_results[keypoint_index][i]
                try:
                    if keypoint[2] < 0.5:  # change here in case of different confidence threshold
                        keypoint[0] = ''
                        keypoint[1] = ''
                        row.extend([keypoint[0], keypoint[1]])
                    else:
                        row.extend([keypoint[0], 720 - float(keypoint[1])])
                except ValueError:
                    pass
            writer.writerow(row)
    project_directory = os.getcwd()  # Get the current working directory (project directory)
    csv_filepath = os.path.join(project_directory, csv_filename)
    print(f"Results saved to {csv_filename}")
    return csv_filepath

# a function that visualizes the results of the model on a specific video
def visualize_results(model, frames_folder, video_name, training_name):
    # Get a list of all image files in the frames folder
    image_files = [f for f in os.listdir(frames_folder) if f.endswith((".jpg", ".jpeg", ".png"))]

    # Sort the image files in ascending order
    image_files.sort()

    # Open the first image to get the dimensions
    first_image_path = os.path.join(frames_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Create a VideoWriter object to save the predicted frames as a video
    video_output_path = f"{video_name}_{training_name}.mp4"
    video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))

    # Loop through the image files
    for image_file in image_files:
        # Read the image
        image_path = os.path.join(frames_folder, image_file)
        frame = cv2.imread(image_path)

        # Run YOLOv8 inference on the frame
        results = model(frame, save = True, save_txt = True, max_det = 1)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the annotated frame to the video
        video_writer.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video writer and close the display window
    video_writer.release()
    cv2.destroyAllWindows()

    print("Video visualization complete. Output saved to:", video_output_path)

# a function that saves the true values of the keypoints to a csv file
def save_true_values(video_name, folder_path):
    project_directory = os.getcwd()  # Get the current working directory (project directory)
    csv_filename = f"true_labels_{video_name}.csv"
    csv_filepath = os.path.join(project_directory, csv_filename)

    labels_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    with open(csv_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        strings_list = ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
        for i in range(13):
            x_string = 'x_' + str(i)
            y_string = 'y_' + str(i)
            strings_list.append(x_string)
            strings_list.append(y_string)
        writer.writerow(strings_list)  # Add column names

        for filename in labels_files:
            if filename.endswith('.txt'):
                txt_filepath = os.path.join(folder_path, filename)

                with open(txt_filepath, 'r') as txtfile:
                    lines = txtfile.readlines()
                    for line in lines:
                        line_parts = line.strip().split(' ')
                        category = int(line_parts[0])
                        if line_parts[1] != 'nan':
                            bbox_x = float(line_parts[1]) * 1280
                            bbox_y = 720 - float(line_parts[2]) * 720  # Flip the y-axis
                            bbox_w = float(line_parts[3]) * 1280
                            bbox_h = float(line_parts[4]) * 720
                        row = [bbox_x, bbox_y, bbox_w, bbox_h]
                        for keypoint_index in range(13):
                            visibility = float(line_parts[7+keypoint_index*3])
                            x = float(line_parts[5+keypoint_index*3])
                            y = float(line_parts[6+keypoint_index*3])
                            if visibility == 0.0:
                                x = ''
                                y = ''
                            else:
                                x *= 1280
                                y *= 720
                                y = 720 - y  # Flip the y-axis
                            row.extend([x, y])

                        writer.writerow(row)

    print(f"CSV file '{csv_filename}' has been created in the project directory.")
    return csv_filepath


def calculate_pdj(prediction_path, labels_path, video_name, training_name, threshold=0.05):
    # Load prediction and labels CSV files
    prediction_data = pd.read_csv(prediction_path)
    labels_data = pd.read_csv(labels_path)

    # Select relevant columns
    prediction_columns = ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
    prediction_columns += [f'x_{i}' for i in range(13)]
    prediction_columns += [f'y_{i}' for i in range(13)]

    labels_columns = ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
    labels_columns += [f'x_{i}' for i in range(13)]
    labels_columns += [f'y_{i}' for i in range(13)]

    prediction_data = prediction_data[prediction_columns]
    labels_data = labels_data[labels_columns]

    # Calculate PDJ for each frame
    num_frames = len(prediction_data)
    num_keypoints = 13
    pdj_scores = []
    bbox_diagonals = []

    for frame in range(num_frames):
        pred_frame = prediction_data.iloc[frame]
        labels_frame = labels_data.iloc[frame]

        # Calculate distances between predicted and labeled keypoints
        distances = []
        num_detected = 0
        counter = 0
        bbox_diagonal = np.sqrt(labels_frame['bbox_w'] ** 2 + labels_frame['bbox_h'] ** 2)
        bbox_diagonals.append(bbox_diagonal * threshold)
        for i in range(num_keypoints):
            pred_x = pred_frame[f'x_{i}']
            pred_y = pred_frame[f'y_{i}']
            label_x = labels_frame[f'x_{i}']
            label_y = labels_frame[f'y_{i}']
            if pd.isna(label_x):
                continue
            elif pd.isna(pred_x): # if the prediction is nan and the label is not nan, we will count it as a miss
                counter += 1
                continue
            else:
                counter += 1
                distance = np.sqrt((pred_x - label_x)**2 + (pred_y - label_y)**2)
                distances.append(distance)
                if distance <= bbox_diagonal * threshold:
                    num_detected += 1


        # Calculate PDJ score for the frame
        pdj_score = num_detected / counter
        pdj_scores.append(pdj_score)

    # Calculate overall PDJ score
    overall_pdj = sum(pdj_scores) / num_frames
    print(f"Overall PDJ score for {video_name} trained with {training_name} is {overall_pdj}")
    # Save PDJ scores as a CSV file
    pdj_filename = f"PDJ_{video_name}_{training_name}.csv"
    pdj_data = pd.DataFrame({'PDJ_scores': pdj_scores})
    pdj_data.to_csv(pdj_filename, index=False)

    return overall_pdj

def calculate_oks(prediction_path, labels_path, video_name, training_name, threshold=0.02, image_size=(1280, 720)):
    # Load prediction and labels CSV files
    prediction_data = pd.read_csv(prediction_path)
    labels_data = pd.read_csv(labels_path)

    # Select relevant columns
    prediction_columns = ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
    prediction_columns += [f'x_{i}' for i in range(13)]
    prediction_columns += [f'y_{i}' for i in range(13)]

    labels_columns = ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
    labels_columns += [f'x_{i}' for i in range(13)]
    labels_columns += [f'y_{i}' for i in range(13)]

    prediction_data = prediction_data[prediction_columns]
    labels_data = labels_data[labels_columns]

    # Calculate OKS for each frame
    num_frames = len(prediction_data)
    num_keypoints = 13
    k_values = [0.05, 0.079, 0.072, 0.062, 0.079, 0.072, 0.062, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089]
    oks_scores = []

    for frame in range(num_frames):
        pred_frame = prediction_data.iloc[frame]
        labels_frame = labels_data.iloc[frame]
        # Calculate distances between predicted and labeled keypoints
        distances = []
        counter = 0
        ks_scores = []
        for i in range(num_keypoints):
            pred_x = pred_frame[f'x_{i}']
            pred_y = pred_frame[f'y_{i}']
            label_x = labels_frame[f'x_{i}']
            label_y = labels_frame[f'y_{i}']
            if pd.isna(label_x):
                continue
            elif pd.isna(pred_x): #if the prediction is NaN although the keypoint is visible, the ks score is 0
                counter += 1
                ks_scores.append(0)
            else:
                distance = np.sqrt((pred_x - label_x)**2 + (pred_y - label_y)**2)
                distances.append(distance)
                bbox_area = labels_frame['bbox_w'] * labels_frame['bbox_h']
                s = math.sqrt(bbox_area)
                ks = math.exp(-distance**2 / (2 * s**2 * k_values[i]**2))
                ks_scores.append(ks)
                counter +=1

        # Calculate OKS score for the frame
        oks_score = sum(ks_scores) / counter
        oks_scores.append(oks_score)

    # Calculate overall OKS score
    overall_oks = sum(oks_scores) / num_frames
    print(f"Overall OKS score for {video_name} trained with {training_name} is {overall_oks}")
    # Save OKS scores as a CSV file
    oks_filename = f"OKS_{video_name}_{training_name}.csv"
    oks_data = pd.DataFrame({'OKS_scores': oks_scores})
    oks_data.to_csv(oks_filename, index=False)

    return overall_oks


# prediction pipleline, a function that takes a folder with video frames and a model, plots the predictions for the video frames and returns the predicted keypoints
def prediction_pipeline(frames_folder, labels_folder, model, video_name, training_name):
    visualize_results(model, frames_folder, video_name, training_name)
    prediction_path = save_keypoints_to_csv(frames_folder, model, video_name, training_name)
    labels_path = save_true_values(video_name, labels_folder)
    calculate_pdj(prediction_path, labels_path, video_name, training_name)
    calculate_oks(prediction_path, labels_path, video_name, training_name)

